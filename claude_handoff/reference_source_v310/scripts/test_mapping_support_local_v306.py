"""Small native CPU fixtures; no worlds, checkpoints, transport or training."""
import copy
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from scripts import mapping_support_local_v306 as core


def private(size=128,n=1):
    x=torch.full((1,n,size,size),-1,dtype=torch.int8)
    for i in range(n):
        x[:,i,size//16:size//2,size//16:size//2]=0
        x[:,i,size//8:size//2,size//4+i:size//4+i+1]=1
    return x


def objective(heads,labels):
    labels=labels.flatten(-2);seen=labels>=0;wall=(labels==1).float()
    known=F.binary_cross_entropy_with_logits(heads['known_logits'],seen.float(),reduction='none').mean(-1)
    occupied=(F.binary_cross_entropy_with_logits(heads['wall_logits'],wall,reduction='none')*seen).sum(-1)/seen.sum(-1).clamp_min(1)
    return (known+occupied).mean()


class SupportLocalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_common_initial_function_rng_and_parameter_accounting(self):
        rng=torch.get_rng_state().clone()
        with mock.patch('torch.cuda.manual_seed_all',side_effect=AssertionError('CUDA RNG touched')):
            local=core.SupportLocal64();dense=core.SupportLocal64(arm='cross_address')
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(local.weights_hash(),dense.weights_hash())
        with torch.no_grad():
            a,b=local(private(n=2)),dense(private(n=2))
        for key in ('q','h','mu','learned_q','data_q'):
            self.assertTrue(torch.equal(a['source_factors'][key],b['source_factors'][key]),key)
        self.assertTrue(torch.equal(a['features'],b['features']))
        for name in a['decoded']:
            self.assertTrue(torch.equal(a['decoded'][name],b['decoded'][name]))
        ma,mb=local.construction_metadata(),dense.construction_metadata()
        self.assertEqual(ma['stored_parameter_count'],mb['stored_parameter_count'])
        self.assertGreater(mb['effective_parameter_count'],ma['effective_parameter_count'])
        self.assertEqual(ma['ineffective_masked_parameter_count'],16*15*(64+8))
        self.assertFalse(ma['wire_end_to_end_noninterference_claim'])

    def test_native_shapes_one_full512_no_wholemap_resize(self):
        model=core.SupportLocal64()
        for size in core.RESOLUTIONS:
            seen=[]
            hook=model.encoder[0].register_forward_pre_hook(lambda module,args:seen.append(tuple(args[0].shape)))
            with torch.no_grad():
                result=model(private(size))
            hook.remove()
            self.assertEqual(seen,[(16,2,size//4,size//4)])
            self.assertEqual(result['source_factors']['q'].shape,(1,1,64,64))
            self.assertEqual(result['features'].shape,(1,1,128))
            for value in result['decoded'].values():
                self.assertEqual(value.shape,(1,1,size*size))
                self.assertTrue(torch.isfinite(value).all())
            p=core.patches(private(size)); self.assertTrue(torch.equal(core.assemble_patches(p),private(size)))

    def test_empty_address_and_prior_once(self):
        for arm in core.ARMS:
            model=core.SupportLocal64(arm=arm)
            x=private(n=3);x[:,1].fill_(-1)
            with torch.no_grad():f=model.source_factors(x)
            mask=f['active_per_address'][...,None].expand(-1,-1,-1,4).reshape(1,3,64)
            self.assertEqual(int(torch.count_nonzero(f['data_q'][~mask])),0)
            self.assertEqual(int(torch.count_nonzero(f['h'][~mask])),0)
            p=f['q'].diagonal(dim1=-2,dim2=-1)
            self.assertTrue(torch.equal(p[~mask],torch.full_like(p[~mask],core.LATENT_PRIOR/(3*64))))
            empty=model.source_factors(torch.full_like(x,-1))
            torch.testing.assert_close(empty['q'].diagonal(dim1=-2,dim2=-1).sum(1),
                torch.full((1,64),core.LATENT_PRIOR/64,dtype=torch.float64),rtol=1e-15,atol=0)
            self.assertTrue(torch.equal(f['h'],f['data_q']*f['mu'].double()))
            self.assertTrue((f['learned_q']>=torch.tensor(core.Q_MIN)).all())
            self.assertTrue((f['learned_q']<=core.Q_MAX).all())

    def test_original_hard8_and_byte_roundtrip(self):
        model=core.SupportLocal64()
        with torch.no_grad():f=model.source_factors(private())
        self.assertEqual(f['code_bytes'].shape,(1,1,64))
        self.assertEqual(f['code_bytes'].element_size()*f['code_bytes'].numel(),64)
        self.assertFalse((f['code_bytes']==255).any())
        self.assertTrue(torch.equal(core.codes_to_latent(f['code_bytes']),f['mu']))
        points=torch.tensor([[-2.,-1.,-.5,0.,.5,1.,2.]],requires_grad=True)
        code=core.latent_quantize(points)
        self.assertTrue(torch.equal(code,core.codes_to_latent(core.latent_codes(points))))
        code.sum().backward()
        self.assertTrue(torch.equal(points.grad,torch.tensor([[0.,1.,1.,1.,1.,1.,0.]])))

    def test_encoder_factor_isolation_with_nonzero_mixer(self):
        model=core.SupportLocal64()
        with torch.no_grad():
            model.source_mixer.weight.fill_(.03)
            model.precision_head.weight.fill_(.02)
        for size in core.RESOLUTIONS:
            x=private(size);other=x.clone();side=size//4
            other[:,:,-side:,-side:]=1  # Change only address15, including its support.
            with torch.no_grad():a,b=model.source_factors(x),model.source_factors(other)
            for key in ('mu','learned_q','data_q','h'):
                self.assertTrue(torch.equal(a[key][...,:60],b[key][...,:60]),key)
            self.assertTrue(torch.equal(a['q'][...,:60,:60],b['q'][...,:60,:60]))
        # Differentiate live two-channel encoder features, not ternary comparisons.
        feature=torch.randn(1,1,16,64,requires_grad=True)
        out=model.mean_head(model.source_mixer(feature)).tanh()
        grad=torch.autograd.grad(out[...,0,:].sum(),feature)[0]
        self.assertEqual(int(torch.count_nonzero(grad[...,1:,:])),0)
        self.assertGreater(float(grad[...,0,:].abs().sum()),0)
        channels=torch.rand(16,2,32,32,requires_grad=True)
        feature=model.encoder(channels).reshape(1,1,16,64)
        out=model.mean_head(model.source_mixer(feature)).tanh()
        grad=torch.autograd.grad(out[...,0,:].sum(),channels)[0]
        self.assertEqual(int(torch.count_nonzero(grad[1:])),0)
        self.assertGreater(float(grad[0].abs().sum()),0)

    def test_readout_isolation_and_cross_address_control_has_live_links(self):
        local=core.SupportLocal64();dense=core.SupportLocal64(arm='cross_address')
        with torch.no_grad():
            local.readout_mixer.weight.fill_(.03)
            dense.readout_mixer.weight[0,15].fill_(.2)
            dense.source_mixer.weight[0,15].fill_(.2)
        feature=torch.cat((core.latent_quantize(torch.randn(1,1,64)),torch.zeros(1,1,64)),-1)
        changed=feature.clone();changed[...,60:64]=1.;changed[...,124:128]=2.
        for size in core.RESOLUTIONS:
            with torch.no_grad():a=local.decode(feature,resolution=size);b=local.decode(changed,resolution=size)
            for key in a:
                pa=core.patches(a[key].reshape(1,1,size,size));pb=core.patches(b[key].reshape(1,1,size,size))
                self.assertTrue(torch.equal(pa[...,:15,:,:],pb[...,:15,:,:]))
        feature=feature.requires_grad_()
        target=core.patches(local.decode(feature)['wall_logits'].reshape(1,1,128,128))[...,0,:,:].sum()
        grad=torch.autograd.grad(target,feature)[0]
        allowed=torch.zeros(128,dtype=torch.bool);allowed[:4]=True;allowed[64:68]=True
        self.assertEqual(int(torch.count_nonzero(grad[...,~allowed])),0)
        self.assertGreater(float(grad[...,allowed].abs().sum()),0)
        with torch.no_grad():a=dense.decode(feature);b=dense.decode(changed)
        self.assertFalse(torch.equal(a['wall_logits'][...,:32],b['wall_logits'][...,:32]))
        x=private();other=x.clone();other[:,:,-32:,-32:]=1
        with torch.no_grad():a=dense.source_factors(x);b=dense.source_factors(other)
        self.assertFalse(torch.equal(a['mu'][...,:4],b['mu'][...,:4]))

    def test_full_gradients_and_masked_parameter_freedom(self):
        offdiag=~torch.eye(16,dtype=torch.bool)
        for arm in core.ARMS:
            model=core.SupportLocal64(arm=arm);x=private(n=2)
            result=model(x);objective(result['decoded'],x).backward()
            for name,group in model.parameter_groups().items():
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in group),name)
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in group),0,name)
            for mixer in (model.source_mixer,model.readout_mixer):
                value=mixer.weight.grad[offdiag]
                if arm=='support_local':self.assertEqual(int(torch.count_nonzero(value)),0)
                else:self.assertGreater(float(value.abs().sum()),0)
            model.train();self.assertFalse(any(module.training for module in model.modules()))

    def test_agent_permutation_no_cross_agent_normalization(self):
        order=torch.tensor([2,0,1]);x=private(n=3)
        for arm in core.ARMS:
            model=core.SupportLocal64(arm=arm)
            with torch.no_grad():a=model(x);b=model(x[:,order])
            for key in ('q','h','mu','data_q'):
                self.assertTrue(torch.equal(a['source_factors'][key][:,order],b['source_factors'][key]),key)
            for key in a['decoded']:
                self.assertTrue(torch.equal(a['decoded'][key][:,order],b['decoded'][key]))

    def test_typed_restore_invalid_shapes_and_no_private_decoder_api(self):
        model=core.SupportLocal64();payload=model.to_payload()
        restored=core.SupportLocal64.from_payload(payload)
        self.assertEqual(model.weights_hash(),restored.weights_hash())
        with torch.no_grad():a=model(private());b=restored(private())
        self.assertTrue(torch.equal(a['decoded']['wall_logits'],b['decoded']['wall_logits']))
        bad=copy.deepcopy(payload);bad['construction']['total_dimensions']['mean']=1024
        with self.assertRaises(ValueError):core.SupportLocal64.from_payload(bad)
        for x in (torch.zeros(1,1,96,96),torch.zeros(1,17,128,128),torch.full((1,1,128,128),2)):
            with self.assertRaises(ValueError):model(x)
        feature=a['features'].detach().clone();feature[...,0]=.123456
        with self.assertRaises(ValueError):model.decode(feature)
        feature=a['features'].detach().clone();feature[...,100]=float('nan')
        with self.assertRaises(ValueError):model.decode(feature)
        with self.assertRaises(TypeError):model.decode(a['features'],private=private())
        with self.assertRaises(ValueError):core.belief_features(torch.zeros(1,64),torch.zeros(1,64))


if __name__=='__main__':
    unittest.main()
