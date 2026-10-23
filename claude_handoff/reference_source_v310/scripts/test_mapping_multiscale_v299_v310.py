"""Synthetic CPU adapter contracts; no actual checkpoint, training, or GPU."""
import copy
import json
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from scripts import mapping_multiscale_v299_v310 as core
from scripts import mapping_content_readout_v288 as content
from scripts import mapping_common_map_v284 as common
from scripts import mapping_multiroom_forest_v295 as forest
from scripts.mapping_implicit_model_v192 import latent_quantize
from scripts.mapping_support_training_v306 import proper_terms
from scripts import test_mapping_content_readout_v288 as fixture


def private(size=128,n=1):
    x=torch.full((1,n,size,size),-1,dtype=torch.int8)
    for node in range(n):
        x[:,node,size//8:size//2,size//8:size//2]=0
        x[:,node,size//4:size//2,size//4+node:size//4+node+1]=1
    return x


def own_features(model,x):
    f=model.source.factors(x);p=f['q'].diagonal(dim1=-2,dim2=-1)
    return common.belief_features(latent_quantize((f['h']/p).float()),torch.diag_embed(p.reciprocal()))


class MultiscaleV299Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);fixture.ContentReadoutTests.setUpClass()
        original=content.ContentReadout64.from_parent_payload(fixture.ContentReadoutTests.payload,arm='joint_belief')
        cls.payload=dict(construction=original.construction_metadata(),
            state_dict=copy.deepcopy(original.state_dict()),model_sha256=original.weights_hash())
        cls.features=fixture.ContentReadoutTests.features.clone()

    def parent(self):return content.ContentReadout64.from_payload(self.payload)
    def model(self):return core.MultiscaleV299.from_parent_payload(self.payload)

    def test_exact_keys_weights_flags_buffers_rng_and_metadata(self):
        rng=torch.get_rng_state().clone()
        with mock.patch('torch.cuda.manual_seed_all',side_effect=AssertionError('CUDA touched')):
            parent=self.parent();model=self.model()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(list(parent.state_dict()),list(model.state_dict()))
        self.assertEqual(parent.weights_hash(),model.weights_hash())
        self.assertEqual(parent.frozen_unused_hash(),model.frozen_unused_hash())
        self.assertEqual([(n,p.requires_grad) for n,p in parent.named_parameters()],
            [(n,p.requires_grad) for n,p in model.named_parameters()])
        self.assertEqual(list(parent.parameter_groups()),list(model.parameter_groups()))
        for name,value in parent.state_dict().items():self.assertTrue(torch.equal(value,model.state_dict()[name]),name)
        meta=model.construction_metadata();json.dumps(meta,allow_nan=False)
        self.assertEqual(meta['trainable_parameter_count'],1387970)
        self.assertEqual(meta['total_parameters'],2048932)
        self.assertEqual(meta['new_parameter_count'],0);self.assertEqual(meta['new_persistent_buffer_count'],0)
        self.assertFalse(meta['input_raster_resized']);self.assertFalse(meta['final_logits_resized'])
        self.assertTrue(meta['high_resolution_receptive_field_schedule_changed'])
        self.assertEqual(meta['registered_precision_profile'],4)

    def test_original128_forward_factors_and_all_group_gradients_bitwise(self):
        parent=self.parent();model=self.model();x=private()
        a=parent.source.factors(x);b=model.source.factors(x)
        for key in a:self.assertTrue(torch.equal(a[key],b[key]) if isinstance(a[key],torch.Tensor) else a[key]==b[key],key)
        af=own_features(parent,x);bf=own_features(model,x)
        self.assertTrue(torch.equal(af,bf))
        expected=parent.decode(af);actual=model.decode(bf,resolution=128)
        for name in expected:self.assertTrue(torch.equal(expected[name],actual[name]),name)
        proper_terms(expected,x)['total'].backward();proper_terms(actual,x)['total'].backward()
        for (name,p),(other,q) in zip(parent.named_parameters(),model.named_parameters()):
            self.assertEqual(name,other)
            if p.grad is None:self.assertIsNone(q.grad)
            else:
                self.assertTrue(torch.equal(p.grad,q.grad),name)
                self.assertTrue(torch.isfinite(q.grad).all(),name)
        for name,group in model.parameter_groups().items():
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in group),0,name)

    def test_native_source_pixels_first_layer_and_empty_prior_all_sizes(self):
        model=self.model()
        for size in core.RESOLUTIONS:
            seen=[]
            handle=model.source.capacity.encoder[0].register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().clone()))
            x=private(size,n=2);x[:,1].fill_(-1)
            with torch.no_grad(),mock.patch.object(core.F,'interpolate',side_effect=AssertionError('Raw input resized')):
                f=model.source.factors(x)
            handle.remove()
            self.assertEqual(seen[0].shape,(2,5,size,size))
            self.assertTrue(torch.equal(seen[0][:,0],(x.reshape(2,size,size)>=0).float()))
            self.assertTrue(torch.equal(seen[0][:,2],(x.reshape(2,size,size)==1).float()))
            self.assertEqual(f['mu'].shape,(1,2,64));self.assertTrue(torch.isfinite(f['q']).all())
            self.assertEqual(f['q'].shape,(1,2,64,64));self.assertEqual(f['h'].shape,(1,2,64))
            self.assertTrue(torch.equal(f['h'][:,1],torch.zeros_like(f['h'][:,1])))
            self.assertTrue(torch.equal(f['q'][:,1].diagonal(dim1=-2,dim2=-1),
                torch.full_like(f['mu'][:,1],core.recognition.LATENT_PRIOR/(2*64),dtype=torch.float64)))
        self.assertEqual(model.source.capacity.public_xy.shape,(2,128,128))
        self.assertEqual(model.weights_hash(),self.payload['model_sha256'])

    def test_native_decoder_feature_schedule_not_logit_upsampling(self):
        model=self.model();features=self.features[:,:1]
        for size in (256,512):
            block_shapes=[];output_shapes=[];interpolation_channels=[]
            handles=[block.register_forward_pre_hook(lambda module,args:block_shapes.append(tuple(args[0].shape)))
                for block in model.readout.decoder.blocks]
            handles.append(model.readout.decoder.output.register_forward_pre_hook(
                lambda module,args:output_shapes.append(tuple(args[0].shape))))
            original=F.interpolate
            def watch(value,*args,**kwargs):
                interpolation_channels.append(value.shape[1])
                return original(value,*args,**kwargs)
            with torch.no_grad(),mock.patch.object(core.F,'interpolate',side_effect=watch):
                result=model.decode(features,resolution=size)
            for handle in handles:handle.remove()
            self.assertEqual([s[-1] for s in block_shapes],list(core.decoder_extents(size))*2)
            self.assertEqual(output_shapes,[(1,32,size,size)]*2)
            self.assertEqual(interpolation_channels,[128,96,64,48]*2)
            self.assertEqual(set(result),{'known_logits','wall_logits'})
            for value in result.values():
                self.assertEqual(value.shape,(1,1,size*size));self.assertTrue(torch.isfinite(value).all())
        # Explicit public resolution has no private-input-dependent state.
        with torch.no_grad():
            first=model.decode(features,resolution=128)
            model.source.factors(private(512))
            second=model.decode(features,resolution=128)
        for name in first:self.assertTrue(torch.equal(first[name],second[name]))

    def test_recipient_slicing_independence_and_finite_native_gradients(self):
        model=self.model();features=self.features.clone().requires_grad_(True)
        with torch.no_grad():
            joint=model.decode(features,resolution=256)
            slices=[model.decode_recipients(features,resolution=256,start=i,stop=i+1) for i in range(2)]
        for name in joint:
            self.assertTrue(torch.allclose(joint[name],torch.cat([s[name] for s in slices],1),rtol=2e-5,atol=2e-6))
        selected=model.decode_recipients(features,resolution=256,start=1,stop=2)
        sum(value.square().mean() for value in selected.values()).backward()
        self.assertEqual(int(torch.count_nonzero(features.grad[:,0])),0)
        self.assertGreater(float(features.grad[:,1].abs().sum()),0)
        for name in ('lift','decoder'):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameter_groups()[name]))
        source=self.model();f=source.source.factors(private(512))
        (f['h'].square().mean()+f['q'].diagonal(dim1=-2,dim2=-1).mean()).backward()
        for name in ('encoder','mean_head','precision_head'):
            group=source.parameter_groups()[name]
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in group),name)
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in group),0,name)

    def test_unchanged_forest_packets_at128_and_native256_received_audit(self):
        model=self.model();parent=self.parent();edges=[(0,1,1.)];contexts={(0,1):bytes(range(16))}
        with torch.no_grad():
            original=forest.rollout(parent,private(128,2),edges,rounds=1,decode_rounds=(),serialize=True,edge_contexts=contexts)
            adapted=forest.rollout(model,private(128,2),edges,rounds=1,decode_rounds=(),serialize=True,edge_contexts=contexts)
            native=forest.rollout(model,private(256,2),edges,rounds=1,decode_rounds=(),serialize=True,edge_contexts=contexts)
        self.assertEqual(original['packets'],adapted['packets'])
        for key in ('means','variances','features'):
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(original[key],adapted[key])),key)
        self.assertEqual(native['decoded'],{})
        self.assertTrue(forest.audit_rollout(native,edges,contexts)['passed'])
        self.assertEqual([len(p) for p in native['packets']],[172,172])

    def test_strict_payload_roundtrip_after_weight_update_and_metadata(self):
        model=self.model();saved=model.to_payload();restored=core.MultiscaleV299.from_payload(saved)
        self.assertEqual(restored.weights_hash(),model.weights_hash())
        with torch.no_grad():
            a=model.decode(self.features[:,:1],resolution=128);b=restored.decode(self.features[:,:1],resolution=128)
        for name in a:self.assertTrue(torch.equal(a[name],b[name]))
        with torch.no_grad():model.readout.lift.bias.add_(.001)
        changed=model.to_payload();resumed=core.MultiscaleV299.from_payload(changed)
        self.assertEqual(model.weights_hash(),resumed.weights_hash())
        self.assertEqual(resumed.continuation_parent_state_sha256,self.payload['model_sha256'])
        self.assertEqual(model.construction_metadata(),resumed.construction_metadata())
        self.assertEqual([(n,p.requires_grad) for n,p in model.named_parameters()],
            [(n,p.requires_grad) for n,p in resumed.named_parameters()])
        edits=(lambda p:p['construction'].update(decoder_feature_grid_schedule='16,32,64,128'),
            lambda p:p['construction'].update(final_logits_resized=True),
            lambda p:p.update(model_sha256='0'*64),
            lambda p:p['state_dict'].update({'readout.lift.bias':p['state_dict']['readout.lift.bias'].double()}))
        for edit in edits:
            corrupt=copy.deepcopy(saved);edit(corrupt)
            with self.assertRaises(ValueError):core.MultiscaleV299.from_payload(corrupt)
        with self.assertRaises(ValueError):core.MultiscaleV299.from_payload(self.payload)
        wrong=self.parent();wrong.arm='mean_wall'
        with self.assertRaises(ValueError):core.MultiscaleV299(wrong)

    def test_invalid_native_inputs_and_public_resolution_rejected(self):
        model=self.model()
        for size in (64,129,1024,True):
            with self.assertRaises(ValueError):model.decode(self.features,resolution=size)
        for x in (private(64),private().bool(),private()[:,:,:,:127],private(256).float()*2):
            with self.assertRaises(ValueError):model.source.factors(x)
        nonhard=self.features.clone();nonhard[...,0]=.123456
        for features in (self.features.double(),self.features[...,:127],self.features*float('nan'),nonhard):
            with self.assertRaises(ValueError):model.decode(features,resolution=256)
        with self.assertRaises(ValueError):model.decode_recipients(self.features,resolution=256,start=1,stop=1)
        with self.assertRaises(TypeError):model.decode(self.features,resolution=256,private=private(256))


if __name__=='__main__':unittest.main()
