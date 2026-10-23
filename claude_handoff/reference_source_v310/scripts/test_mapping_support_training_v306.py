"""CPU parity of scheduling, not model efficacy or an optimizer run admission."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import unittest

os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import torch
from torch import nn
from torch.nn import functional as F

spec=importlib.util.spec_from_file_location('support_training306_test',Path(__file__).with_name('mapping_support_training_v306.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class TinyReadout(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(3069);self.linear=nn.Linear(128,2)
    def decode(self,features,*,resolution):
        heads=self.linear(features)
        # All native pixels are evaluated; no mask/target can enter decode.
        axis=torch.linspace(-.2,.2,resolution*resolution,device=features.device)
        return {key:heads[...,i,None]+axis for i,key in enumerate(('known_logits','wall_logits'))}


def private_pair():
    truth=torch.zeros((128,128),dtype=torch.int8);truth[::13,:]=1;truth[:,::17]=1
    private=truth[None,None].expand(1,2,-1,-1).clone()
    private[:,0,:,80:]=-1;private[:,1,:,:48]=-1;private[:,:,-24:,:]=-1
    return private


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_actual_both_arms_all_six_groups_and_one_adamw_update(self):
        from scripts.mapping_support_local_v306 import SupportLocal64
        from scripts.mapping_support_forest_v306 import rollout
        private=private_pair();union=private.max(1,keepdim=True).values.expand_as(private)
        targets={k:private if k==0 else union for k in m.STAGES}
        for arm in ('support_local','cross_address'):
            reference=SupportLocal64(arm=arm);staged=copy.deepcopy(reference)
            before=reference.weights_hash()
            a=rollout(reference,private,[(0,1,1.)],features_only=True,trace=False)
            b=rollout(staged,private,[(0,1,1.)],features_only=True,trace=False)
            original=[v.detach().clone() for v in b['features']]
            loss,ra=m.monolithic_objective(reference,{k:a['features'][k] for k in m.STAGES},targets)
            loss.backward();rb=m.backward_staged(staged,{k:b['features'][k] for k in m.STAGES},targets)
            self.assertEqual(reference.weights_hash(),before);self.assertEqual(staged.weights_hash(),before)
            self.assertAlmostEqual(ra['total'],rb['total'],places=7);self.assertEqual(rb['source_backward_calls'],1)
            self.assertEqual(rb['decoder_backward_calls'],5);self.assertFalse(rb['optimizer_step_performed'])
            self.assertEqual(json.loads(json.dumps(rb,allow_nan=False))['total'],rb['total'])
            for old,value in zip(original,b['features']):self.assertTrue(torch.equal(old,value))
            self.assertEqual(set(reference.parameter_groups()),{'encoder','source_mixer','mean_head','precision_head','readout_mixer','decoder'})
            for group,params in reference.parameter_groups().items():
                other=staged.parameter_groups()[group]
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params+other),group)
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in params),0.,group)
                for p,q in zip(params,other):torch.testing.assert_close(p.grad,q.grad,**m.PARITY_TOLERANCE,msg=group)
            # The helper itself never steps; this explicitly compares one TEST update.
            opta=torch.optim.AdamW(reference.parameters(),lr=5e-5,weight_decay=0.)
            optb=torch.optim.AdamW(staged.parameters(),lr=5e-5,weight_decay=0.)
            opta.step();optb.step()
            for p,q in zip(reference.parameters(),staged.parameters()):torch.testing.assert_close(p,q,**m.PARITY_TOLERANCE)

    def test_arbitrary_live_feature_chain_accumulates_existing_grad_all_native_sizes(self):
        for size in (128,256,512):
            a=TinyReadout();b=copy.deepcopy(a)
            x=torch.linspace(-.3,.4,128).reshape(1,1,128).requires_grad_();y=x.detach().clone().requires_grad_()
            fa={k:x+(.1+k*.01)*x.sin() for k in m.STAGES};fb={k:y+(.1+k*.01)*y.sin() for k in m.STAGES}
            target=torch.zeros((1,1,size,size),dtype=torch.int8);target[...,::5,:]=1;target[...,:3,:]=-1
            targets={k:target for k in m.STAGES}
            for p,q in zip(a.parameters(),b.parameters()):p.grad=torch.full_like(p,.003);q.grad=p.grad.clone()
            x.grad=torch.full_like(x,.003);y.grad=x.grad.clone()
            loss,ra=m.monolithic_objective(a,fa,targets);loss.backward()
            rb=m.backward_staged(b,fb,targets)
            torch.testing.assert_close(x.grad,y.grad,**m.PARITY_TOLERANCE)
            self.assertGreater(float((y.grad-.003).abs().sum()),0.)
            for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,**m.PARITY_TOLERANCE)
            self.assertAlmostEqual(ra['total'],rb['total'],places=7)
            self.assertEqual(rb['stages'][8]['observed_counts'][0][0],size*(size-3))

    def test_proper_objective_recipient_not_global_pixel_normalization(self):
        target=torch.full((1,2,128,128),-1,dtype=torch.int8);target[:,0,0,0]=1;target[:,1]=0
        known=torch.zeros((1,2,128*128),requires_grad=True)
        wall=torch.zeros_like(known).detach();wall[:,0]=2.;wall[:,1]=-1.;wall.requires_grad_()
        terms=m.proper_terms({'known_logits':known,'wall_logits':wall},target)
        expected=.5*(F.softplus(torch.tensor(-2.))+F.softplus(torch.tensor(-1.)))
        torch.testing.assert_close(terms['wall'],expected,rtol=1e-6,atol=1e-7)
        terms['total'].backward()
        unknown=target.flatten(-2)<0
        self.assertTrue(torch.equal(wall.grad[unknown],torch.zeros_like(wall.grad[unknown])))
        changed=wall.detach().clone();changed[unknown]=1000.
        again=m.proper_terms({'known_logits':known.detach(),'wall_logits':changed},target)
        self.assertTrue(torch.equal(terms['wall'],again['wall']))
        empty=torch.full_like(target,-1);emptywall=wall.detach().clone().requires_grad_()
        zero=m.proper_terms({'known_logits':known.detach(),'wall_logits':emptywall},empty)['wall'];zero.backward()
        self.assertEqual(float(zero.detach()),0.);self.assertTrue(torch.equal(emptywall.grad,torch.zeros_like(emptywall)))

    def test_stage_validation_rejects_missing_detached_malformed_nonfinite_before_backward(self):
        model=TinyReadout();f=torch.zeros((1,1,128),requires_grad=True);t=torch.zeros((1,1,128,128),dtype=torch.int8)
        features={k:f for k in m.STAGES};targets={k:t for k in m.STAGES}
        cases=[]
        missing=dict(features);missing.pop(8);cases.append((missing,targets))
        detached=dict(features);detached[8]=f.detach();cases.append((detached,targets))
        bad=dict(features);bad[4]=(f*float('nan'));cases.append((bad,targets))
        boolean={False:f,1:f,2:f,4:f,8:f};cases.append((boolean,targets))
        badt=dict(targets);badt[8]=t.float();cases.append((features,badt))
        badt=dict(targets);badt[8]=torch.full_like(t,2);cases.append((features,badt))
        badt=dict(targets);badt[8]=torch.zeros((1,1,256,256),dtype=torch.int8);cases.append((features,badt))
        for fs,ts in cases:
            with self.assertRaises(ValueError):m.backward_staged(model,fs,ts)
        self.assertIsNone(f.grad);self.assertTrue(all(p.grad is None for p in model.parameters()))
        with torch.no_grad():
            with self.assertRaises(ValueError):m.backward_staged(model,features,targets)

    def test_head_schema_nonfinite_dtype_and_shape_fail_closed(self):
        target=torch.zeros((1,1,128,128),dtype=torch.int8);valid=torch.zeros((1,1,128*128),requires_grad=True)
        heads={'known_logits':valid,'wall_logits':valid}
        variants=[dict(heads,mask=valid),dict(heads,known_logits=valid.double()),dict(heads,wall_logits=valid[...,:5]),
            dict(heads,wall_logits=valid*float('inf'))]
        for bad in variants:
            with self.assertRaises(ValueError):m.proper_terms(bad,target)
        with self.assertRaises(ValueError):m.proper_terms(heads,target.bool())


if __name__=='__main__':unittest.main()
