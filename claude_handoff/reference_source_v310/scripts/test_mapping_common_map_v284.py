"""CPU-only actual math, codec and joint-gradient contracts; synthetic weights."""
import copy
import unittest
from unittest import mock

import numpy as np
import torch

from scripts import mapping_common_map_v284 as core
from scripts import mapping_belief_readout_v282 as old_readout
from scripts import mapping_observed_wire_v246 as old_wire
from scripts.mapping_implicit_model_v192 import latent_quantize
from scripts import test_mapping_belief_readout_v282 as fixture
from scripts.test_mapping_recognition_hops_v277 import maps


class CommonMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        fixture.BeliefReadoutTests.setUpClass()
        cls.source_payload=copy.deepcopy(fixture.BeliefReadoutTests.payload)
        readout=old_readout.BeliefReadout64.from_parent_payload(
            cls.source_payload,fixture.BeliefReadoutTests.norm,'posterior_diagonal')
        # Emulate an already-trained uncertainty readout, not a zero-C fixture.
        with torch.no_grad():
            readout.lift.weight[:,64:].copy_(torch.linspace(-.001,.001,2048*64).reshape(2048,64))
        cls.readout_payload=dict(construction=readout.construction_metadata(),state_dict=copy.deepcopy(readout.state_dict()))

    def model(self):
        return core.CommonMap64.from_parent_payloads(self.source_payload,self.readout_payload)

    def factors(self,n=3):
        p=torch.linspace(.1,3.,n*64,dtype=torch.float64).reshape(1,n,64)
        mu=torch.sin(torch.arange(n*64,dtype=torch.float64)).reshape(1,n,64)*.7
        return torch.diag_embed(p),p*mu

    def test_identical_initial_state_original_readout_and_freeze_ownership(self):
        rng=torch.get_rng_state().clone()
        with mock.patch('torch.cuda.manual_seed_all',side_effect=AssertionError('No CUDA')):
            first,second=self.model(),self.model()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(first.weights_hash(),second.weights_hash())
        self.assertEqual(set(first.parameter_groups()),{'encoder','mean_head','precision_head','lift','decoder'})
        active=first.trainable_parameters()
        self.assertEqual(len(active),len({id(p) for p in active}))
        for name,p in first.named_parameters():
            self.assertEqual(p.requires_grad,not name.startswith(('source.capacity.lift.','source.capacity.decoder.')))
        features=fixture.BeliefReadoutTests.normalization_features[:2].reshape(1,2,128)
        original=old_readout.BeliefReadout64.from_payload(self.readout_payload)
        with torch.no_grad():
            a,b=first.decode(features),original(features)
        for key in a:self.assertTrue(torch.equal(a[key],b[key]),key)
        bad=copy.deepcopy(self.readout_payload)
        bad['construction']['source_state_sha256']='b'*64
        with self.assertRaises(ValueError):core.CommonMap64.from_parent_payloads(self.source_payload,bad)

    def test_float_tree_exact_mean_and_covariance_for_each_objective(self):
        for n,edges,rounds in ((2,[(0,1,1.)],1),(3,[(0,1,1.),(1,2,1.)],2)):
            q,h=self.factors(n)
            for mode in core.ARMS:
                state=core.initialize(q,h,edges)
                result=core.advance(state,rounds,mode=mode,quantized=False)
                if mode=='equality':
                    precision=q.diagonal(dim1=-2,dim2=-1).sum(1)
                    expected_mean=(h.sum(1)/precision)[:,None].expand(-1,n,-1)
                    expected_cov=torch.diag_embed(precision.reciprocal())[:,None].expand(-1,n,-1,-1)
                else:
                    full=torch.block_diag(*q[0].unbind())
                    for i,j,w in edges:
                        eye=torch.eye(64,dtype=torch.float64)*w
                        full[i*64:(i+1)*64,i*64:(i+1)*64]+=eye
                        full[j*64:(j+1)*64,j*64:(j+1)*64]+=eye
                        full[i*64:(i+1)*64,j*64:(j+1)*64]-=eye
                        full[j*64:(j+1)*64,i*64:(i+1)*64]-=eye
                    covariance=torch.linalg.inv(full)
                    expected_mean=torch.linalg.solve(full,h.flatten()).reshape(1,n,64)
                    expected_cov=torch.stack([covariance[i*64:(i+1)*64,i*64:(i+1)*64] for i in range(n)])[None]
                torch.testing.assert_close(result['mean'],expected_mean,rtol=2e-13,atol=2e-13)
                torch.testing.assert_close(result['covariance'],expected_cov,rtol=2e-13,atol=2e-13)
                self.assertIs(result['state']['q'],q);self.assertIs(result['state']['h'],h)
            # The two objectives are deliberately different, not a solver tweak.
            soft=core.advance(core.initialize(q,h,edges),rounds,mode='finite_beta',quantized=False)
            common=core.advance(core.initialize(q,h,edges),rounds,mode='equality',quantized=False)
            self.assertGreater((soft['mean']-common['mean']).abs().max().item(),.01)

    def test_quantizer_literal_bytes_gradients_profiles_and_failclosed(self):
        for profile,upper in ((2,1.),(3,16.)):
            p=torch.logspace(-7,np.log10(upper),64,dtype=torch.float64)[None].requires_grad_()
            p=p.clamp_max(upper)
            h=(torch.linspace(-.4,.4,64,dtype=torch.float64)[None]*p.sqrt()).detach().requires_grad_()
            hp,hh,_=core.quantize(p,h,precision_profile=profile)
            raw,_=core.CommonMapCodec64.encode(p[0].detach().numpy(),h[0].detach().numpy(),0,1,0,
                mode='standardized',precision_profile=profile)
            d=core.CommonMapCodec64.decode(raw)
            self.assertEqual(len(raw),148)
            self.assertTrue(torch.equal(hp[0].detach(),torch.from_numpy(d['precision'].astype(np.float32)).double()))
            self.assertTrue(torch.equal(hh[0].detach(),torch.from_numpy(d['eta'].astype(np.float32)).double()))
            gp,gh=torch.autograd.grad(hp.sum()+hh.sum(),(p,h))
            self.assertTrue(torch.equal(gp,torch.ones_like(p)))
            self.assertTrue(torch.equal(gh,torch.ones_like(h)))
        for p,h in ((torch.full((1,64),16.1),torch.zeros(1,64)),
                    (torch.full((1,64),1e-9),torch.zeros(1,64)),
                    (torch.ones(1,64),torch.full((1,64),257.)),
                    (torch.zeros(1,64),torch.ones(1,64))):
            with self.assertRaises(ValueError):core.quantize(p.double(),h.double())
        p=torch.ones(1,64,dtype=torch.float64)*.2;h=torch.zeros_like(p)
        a=core.quantize(p,h,precision_profile=2);b=old_wire.quantize(p,h)
        self.assertTrue(torch.equal(a[0],b[0]));self.assertTrue(torch.equal(a[1],b[1]))

    def test_actual_packets_receive_duplicate_stale_context_and_source_immutability(self):
        model=self.model()
        for n,edges in ((2,[(0,1,1.)]),(3,[(0,1,1.),(1,2,1.)])):
            contexts={(i,j):(i+1).to_bytes(16,'big') for i,j,_ in edges}
            for mode in core.ARMS:
                with torch.no_grad():
                    result=core.rollout(model,maps(n),edges,mode=mode,serialize=True,edge_contexts=contexts)
                    fast=core.rollout(model,maps(n),edges,mode=mode)
                audit=core.audit_rollout(result,edges,contexts)
                self.assertTrue(audit['passed'])
                self.assertEqual(audit['actual_data_bytes'],172*4*(n-1))
                self.assertEqual(audit['additional_forced_replay_bytes'],audit['actual_data_bytes'])
                for hop in range(3):
                    for key in ('codes','means','covariances','features'):
                        self.assertTrue(torch.equal(result[key][hop],fast[key][hop]),(mode,hop,key))
                state=result['state'];packet=result['packets'][-1]
                corrupt=bytes([packet[0]^1])+packet[1:]
                after,accepted=core.receive(state,corrupt)
                self.assertFalse(accepted);self.assertIs(after,state)
                decoded=core.CommonMapCodec64.decode(packet[24:])
                raw,_=core.CommonMapCodec64.encode(np.ones(64)*.1,np.zeros(64),decoded['sender'],decoded['receiver'],decoded['sequence'],
                    mode='standardized',precision_profile=3)
                with self.assertRaises(ValueError):core.receive(state,packet[:24]+raw)

    def test_true_two_hop_delay_and_empty_original_source(self):
        model=self.model();edges=[(0,1,1.),(1,2,1.)]
        own=maps(3);changed=own.clone();changed[:,0,:64]=1
        for mode in core.ARMS:
            with torch.no_grad():
                a=core.rollout(model,own,edges,mode=mode)
                b=core.rollout(model,changed,edges,mode=mode)
            for hop in (0,1):self.assertTrue(torch.equal(a['means'][hop][:,2],b['means'][hop][:,2]))
            self.assertFalse(torch.equal(a['means'][2][:,2],b['means'][2][:,2]))
            factors=a['source_factors'];prior=factors['prior_precision']
            self.assertTrue(torch.equal(factors['q'][:,1],torch.eye(64,dtype=torch.float64)[None]*prior))
            self.assertEqual(torch.count_nonzero(factors['h'][:,1]).item(),0)

    def test_end_to_end_and_covariance_only_gradient_paths(self):
        for mode in core.ARMS:
            model=self.model();frozen=model.frozen_unused_hash()
            result=core.rollout(model,maps(3),[(0,1,1.),(1,2,1.)],mode=mode)
            loss=sum(v.square().mean() for v in result['decoded'][-1].values())
            loss.backward()
            for group,parameters in model.parameter_groups().items():
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters),group)
                self.assertGreater(sum(p.grad.abs().sum().item() for p in parameters),0,group)
            self.assertEqual(frozen,model.frozen_unused_hash())
            self.assertTrue(all(p.grad is None for name,p in model.named_parameters() if not p.requires_grad))
            # Hold mean completely detached: log C must still train precision.
            model.zero_grad(set_to_none=True)
            factors=model.source.factors(maps(3))
            state=core.advance(core.initialize(factors['q'],factors['h'],[(0,1,1.),(1,2,1.)]),2,mode=mode)['state']
            mean,covariance=core.belief(state)
            feature=core.belief_features(latent_quantize(mean.float()).detach(),covariance)
            loss=model.decode(feature)['known_logits'].square().mean()
            gradient=torch.autograd.grad(loss,model.source.precision_head.weight)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(),0)

    def test_selfcontained_restore_and_invalid_modes_diagonal_profile(self):
        model=self.model()
        # Restore genuinely changed active weights without treating historical
        # v282 source provenance as a claim that current trained weights match.
        with torch.no_grad():
            model.source.precision_head.bias.add_(.01)
            model.readout.lift.bias.add_(.001)
        saved=dict(construction=model.construction_metadata(),state_dict=copy.deepcopy(model.state_dict()))
        with mock.patch('torch.load',side_effect=AssertionError('No ancestor reads')):
            restored=core.CommonMap64.from_payload(saved)
        self.assertEqual(model.weights_hash(),restored.weights_hash())
        self.assertEqual(model.construction_metadata(),restored.construction_metadata())
        q,h=self.factors(2)
        q=q.clone();q[...,0,1]=.01
        with self.assertRaises(ValueError):core.initialize(q,h,[(0,1,1.)])
        for kwargs in (dict(mode='equality',damping=.5),dict(mode='equality',precision_profile=2),dict(mode='other')):
            with self.assertRaises(ValueError):core.rollout(model,maps(2),[(0,1,1.)],**kwargs)

    def test_old_profile_schedule_diagnostics_and_no_fresh_evidence_from_resweeps(self):
        model=self.model();edges=[(0,1,1.),(1,2,1.)]
        contexts={(0,1):b'a'*16,(1,2):b'b'*16}
        # Actual original source implementation is frozen for the old reference.
        source=core.RecognitionHops64.from_payload(self.source_payload).requires_grad_(False).eval()
        with torch.no_grad():
            original=old_readout.belief_rollout(source,maps(3),edges,serialize=True,edge_contexts=contexts)
            revised=core.rollout(model,maps(3),edges,damping=.5,precision_profile=2,serialize=True,edge_contexts=contexts)
        self.assertEqual(original['packets'],revised['packets'])
        for hop in range(3):
            self.assertTrue(torch.equal(original['codes'][hop],revised['codes'][hop]))
            torch.testing.assert_close(original['means'][hop],revised['means'][hop],rtol=2e-13,atol=2e-13)
            torch.testing.assert_close(original['covariances'][hop],revised['covariances'][hop],rtol=2e-13,atol=2e-13)
        q,h=self.factors(3)
        for mode in core.ARMS:
            state=core.advance(core.initialize(q,h,edges),2,mode=mode,quantized=False)['state']
            again=core.advance(state,1,mode=mode,quantized=False)
            mean,cov=core.belief(state)
            self.assertTrue(torch.equal(mean,again['mean']))
            self.assertTrue(torch.equal(cov,again['covariance']))
            self.assertIs(again['state']['q'],q);self.assertIs(again['state']['h'],h)


if __name__=='__main__':
    unittest.main()
