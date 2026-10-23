"""Synthetic CPU contracts: no checkpoints, maps, GPU or training."""
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from scripts import mapping_multiroom_forest_v295 as core
from scripts import mapping_common_map_v284 as old
from scripts.mapping_recognition_hops_v277 import LATENT_PRIOR
from scripts.mapping_implicit_model_v192 import latent_quantize


def factors(n, batch=1, maximum=False):
    p=(torch.full((batch,n,64),4.,dtype=torch.float64) if maximum else
       torch.linspace(.1,3.9,batch*n*64,dtype=torch.float64).reshape(batch,n,64))
    mu=torch.sin(torch.arange(batch*n*64,dtype=torch.float64)).reshape(batch,n,64)*.6
    return torch.diag_embed(p+LATENT_PRIOR/(n*64)),p*mu


def contexts(edges):
    return {(i,j):(1+16*i+j).to_bytes(16,'big') for i,j,_ in edges}


class SyntheticSource(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.p=nn.Parameter(torch.full((1,n,64),.3,dtype=torch.float64))
        self.mu=nn.Parameter(torch.linspace(-.7,.7,n*64,dtype=torch.float64).reshape(1,n,64))
        self.calls=0

    def factors(self,private):
        self.calls+=1; b,n=private.shape[:2]
        p=self.p.expand(b,-1,-1); mu=self.mu.expand(b,-1,-1)
        return dict(q=torch.diag_embed(p+LATENT_PRIOR/(n*64)),h=p*mu,
            mu=mu,prior_precision=LATENT_PRIOR/(n*64),active=torch.ones(b,n,dtype=torch.bool))


class SyntheticModel(old.CommonMap64):
    def __init__(self,n):
        nn.Module.__init__(self)
        self.source=SyntheticSource(n); self.calls=[]

    def decode(self,features):
        self.calls.append(features.detach().clone())
        return dict(known_logits=features.sum(-1,keepdim=True),wall_logits=features[...,:64].sum(-1,keepdim=True))


class ForestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_deterministic_physical_bfs_all_nodes_and_no_map_selection(self):
        positions=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[8.,8.]])
        selected=core.select_forest(positions,1.1)
        self.assertEqual(selected['edges'],[(0,1,1.),(0,2,1.),(1,3,1.)])
        self.assertEqual(selected['components'],[[0,1,2,3],[4]])
        self.assertTrue(selected['offline_public_schedule'])
        again=core.select_forest(positions,1.1,selected['physical_adjacency'])
        self.assertEqual(again['edges'],selected['edges'])
        wrong=selected['physical_adjacency'].copy();wrong[0,4]=True
        with self.assertRaises(ValueError):core.select_forest(positions,1.1,wrong)
        for graph in ([(0,1,1.),(1,2,1.),(0,2,1.)],[(0,1,1.),(1,0,1.)]):
            with self.assertRaises(ValueError):core.validate_forest(graph,3)

    def test_float_every_reached_subset_with_partial_prior_chain_star_forest(self):
        cases=[(16,[(i,i+1,1.) for i in range(15)]),
               (16,[(0,i,1.) for i in range(1,16)]),
               (8,[(0,1,1.),(1,2,1.),(3,4,1.),(4,5,1.)]),(1,[])]
        for n,edges in cases:
            q,h=factors(n,batch=2); state=core.initialize(q,h,edges)
            for r in range(17):
                m,v=core.scalar_belief(state); central=core.central_reference(q,h,edges,r)
                torch.testing.assert_close(m,central['mean'],rtol=3e-14,atol=3e-14)
                torch.testing.assert_close(v,central['variance'],rtol=3e-14,atol=3e-14)
                self.assertIs(state['q'],q);self.assertIs(state['h'],h)
                if r<16:state=core.advance(state,quantized=False)['state']
        q=torch.eye(64,dtype=torch.float64)[None,None].expand(1,8,64,64)*LATENT_PRIOR/(8*64)
        h=torch.zeros(1,8,64,dtype=torch.float64)
        ref=core.central_reference(q,h,[(i,i+1,1.) for i in range(7)],2)
        self.assertAlmostEqual(ref['precision'][0,0,0].item(),3*LATENT_PRIOR/(8*64))
        self.assertAlmostEqual(ref['precision'][0,3,0].item(),5*LATENT_PRIOR/(8*64))

    def test_profile_bound_literal_quantization_and_no_old_global_patch(self):
        bound=core.precision_bound(16)
        self.assertTrue(bound['strictly_inside']);self.assertLess(bound['outgoing_precision_upper_bound'],128)
        self.assertNotIn(4,old.CommonMapCodec64.RANGES)
        p=torch.logspace(-8,np.log10(128.),64,dtype=torch.float64).clamp_max(128.)[None]
        h=torch.linspace(-.75,.75,64,dtype=torch.float64)[None]*p.sqrt()
        p.requires_grad_();h.requires_grad_()
        pp,hh,_=core.quantize(p,h)
        raw,_=core.ForestCodec64.encode(p.detach()[0].numpy(),h.detach()[0].numpy(),0,1,0,
            mode='standardized',precision_profile=4)
        d=core.ForestCodec64.decode(raw)
        self.assertEqual(len(raw),148)
        self.assertTrue(torch.equal(pp[0],torch.from_numpy(d['precision'].astype(np.float32)).double()))
        self.assertTrue(torch.equal(hh[0],torch.from_numpy(d['eta'].astype(np.float32)).double()))
        gp,gh=torch.autograd.grad(pp.sum()+hh.sum(),(p,h))
        self.assertTrue(torch.equal(gp,torch.ones_like(p)));self.assertTrue(torch.equal(gh,torch.ones_like(h)))
        for badp,badh in ((128.001,0.),(1e-9,0.),(0.,1.),(.2,1000.)):
            with self.assertRaises(ValueError):core.quantize(torch.full((1,64),badp,dtype=torch.float64),torch.full((1,64),badh,dtype=torch.float64))

    def test_maxq16_chain_star_literal_every_frame_and_stability(self):
        for edges in ([(i,i+1,1.) for i in range(15)],[(0,i,1.) for i in range(1,16)]):
            q,h=factors(16,maximum=True); ctx=contexts(edges)
            state=core.bind_context(core.initialize(q,h,edges),ctx)
            packets=[]; before=None
            for r in range(16):
                out=core.advance(state,trace=True)
                packets.extend(core.serialize_trace(out['state'],out['trace']));state=out['state']
                self.assertLessEqual(max(x.max().item() for x in state['p'].values()),128.)
                if r==14:before=core.scalar_belief(state)
            after=core.scalar_belief(state)
            self.assertTrue(torch.equal(before[0],after[0]));self.assertTrue(torch.equal(before[1],after[1]))
            replay=core.bind_context(core.initialize(q,h,edges),ctx)
            # Reordering inside each synchronous sweep does not affect belief.
            width=len(replay['directed'])
            for start in range(0,len(packets),width):
                for packet in reversed(packets[start:start+width]):
                    replay,accepted=core.receive(replay,packet);self.assertTrue(accepted)
            self.assertTrue(torch.equal(core.scalar_belief(replay)[0],after[0]))
            self.assertTrue(torch.equal(core.scalar_belief(replay)[1],after[1]))
            for packet in reversed(packets):
                same,accepted=core.receive(replay,packet)
                self.assertFalse(accepted);self.assertIs(same,replay)

    def test_duplicate_stale_conflict_context_and_new_sequence_no_new_evidence(self):
        edges=[(0,1,1.)];q,h=factors(2);ctx=contexts(edges)
        state=core.bind_context(core.initialize(q,h,edges),ctx)
        first=core.advance(state,trace=True); packet=core.serialize_trace(first['state'],first['trace'])[0]
        got,yes=core.receive(state,packet);self.assertTrue(yes)
        same,no=core.receive(got,packet);self.assertFalse(no);self.assertIs(same,got)
        bad=bytes([packet[0]^1])+packet[1:]
        same,no=core.receive(got,bad);self.assertFalse(no);self.assertIs(same,got)
        d=core.ForestCodec64.decode(packet[24:])
        raw,_=core.ForestCodec64.encode(np.ones(64)*.2,np.zeros(64),d['sender'],d['receiver'],0,mode='standardized',precision_profile=4)
        with self.assertRaises(ValueError):core.receive(got,packet[:24]+raw)
        full=first['state'];m,v=core.scalar_belief(full)
        final=core.advance(full,15)['state']
        self.assertTrue(torch.equal(core.scalar_belief(final)[0],m))
        self.assertTrue(torch.equal(core.scalar_belief(final)[1],v))

    def test_actual_h_delay_empty_component_and_live_gradients(self):
        q,h=factors(8);edges=[(i,i+1,1.) for i in range(6)]
        changed=h.clone();changed[:,0]+=.2
        a,b=core.initialize(q,h,edges),core.initialize(q,changed,edges)
        for r in range(7):
            ma,_=core.scalar_belief(a);mb,_=core.scalar_belief(b)
            if r<6:self.assertTrue(torch.equal(ma[:,6],mb[:,6]))
            else:self.assertFalse(torch.equal(ma[:,6],mb[:,6]))
            self.assertTrue(torch.equal(ma[:,7],mb[:,7]))
            a=core.advance(a)['state'];b=core.advance(b)['state']
        p=torch.full((1,8,64),.3,dtype=torch.float64,requires_grad=True)
        mu=torch.linspace(-.7,.7,512,dtype=torch.float64).reshape(1,8,64).requires_grad_()
        state=core.advance(core.initialize(torch.diag_embed(p+LATENT_PRIOR/(8*64)),p*mu,edges),8)['state']
        mean,var=core.scalar_belief(state)
        gp,gm=torch.autograd.grad((mean.square()+var.log().square()).mean(),(p,mu))
        for g in (gp,gm):
            self.assertTrue(bool(torch.isfinite(g).all()));self.assertGreater(g.abs().sum().item(),0)

    def test_rollout_selected_decode_only_source_once_literal_audit(self):
        model=SyntheticModel(8);private=torch.zeros(1,8,128,128,dtype=torch.int8)
        edges=[(i,i+1,1.) for i in range(7)];ctx=contexts(edges)
        with mock.patch('torch.linalg.inv',side_effect=AssertionError('No dense inverse')):
            result=core.rollout(model,private,edges,serialize=True,edge_contexts=ctx)
        self.assertEqual(model.source.calls,1);self.assertEqual(len(model.calls),5)
        self.assertEqual(set(result['decoded']),{0,1,2,4,8});self.assertEqual(len(result['means']),9)
        audit=core.audit_rollout(result,edges,ctx)
        self.assertTrue(audit['passed']);self.assertEqual(audit['actual_data_bytes'],172*14*8)
        self.assertEqual(audit['additional_forced_replay_bytes'],audit['actual_data_bytes'])
        loss=sum(x['known_logits'].square().mean() for x in result['decoded'].values())
        loss.backward()
        self.assertTrue(torch.isfinite(model.source.p.grad).all());self.assertTrue(torch.isfinite(model.source.mu.grad).all())
        # Empty-edge forest is a legal source-only graph; no fake packet traffic.
        one=SyntheticModel(1)
        empty=core.rollout(one,private[:,:1],[],rounds=2,decode_rounds=(0,2),serialize=True,edge_contexts={})
        self.assertEqual(empty['packets'],[]);self.assertTrue(core.audit_rollout(empty,[],{})['passed'])

    def test_input_failclosed_and_reachability(self):
        q,h=factors(3)
        with self.assertRaises(ValueError):core.initialize(q,h,[(0,1,1.),(1,2,1.),(0,2,1.)])
        bad=q.clone();bad[...,0,1]=1e-8
        with self.assertRaises(ValueError):core.initialize(bad,h,[(0,1,1.)])
        with self.assertRaises(ValueError):core.initialize(q*100,h,[])
        with self.assertRaises(ValueError):core.advance(core.initialize(q,h,[]),17)
        self.assertTrue(np.array_equal(core.reachability([],3,16),np.eye(3,dtype=bool)))
        with self.assertRaises(ValueError):core.select_forest(np.zeros((17,2)),1.)


if __name__=='__main__':
    unittest.main()
