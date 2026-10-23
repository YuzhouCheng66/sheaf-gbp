"""CPU synthetic transport/gradient tests, not learned fidelity experiments."""
import unittest
from unittest import mock

import numpy as np
import torch

from scripts import mapping_support_forest_v306 as adapter
from scripts import mapping_multiroom_forest_v295 as forest
from scripts import mapping_support_local_v306 as local


def inputs(size=128,n=3):
    x=torch.full((1,n,size,size),-1,dtype=torch.int8)
    side=size//4
    for i in range(n):
        t=i%16;row,col=divmod(t,4)
        x[:,i,row*side:(row+1)*side,col*side:(col+1)*side]=0
        x[:,i,row*side:(row+1)*side,col*side+side//2]=1
    return x


def contexts(edges):
    return {(i,j):(1+16*i+j).to_bytes(16,'big') for i,j,_ in edges}


class SupportForestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_float_central_every_round_and_disconnected_sources(self):
        model=local.SupportLocal64();x=inputs(n=4);edges=[(0,1,1.),(1,2,1.)]
        with torch.no_grad():
            result=adapter.rollout(model,x,edges,quantized=False,features_only=True)
        f=result['source_factors']
        for r in range(9):
            ref=forest.central_reference(f['q'],f['h'],edges,r)
            torch.testing.assert_close(result['means'][r],ref['mean'],rtol=3e-14,atol=3e-14)
            torch.testing.assert_close(result['variances'][r],ref['variance'],rtol=3e-14,atol=3e-14)
            self.assertIs(result['states'][r]['q'],f['q']);self.assertIs(result['states'][r]['h'],f['h'])
            self.assertTrue(torch.equal(result['features'][r][:,3],result['features'][0][:,3]))
        self.assertTrue(np.array_equal(forest.reachability(edges,4,8)[3],[False,False,False,True]))

    def test_actual_packets_source_once_replay_and_no_new_evidence(self):
        model=local.SupportLocal64();x=inputs();edges=[(0,1,1.),(1,2,1.)];ctx=contexts(edges)
        with torch.no_grad(),mock.patch.object(model,'source_factors',wraps=model.source_factors) as source:
            result=adapter.rollout(model,x,edges,serialize=True,edge_contexts=ctx,trace=False)
        self.assertEqual(source.call_count,1)
        self.assertEqual(set(result['decoded']),{0,1,2,4,8})
        self.assertEqual(len(result['packets']),32)
        self.assertTrue(all(len(packet)==172 for packet in result['packets']))
        proof=adapter.audit_rollout(result,edges,ctx)
        self.assertTrue(proof['passed']);self.assertTrue(proof['actual_receive_parity'])
        self.assertTrue(proof['replay_idempotent']);self.assertEqual(proof['actual_data_bytes'],32*172)
        self.assertEqual(proof['additional_forced_replay_bytes'],32*172)
        state=result['state'];q=state['q'].clone();h=state['h'].clone()
        before=forest.scalar_belief(state)
        for _ in range(3):
            with torch.no_grad():model.decode(result['features'][-1],resolution=128)
            no_send=forest.advance(state,rounds=0)
            self.assertIs(no_send['state'],state);self.assertEqual(no_send['trace'],[])
        self.assertTrue(torch.equal(q,state['q']));self.assertTrue(torch.equal(h,state['h']))
        for packet in reversed(result['packets']):
            same,accepted=forest.receive(state,packet)
            self.assertFalse(accepted);self.assertIs(same,state)
        after=forest.scalar_belief(state)
        self.assertTrue(torch.equal(before[0],after[0]));self.assertTrue(torch.equal(before[1],after[1]))
        # A static tree's saturated cavity replaces, rather than adds, evidence.
        self.assertTrue(torch.equal(result['means'][2],result['means'][8]))
        self.assertTrue(torch.equal(result['variances'][2],result['variances'][8]))

    def test_features_only_skips_decoder_retains_chunkwise_full_gradient(self):
        full=local.SupportLocal64();chunked=local.SupportLocal64.from_payload(full.to_payload())
        x=inputs(n=2);edges=[(0,1,1.)];selected=(0,1,2)
        a=adapter.rollout(full,x,edges,rounds=2,decode_rounds=selected)
        with mock.patch.object(chunked,'decode',side_effect=AssertionError('D must not run')):
            b=adapter.rollout(chunked,x,edges,rounds=2,decode_rounds=selected,features_only=True)
        self.assertEqual(b['decoded'],{});self.assertEqual(b['decoded_rounds'],())
        self.assertEqual(b['requested_decode_rounds'],selected)
        for k in selected:self.assertTrue(torch.equal(a['features'][k],b['features'][k]))
        loss=sum(sum(v.square().mean() for v in a['decoded'][k].values()) for k in selected)/len(selected)
        loss.backward()
        counter=0
        for k in selected:
            for node in range(2):
                heads=chunked.decode(b['features'][k][:,node],resolution=128)
                component=sum(v.square().mean() for v in heads.values())/(2*len(selected))
                counter+=1;component.backward(retain_graph=counter<2*len(selected))
        for (name,p),(other,q) in zip(full.named_parameters(),chunked.named_parameters()):
            self.assertEqual(name,other);self.assertIsNotNone(p.grad,name);self.assertIsNotNone(q.grad,name)
            self.assertTrue(torch.isfinite(q.grad).all(),name)
            # Same mathematical gradient; floating reduction ordering need not be bitwise.
            torch.testing.assert_close(p.grad,q.grad,rtol=3e-5,atol=2e-7,msg=name)
        self.assertGreater(float(chunked.precision_head.weight.grad.abs().sum()),0)

    def test_native_shape_smokes_and_empty_forest(self):
        model=local.SupportLocal64()
        for size in local.RESOLUTIONS:
            with torch.no_grad():
                result=adapter.rollout(model,inputs(size,n=1),[],rounds=0,decode_rounds=(0,),
                    serialize=True,edge_contexts={})
            self.assertEqual(result['native_resolution'],size)
            self.assertEqual(result['decoded'][0]['wall_logits'].shape,(1,1,size*size))
            self.assertEqual(result['packets'],[])
            self.assertTrue(adapter.audit_rollout(result,[],{})['passed'])

    def test_float_address_isolation_fixed_prior_and_empty_source_prior_caveat(self):
        model=local.SupportLocal64();x=inputs();y=x.clone();y[:,2,-32:,-32:]=1
        edges=[(0,1,1.),(1,2,1.)]
        with torch.no_grad():
            a=adapter.rollout(model,x,edges,rounds=2,decode_rounds=(2,),quantized=False)
            b=adapter.rollout(model,y,edges,rounds=2,decode_rounds=(2,),quantized=False)
        for r in range(3):
            for field in ('means','variances','codes'):
                self.assertTrue(torch.equal(a[field][r][...,:60],b[field][r][...,:60]))
        for key in a['decoded'][2]:
            pa=local.patches(a['decoded'][2][key].reshape(1,3,128,128))
            pb=local.patches(b['decoded'][2][key].reshape(1,3,128,128))
            self.assertTrue(torch.equal(pa[...,:15,:,:],pb[...,:15,:,:]))
        # Address15 is empty everywhere in a. Arrival still adds fixed prior.
        f=a['source_factors'];self.assertEqual(int(torch.count_nonzero(f['data_q'][...,60:])),0)
        self.assertTrue(torch.equal(a['means'][0][...,60:],a['means'][2][...,60:]))
        self.assertFalse(torch.equal(a['variances'][0][...,60:],a['variances'][2][...,60:]))
        torch.testing.assert_close(a['variances'][2][0,0,60:],a['variances'][0][0,0,60:]/3,rtol=1e-15,atol=0)

    def test_shared_exponent_actual_counterexample_not_wire_noninterference(self):
        # Legal bounded data q, original hard8 source mu, empty receiver tile0.
        data=torch.ones(1,2,64,dtype=torch.float32).double()
        data[:,0,0]=torch.tensor(.14,dtype=torch.float32).double();data[:,1,:4]=0.
        mu=torch.zeros(1,2,64);mu[:,0,0]=local.codes_to_latent(torch.tensor(128,dtype=torch.uint8))
        mu[:,0,4]=local.codes_to_latent(torch.tensor(140,dtype=torch.uint8))
        changed=mu.clone();changed[:,0,4]=1.
        q=torch.diag_embed(data+local.LATENT_PRIOR/(2*64));h=data*mu.double();hh=data*changed.double()
        edges=[(0,1,1.)];ctx=contexts(edges)
        a=forest.bind_context(forest.initialize(q,h,edges),ctx)
        b=forest.bind_context(forest.initialize(q,hh,edges),ctx)
        fa=forest.advance(a,quantized=False);fb=forest.advance(b,quantized=False)
        self.assertTrue(torch.equal(fa['mean'][...,0:4],fb['mean'][...,0:4]))
        ra=forest.advance(a,trace=True);rb=forest.advance(b,trace=True)
        pa=forest.serialize_trace(ra['state'],ra['trace']);pb=forest.serialize_trace(rb['state'],rb['trace'])
        self.assertTrue(all(len(p)==172 for p in pa+pb))
        for packet in pa:a,accepted=forest.receive(a,packet);self.assertTrue(accepted)
        for packet in pb:b,accepted=forest.receive(b,packet);self.assertTrue(accepted)
        ma,va=forest.scalar_belief(a);mb,vb=forest.scalar_belief(b)
        self.assertTrue(torch.equal(va,vb))
        self.assertGreater(float(ma[0,1,0]),0);self.assertEqual(float(mb[0,1,0]),0)
        self.assertFalse(torch.equal(local.latent_quantize(ma.float())[...,0],local.latent_quantize(mb.float())[...,0]))
        self.assertFalse(torch.equal(ra['trace'][0]['exponent'],rb['trace'][0]['exponent']))

    def test_reverse_exclusion_causal_delay_and_source_hashes(self):
        model=local.SupportLocal64();x=inputs();y=x.clone();y[:,0,:32,:32]=1
        edges=[(0,1,1.),(1,2,1.)]
        with torch.no_grad():
            a=adapter.rollout(model,x,edges,rounds=2,decode_rounds=(),features_only=True,quantized=False)
            b=adapter.rollout(model,y,edges,rounds=2,decode_rounds=(),features_only=True,quantized=False)
        for r in (0,1):self.assertTrue(torch.equal(a['features'][r][:,2],b['features'][r][:,2]))
        self.assertFalse(torch.equal(a['features'][2][:,2],b['features'][2][:,2]))
        f=a['source_factors'];prior=local.LATENT_PRIOR/(3*64)
        # At sweep2, direction1->0 includes original1+2, never source0 echo.
        outgoing=next(item for item in a['trace'] if item['round']==1 and item['sender']==1 and item['receiver']==0)
        torch.testing.assert_close(outgoing['raw_p'],f['data_q'][:,1]+f['data_q'][:,2]+2*prior,rtol=1e-15,atol=1e-15)
        torch.testing.assert_close(outgoing['raw_eta'],f['h'][:,1]+f['h'][:,2],rtol=0,atol=0)

    def test_old_guard_unchanged_and_invalid_requests_rejected(self):
        model=local.SupportLocal64();x=inputs();edges=[(0,1,1.),(1,2,1.)]
        with self.assertRaises(ValueError):forest.rollout(model,x,edges)
        with self.assertRaises(ValueError):adapter.rollout(object(),x,edges)
        for options in (dict(rounds=2),dict(decode_rounds=(0,0)),dict(decode_rounds=(2,1)),
                        dict(features_only=1),dict(quantized=False,serialize=True,edge_contexts=contexts(edges)),
                        dict(serialize=True),dict(edge_contexts=contexts(edges))):
            with self.assertRaises(ValueError):adapter.rollout(model,x,edges,**options)
        with self.assertRaises(ValueError):adapter.rollout(model,x,edges+[(0,2,1.)])
        with self.assertRaises(ValueError):adapter.audit_rollout({},edges,contexts(edges))
        for p,h in ((129.,0.),(1e-9,0.),(1.,1000.)):
            with self.assertRaises(ValueError):forest.quantize(torch.full((1,64),p,dtype=torch.float64),torch.full((1,64),h,dtype=torch.float64))


if __name__=='__main__':
    unittest.main()
