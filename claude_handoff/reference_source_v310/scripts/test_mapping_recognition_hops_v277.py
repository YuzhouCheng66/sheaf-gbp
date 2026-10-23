"""Native128 CPU fixtures; no real checkpoints, training, GPU or TEST data."""
import copy
import unittest
from unittest import mock

import torch

from scripts.mapping_context_codec_v121 import ContextCodecConfig, ContextDecoder
from scripts.mapping_implicit_model_v192 import latent_quantize
from scripts.mapping_local_capacity_v269 import LocalCapacity64
from scripts.mapping_recognition_hops_v277 import (
    RecognitionHops64, rollout, source_factor_hashes, LATENT_PRIOR)
from scripts import mapping_recurrent_gbp_v213 as recurrent
from scripts import mapping_observed_wire_v246 as wire
from scripts import mapping_session_bound_wire_v267 as transport


def maps(n=2):
    value = torch.full((1,n,128,128),-1,dtype=torch.int8)
    value[:,0,:64] = 0
    value[:,0,17:20,:64] = 1
    if n>1:
        value[:,-1,64:] = 0
        value[:,-1,90:93,60:] = 1
    return value


class RecognitionHopsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = dict(side=16,planes=4,bits=8,width=128,seed=141021)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(121)
            decoder = ContextDecoder(ContextCodecConfig(**cls.config)).state_dict()
        meta = dict(sha256='a'*64,step=4000,source_only=True,joint_updates=0,
                    family='mapping-context-codec-v121',optimizer_inherited=False)
        cls.parent = LocalCapacity64(decoder,cls.config,meta,train_decoder=True)
        cls.payload = dict(construction=cls.parent.construction_metadata(),state_dict=cls.parent.state_dict())

    def model(self):
        return RecognitionHops64.from_parent_payload(self.payload)

    def test_initial_parent_function_precision_and_CPU_rng(self):
        before = self.parent.weights_hash()
        rng = torch.get_rng_state().clone()
        with mock.patch('torch.cuda.manual_seed_all',side_effect=AssertionError('CUDA RNG touched')):
            model = self.model()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(before,self.parent.weights_hash())
        self.assertEqual(before,model.capacity.weights_hash())
        p = maps()
        f = model.factors(p)
        self.assertTrue(torch.equal(f['mu'].flatten(0,1),self.parent.encode(p.flatten(0,1))))
        torch.testing.assert_close(f['learned_q'],torch.ones_like(f['learned_q']),rtol=0,atol=1e-7)
        self.assertEqual(f['q'].dtype,torch.float64)
        self.assertTrue(torch.equal(f['q'],torch.diag_embed(torch.diagonal(f['q'],dim1=-2,dim2=-1))))
        self.assertGreater(torch.linalg.eigvalsh(f['q']).min().item(),0.)
        for p in model.parameters(): self.assertTrue(p.requires_grad)
        self.assertEqual(sum(p.numel() for p in model.precision_head.parameters()),2048*64+64)

    def test_empty_evidence_and_decoder_only_quantized_mean(self):
        model = self.model()
        empty = torch.full((1,2,128,128),-1,dtype=torch.int8)
        result = rollout(model,empty,[],rounds=0)
        f = result['source_factors']
        self.assertFalse(f['active'].any())
        self.assertTrue(torch.equal(f['h'],torch.zeros_like(f['h'])))
        self.assertTrue(torch.equal(f['q'],torch.eye(64,dtype=torch.float64)[None,None].expand(1,2,-1,-1)*(LATENT_PRIOR/128)))
        self.assertTrue(torch.equal(result['initial'],torch.zeros_like(result['initial'])))
        expected = model.capacity.decode_continuous(torch.zeros(2,64))
        for key in expected:
            self.assertTrue(torch.equal(result['own'][key].flatten(0,1),expected[key]))
        with self.assertRaises(TypeError): model.decode(result['initial'],empty)
        random = torch.linspace(-1.2,1.2,128).reshape(1,2,64)
        out = model.decode(random)
        quantized = model.capacity.decode_continuous(latent_quantize(random).reshape(2,64))
        for key in out: self.assertTrue(torch.equal(out[key].flatten(0,1),quantized[key]))

    def test_source_isolation_hash_and_source_immutable_through_sweeps(self):
        model = self.model()
        original = maps(3)
        changed = original.clone(); changed[:,2] = 1
        a,b = model.factors(original),model.factors(changed)
        for key in ('q','h','mu','learned_q'):
            self.assertTrue(torch.equal(a[key][:,:2],b[key][:,:2]),key)
        self.assertEqual(source_factor_hashes(a)[0][:2],source_factor_hashes(b)[0][:2])
        result = rollout(model,original,[(0,1,1.),(1,2,1.)])
        q,h = result['source_factors']['q'],result['source_factors']['h']
        for state in result['states']:
            self.assertIs(state['q'],q);self.assertIs(state['h'],h)
            self.assertTrue(torch.equal(state['source_versions'],torch.zeros(1,3,dtype=torch.long)))
        self.assertEqual(result['state']['iteration'],2)
        self.assertEqual(len(result['trace']),8)

    def test_synchronous_one_hop_delay_no_two_hop_leakage(self):
        model = self.model()
        a = maps(3)
        b = a.clone();b[:,0,:64] = 1
        with torch.no_grad():
            first = rollout(model,a,[(0,1,1.),(1,2,1.)])
            other = rollout(model,b,[(0,1,1.),(1,2,1.)])
        for hop in (0,1):
            self.assertTrue(torch.equal(first['means'][hop][:,2],other['means'][hop][:,2]))
        self.assertFalse(torch.equal(first['means'][2][:,2],other['means'][2][:,2]))
        # Prefix1 equals an independently executed one-sweep rollout.
        one = rollout(model,a,[(0,1,1.),(1,2,1.)],rounds=1)
        self.assertTrue(torch.equal(one['final_code'],first['codes'][1]))

    def test_actual_v267_packets_and_receive_reconstruction_match_live_math(self):
        model = self.model()
        contexts = {(0,1):b'0123456789abcdef'}
        with torch.no_grad():
            fast = rollout(model,maps(),[(0,1,1.)])
            actual = rollout(model,maps(),[(0,1,1.)],serialize=True,edge_contexts=contexts)
        self.assertEqual(actual['actual_serialized_application_bytes'],4*172)
        self.assertIsNone(fast['actual_serialized_application_bytes'])
        for key in ('final_code','mean','covariance'):
            self.assertTrue(torch.equal(fast[key],actual[key]),key)
        f = actual['source_factors']
        receiver = transport.bind_context(recurrent.initialize(f['q'],f['h'],[(0,1,1.)]),contexts)
        for packet in actual['packets']:
            receiver,accepted = transport.receive(receiver,packet)
            self.assertTrue(accepted)
        mean,cov = recurrent.belief(receiver)
        self.assertTrue(torch.equal(mean,actual['mean']))
        self.assertTrue(torch.equal(cov,actual['covariance']))
        receiver2,accepted = transport.receive(receiver,actual['packets'][-1])
        self.assertFalse(accepted);self.assertIs(receiver2,receiver)
        self.assertEqual(actual['inner_application_byte_budget_per_world'],4*148)

    def test_fused_only_gradients_live_through_true_wire_all_groups(self):
        model = self.model()
        result = rollout(model,maps(),[(0,1,1.)])
        # No source reconstruction term can disguise a detached message path.
        loss = result['fused']['wall_logits'].square().mean()+result['fused']['known_logits'].square().mean()
        loss.backward()
        groups = {key:0. for key in ('encoder','global_head','lift','decoder','precision_head')}
        for name,p in model.named_parameters():
            self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
            key = name.split('.')[1] if name.startswith('capacity.') else name.split('.')[0]
            groups[key] += p.grad.abs().sum().item()
        for key,value in groups.items(): self.assertGreater(value,0.,key)
        # Independently compare final means to the unchanged actual wire solver.
        f = result['source_factors']
        direct = wire.advance(recurrent.initialize(f['q'],f['h'],[(0,1,1.)]),2,trace=True)
        self.assertTrue(torch.equal(direct['mean'],result['mean']))

    def test_self_contained_checkpoint_exact_and_bad_inputs_fail_closed(self):
        model = self.model()
        saved = dict(construction=model.construction_metadata(),state_dict=copy.deepcopy(model.state_dict()))
        with mock.patch('torch.load',side_effect=AssertionError('Ancestor reopened')):
            restored = RecognitionHops64.from_payload(saved)
        self.assertEqual(model.weights_hash(),restored.weights_hash())
        self.assertEqual(model.construction_metadata(),restored.construction_metadata())
        self.assertTrue(torch.equal(model.factors(maps())['q'],restored.factors(maps())['q']))
        for kwargs in ({'rounds':3},{'rounds':True},{'serialize':True},
                       {'edge_contexts':{(0,1):b'0123456789abcdef'}}):
            with self.assertRaises(ValueError): rollout(model,maps(),[(0,1,1.)],**kwargs)
        for edges in ([(0,1,2.)],[(0,1,float('nan'))],[(0,1,1.),(1,0,1.)],[(0,0,1.)]):
            with self.assertRaises(ValueError): rollout(model,maps(),edges)
        with self.assertRaises(ValueError): model.factors(torch.zeros(1,2,64,64))
        with self.assertRaises(ValueError): model.factors(torch.full_like(maps(),2))


if __name__ == '__main__': unittest.main()
