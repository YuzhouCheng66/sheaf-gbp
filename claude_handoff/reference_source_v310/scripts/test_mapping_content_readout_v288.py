"""CPU synthetic typed weights: content-path isolation, not map efficacy."""
import copy
import unittest
from unittest import mock

import torch

from scripts import mapping_common_map_v284 as old
from scripts import mapping_content_readout_v288 as core
from scripts import test_mapping_common_map_v284 as fixture
from scripts.test_mapping_recognition_hops_v277 import maps


class ContentReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        fixture.CommonMapTests.setUpClass()
        parent = old.CommonMap64.from_parent_payloads(
            fixture.CommonMapTests.source_payload, fixture.CommonMapTests.readout_payload)
        cls.payload = dict(construction=parent.construction_metadata(),
                           state_dict=copy.deepcopy(parent.state_dict()), arm='equality', step=4000,
                           model_sha256=parent.weights_hash())
        cls.features = fixture.fixture.BeliefReadoutTests.normalization_features[:2].reshape(1,2,128)

    def model(self, arm='mean_wall'):
        return core.ContentReadout64.from_parent_payload(self.payload, arm=arm)

    def test_control_exact_original_initial_output_hash_parameters_and_rng(self):
        rng = torch.get_rng_state().clone()
        with mock.patch('torch.cuda.manual_seed_all', side_effect=AssertionError('No CUDA')):
            parent = old.CommonMap64.from_payload(self.payload)
            control, candidate = self.model('joint_belief'), self.model()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(parent.weights_hash(), control.weights_hash())
        self.assertEqual(parent.weights_hash(), candidate.weights_hash())
        self.assertEqual(list(parent.state_dict()), list(candidate.state_dict()))
        self.assertEqual([p.numel() for p in parent.trainable_parameters()],
                         [p.numel() for p in candidate.trainable_parameters()])
        with torch.no_grad():
            expected, actual = parent.decode(self.features), control.decode(self.features)
        for key in expected:
            self.assertTrue(torch.equal(expected[key], actual[key]), key)
        self.assertIs(core.rollout, old.rollout)
        self.assertIs(core.audit_rollout, old.audit_rollout)

    def test_both_arms_make_two_shared_original_calls_with_exact_inputs(self):
        original_decode = old.CommonMap64.decode
        for arm in core.ARMS:
            model = self.model(arm)
            inputs = []
            def spy(instance, features):
                self.assertIs(instance, model)
                inputs.append(features.detach().clone())
                return original_decode(instance, features)
            with mock.patch.object(old.CommonMap64, 'decode', spy):
                result = model.decode(self.features)
            self.assertEqual(len(inputs), 2)
            self.assertTrue(torch.equal(inputs[0], self.features))
            self.assertTrue(torch.equal(inputs[1][..., :64], self.features[..., :64]))
            if arm == 'joint_belief':
                self.assertTrue(torch.equal(inputs[1], self.features))
            else:
                expected = model.readout.normalization_mean[64:].expand_as(inputs[1][..., 64:])
                self.assertTrue(torch.equal(inputs[1][..., 64:], expected))
                normalized = (inputs[1][..., 64:]-expected)/model.readout.normalization_std[64:]
                self.assertEqual(torch.count_nonzero(normalized).item(), 0)
            expected_known = original_decode(model, self.features)['known_logits']
            self.assertTrue(torch.equal(result['known_logits'], expected_known))

    def test_wall_covariance_invariance_zero_direct_gradient_and_live_known(self):
        model = self.model()
        features = self.features.detach().clone().requires_grad_()
        changed = features.detach().clone()
        changed[..., 64:] += torch.linspace(-1., 2., 64)
        before = features.detach().clone()
        out = model.decode(features)
        other = model.decode(changed)
        self.assertTrue(torch.equal(out['wall_logits'], other['wall_logits']))
        self.assertFalse(torch.equal(out['known_logits'], other['known_logits']))
        wall_grad = torch.autograd.grad(out['wall_logits'].square().mean(), features, retain_graph=True)[0]
        known_grad = torch.autograd.grad(out['known_logits'].square().mean(), features)[0]
        self.assertEqual(torch.count_nonzero(wall_grad[..., 64:]).item(), 0)
        self.assertGreater(wall_grad[..., :64].abs().sum().item(), 0)
        self.assertGreater(known_grad[..., 64:].abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(wall_grad).all() and torch.isfinite(known_grad).all())
        self.assertTrue(torch.equal(features.detach(), before))

    def test_self_contained_restore_after_active_update_and_invalid_metadata(self):
        for arm in core.ARMS:
            model = self.model(arm)
            original_provenance = model.parent_state_sha256
            with torch.no_grad():
                model.source.precision_head.bias.add_(.007)
                model.readout.lift.bias.add_(.003)
            payload = dict(construction=model.construction_metadata(),
                           state_dict=copy.deepcopy(model.state_dict()), model_sha256=model.weights_hash())
            with mock.patch('torch.load', side_effect=AssertionError('No ancestor file reads')):
                restored = core.ContentReadout64.from_payload(payload)
            self.assertEqual(model.weights_hash(), restored.weights_hash())
            self.assertEqual(model.construction_metadata(), restored.construction_metadata())
            self.assertEqual(original_provenance, restored.parent_state_sha256)
            for key, value in model.decode(self.features).items():
                self.assertTrue(torch.equal(value, restored.decode(self.features)[key]))
            for key, value in (('arm','bad'), ('decoder_calls_per_belief',1), ('new_parameter_count',1)):
                bad = copy.deepcopy(payload); bad['construction'][key] = value
                with self.assertRaises(ValueError): core.ContentReadout64.from_payload(bad)
            bad = copy.deepcopy(payload); bad['state_dict']['readout.lift.bias'][0] = float('nan')
            with self.assertRaises(ValueError): core.ContentReadout64.from_payload(bad)
        for changes in (dict(arm='finite_beta'), dict(step=3000), dict(model_sha256='0'*64)):
            with self.assertRaises(ValueError):
                core.ContentReadout64.from_parent_payload(dict(self.payload, **changes), arm='mean_wall')
        with self.assertRaises(ValueError): self.model('other')

    def test_original_source_packets_beliefs_and_causal_delay_unchanged(self):
        parent = old.CommonMap64.from_payload(self.payload)
        for n, edges in ((2,[(0,1,1.)]), (3,[(0,1,1.),(1,2,1.)])):
            contexts = {(i,j):(i+1).to_bytes(16,'big') for i,j,_ in edges}
            own = maps(n)
            with torch.no_grad():
                expected = old.rollout(parent, own, edges, mode='equality', serialize=True, edge_contexts=contexts)
            for arm in core.ARMS:
                model = self.model(arm)
                with torch.no_grad():
                    actual = core.rollout(model, own, edges, mode='equality', serialize=True, edge_contexts=contexts)
                self.assertEqual(expected['packets'], actual['packets'])
                self.assertTrue(core.audit_rollout(actual, edges, contexts)['passed'])
                for key in ('q','h','mu','learned_q'):
                    self.assertTrue(torch.equal(expected['source_factors'][key], actual['source_factors'][key]),key)
                for hop in range(3):
                    for key in ('means','covariances','codes','features'):
                        self.assertTrue(torch.equal(expected[key][hop], actual[key][hop]), (arm,hop,key))
                self.assertEqual(actual['actual_serialized_application_bytes'],172*4*(n-1))
                if n == 3:
                    changed = own.clone(); changed[:,0,:64] = 1
                    with torch.no_grad(): other = core.rollout(model,changed,edges,mode='equality')
                    for hop in (0,1):
                        self.assertTrue(torch.equal(actual['means'][hop][:,2],other['means'][hop][:,2]))
                    self.assertFalse(torch.equal(actual['means'][2][:,2],other['means'][2][:,2]))

    def test_fused_only_gradients_all_active_groups_and_wall_to_precision_via_mean(self):
        for arm in core.ARMS:
            model = self.model(arm)
            # Remove the equal initial precision values: asymmetric relative
            # precisions make the indirect mean path easy to observe.
            with torch.no_grad():
                model.source.precision_head.weight.add_(torch.linspace(-.002,.002,2048*64).reshape(64,2048))
            frozen = model.frozen_unused_hash()
            result = core.rollout(model,maps(3),[(0,1,1.),(1,2,1.)],mode='equality')
            out = result['decoded'][-1]
            wp = torch.autograd.grad(out['wall_logits'].square().mean(),model.source.precision_head.weight,
                                     retain_graph=True)[0]
            self.assertTrue(torch.isfinite(wp).all())
            self.assertGreater(wp.abs().sum().item(),0)
            sum(v.square().mean() for v in out.values()).backward()
            for group, parameters in model.parameter_groups().items():
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters),group)
                self.assertGreater(sum(p.grad.abs().sum().item() for p in parameters),0,group)
            self.assertEqual(frozen,model.frozen_unused_hash())
            self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))

    def test_failclosed_features_and_no_private_or_covariance_side_channel(self):
        for arm in core.ARMS:
            model = self.model(arm)
            for bad in (torch.zeros(1,127),self.features.double(),torch.full_like(self.features,float('nan'))):
                with self.assertRaises(ValueError): model.decode(bad)
            bad = self.features.clone();bad[...,0] = .0012345
            with self.assertRaises(ValueError): model.decode(bad)
            with self.assertRaises(TypeError): model.decode(self.features,maps(2))
            meta = model.construction_metadata()
            self.assertFalse(meta['source_or_radio_changed'])
            self.assertFalse(meta['covariance_is_provenance'])
            self.assertFalse(meta['communication_100x_claim'])


if __name__ == '__main__':
    unittest.main()
