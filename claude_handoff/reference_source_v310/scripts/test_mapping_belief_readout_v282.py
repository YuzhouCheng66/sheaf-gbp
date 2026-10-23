"""Independent native128 CPU contracts, synthetic typed weights; no training.

No actual checkpoint, GPU, remote data, target union or TEST split is used.
The new readout's local128 inputs are NOT a128-byte transmitted code.
"""
import copy
import unittest
from unittest import mock

import torch

from scripts import mapping_belief_readout_v282 as readout
from scripts.mapping_context_codec_v121 import ContextCodecConfig, ContextDecoder
from scripts.mapping_local_capacity_v269 import LocalCapacity64
from scripts.mapping_implicit_model_v192 import latent_quantize
from scripts.mapping_recognition_hops_v277 import RecognitionHops64, rollout, source_factor_hashes
from scripts.test_mapping_recognition_hops_v277 import maps


class BeliefReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        config = dict(side=16, planes=4, bits=8, width=128, seed=141021)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(121)
            decoder = ContextDecoder(ContextCodecConfig(**config)).state_dict()
        meta = dict(sha256='a'*64, step=4000, source_only=True, joint_updates=0,
                    family='mapping-context-codec-v121', optimizer_inherited=False)
        capacity = LocalCapacity64(decoder, config, meta, train_decoder=True)
        cls.source = RecognitionHops64(capacity)
        cls.payload = dict(construction=cls.source.construction_metadata(),
                           state_dict=copy.deepcopy(cls.source.state_dict()))
        cls.source.requires_grad_(False).eval()
        # Non-degenerate valid TRAIN-only code/log-covariance features, not targets.
        z = latent_quantize(torch.linspace(-1, 1, 8*64).reshape(8, 1, 64))
        c = torch.diag_embed(torch.exp(torch.linspace(-3, 2, 8*64, dtype=torch.float64).reshape(8, 1, 64)))
        cls.normalization_features = readout.belief_features(z, c).reshape(8, 128)
        cls.norm = readout.fit_train_normalization(cls.normalization_features, split='TRAIN')

    def model(self, arm='posterior_diagonal'):
        return readout.BeliefReadout64.from_parent_payload(self.payload, self.norm, arm)

    def test_frozen_rollout_original_means_covariances_codes_and_actual_packets(self):
        before = self.source.weights_hash()
        for n, edges in ((2, [(0, 1, 1.)]), (3, [(0, 1, 1.), (1, 2, 1.)])):
            private = maps(n)
            contexts = {(i, j): (i+1).to_bytes(16, 'big') for i, j, _ in edges}
            with torch.no_grad():
                original = rollout(self.source, private, edges, serialize=True, edge_contexts=contexts)
                current = readout.belief_rollout(self.source, private, edges, serialize=True, edge_contexts=contexts)
            self.assertEqual(original['packets'], current['packets'])
            self.assertEqual(current['actual_serialized_application_bytes'], 172*4*(n-1))
            self.assertEqual(source_factor_hashes(original['source_factors']), source_factor_hashes(current['source_factors']))
            for hop in range(3):
                for field in ('means', 'covariances', 'codes'):
                    self.assertTrue(torch.equal(original[field][hop], current[field][hop]), (n, hop, field))
                expected = readout.belief_features(original['codes'][hop], original['covariances'][hop])
                self.assertTrue(torch.equal(expected, current['features'][hop]))
                self.assertFalse(current['features'][hop].requires_grad)
                for key in ('q', 'h'):
                    self.assertIs(current['states'][hop][key], current['source_factors'][key])
            self.assertEqual(before, self.source.weights_hash())
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.source.parameters()))

    def test_line_causal_delay_and_no_factor_or_raw_map_input_to_decoder(self):
        edges = [(0, 1, 1.), (1, 2, 1.)]
        a = maps(3)
        b = a.clone()
        b[:, 0, :64] = 1
        first = readout.belief_rollout(self.source, a, edges)
        second = readout.belief_rollout(self.source, b, edges)
        for hop in (0, 1):
            self.assertTrue(torch.equal(first['features'][hop][:, 2], second['features'][hop][:, 2]))
        # An untrained fixture can change its continuous belief by less than
        # one hard8 code bin. Causal delivery must change the actual posterior,
        # not necessarily this subsequently quantized decoder observation.
        self.assertFalse(torch.equal(first['means'][2][:, 2], second['means'][2][:, 2]))
        model = self.model()
        feature = first['features'][2]
        with self.assertRaises(TypeError): model(feature, a)
        for bad in (a, first['source_factors'], torch.zeros(1, 3, 129)):
            with self.assertRaises((ValueError, TypeError)): model(bad)

    def test_features_are_hard_mean_and_diagonal_log_only_detached(self):
        z = latent_quantize(torch.linspace(-1, 1, 128).reshape(1, 2, 64)).requires_grad_()
        covariance = torch.eye(64, dtype=torch.float64)[None, None].repeat(1, 2, 1, 1).requires_grad_()
        features = readout.belief_features(z, covariance)
        self.assertEqual(features.shape, (1, 2, 128))
        self.assertEqual(features.dtype, torch.float32)
        self.assertTrue(torch.equal(features[..., :64], z.detach()))
        self.assertTrue(torch.equal(features[..., 64:], torch.zeros_like(z)))
        self.assertFalse(features.requires_grad)
        offdiag = covariance.detach().clone()
        offdiag[..., 0, 1] = offdiag[..., 1, 0] = .2
        self.assertTrue(torch.equal(features, readout.belief_features(z, offdiag)))
        for bad_z, bad_c in ((z[..., :63], covariance), (z, covariance[..., :63, :63]),
                             (z, torch.zeros_like(covariance)), (z, covariance*float('nan'))):
            with self.assertRaises(ValueError): readout.belief_features(bad_z, bad_c)
        nongrid = z.detach().clone()
        nongrid[0, 0, 0] = .12345
        with self.assertRaises(ValueError): readout.belief_features(nongrid, covariance)

    def test_train_only_normalization_reproducible_and_invalid_inputs_fail(self):
        norm = self.norm
        self.assertEqual(norm['split'], 'TRAIN')
        self.assertEqual(norm['rows'], 8)
        self.assertEqual(norm['ddof'], 0)
        self.assertEqual(norm['std_floor'], 1e-6)
        self.assertEqual(len(norm['mean']), 128)
        self.assertEqual(len(norm['std']), 128)
        self.assertTrue(all(v >= torch.tensor(1e-6).item() for v in norm['std']))
        self.assertEqual(norm, readout.fit_train_normalization(self.normalization_features.clone(), split='TRAIN'))
        self.assertEqual(len(norm['input_sha256']), 64)
        for split in ('VALIDATION', 'TEST'):
            with self.assertRaises(ValueError): readout.fit_train_normalization(self.normalization_features, split=split)
        for invalid in (torch.zeros(0, 128), torch.zeros(2, 127), torch.zeros(2, 128, dtype=torch.float64),
                        torch.full((2, 128), float('nan'))):
            with self.assertRaises(ValueError): readout.fit_train_normalization(invalid, split='TRAIN')
        constant = readout.fit_train_normalization(torch.zeros(3, 128), split='TRAIN')
        self.assertTrue(all(v == torch.tensor(1e-6).item() for v in constant['std']))
        bad_norm = copy.deepcopy(norm)
        bad_norm['std'][0] = 0
        with self.assertRaises(ValueError): readout.BeliefReadout64.from_parent_payload(self.payload, bad_norm, 'posterior_diagonal')

    def test_identical_zero_extra_initialization_original_decoder_parity_and_cpu_rng(self):
        rng = torch.get_rng_state().clone()
        with mock.patch('torch.cuda.manual_seed_all', side_effect=AssertionError('No CUDA RNG')):
            control, candidate = self.model('duplicate_mean'), self.model('posterior_diagonal')
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(control.weights_hash(), candidate.weights_hash())
        self.assertEqual(sum(p.numel() for p in control.trainable_parameters()),
                         sum(p.numel() for p in candidate.trainable_parameters()))
        self.assertTrue(torch.equal(control.lift.weight[:, :64], self.source.capacity.lift.weight))
        self.assertEqual(torch.count_nonzero(control.lift.weight[:, 64:]).item(), 0)
        self.assertFalse(any('encoder' in name or 'precision_head' in name for name, _ in candidate.named_parameters()))
        features = self.normalization_features[:2].reshape(1, 2, 128)
        with torch.no_grad():
            original = self.source.decode(features[..., :64])
            a, b = control(features), candidate(features)
        for key in original:
            self.assertTrue(torch.equal(a[key], b[key]), key)
            self.assertTrue(torch.equal(original[key], b[key]), key)
            self.assertEqual(b[key].shape, (1, 2, 16384))

    def test_actual_covariance_information_not_parameter_difference(self):
        control, candidate = self.model('duplicate_mean'), self.model('posterior_diagonal')
        with torch.no_grad():
            weights = torch.linspace(-.002, .002, 2048*64).reshape(2048, 64)
            control.lift.weight[:, 64:].copy_(weights)
            candidate.lift.weight[:, 64:].copy_(weights)
        features = self.normalization_features[:2].reshape(1, 2, 128)
        changed = features.clone()
        changed[..., 64:] += .7
        with torch.no_grad():
            ca, cb = control(features), control(changed)
            aa, ab = candidate(features), candidate(changed)
        self.assertTrue(all(torch.equal(ca[k], cb[k]) for k in ca))
        self.assertTrue(any(not torch.equal(aa[k], ab[k]) for k in aa))
        for model in (control, candidate):
            differentiable_input = features.clone().requires_grad_()
            output = model(differentiable_input)
            sum(value.square().mean() for value in output.values()).backward()
            self.assertIsNone(differentiable_input.grad)
            for name, p in model.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(model.lift.weight.grad[:, 64:].abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in self.source.parameters()))

    def test_self_contained_restoration_and_wrong_arm_rejected(self):
        model = self.model()
        saved = dict(construction=model.construction_metadata(), state_dict=copy.deepcopy(model.state_dict()))
        with mock.patch('torch.load', side_effect=AssertionError('No ancestor checkpoint read')):
            restored = readout.BeliefReadout64.from_payload(saved)
        self.assertEqual(model.weights_hash(), restored.weights_hash())
        self.assertEqual(model.construction_metadata(), restored.construction_metadata())
        features = self.normalization_features[:2]
        with torch.no_grad():
            before, after = model(features), restored(features)
        for key in before: self.assertTrue(torch.equal(before[key], after[key]))
        with self.assertRaises(ValueError): self.model('full_covariance')
        bad = copy.deepcopy(saved)
        bad['construction']['model_kind'] = 'wrong_family'
        with self.assertRaises(ValueError): readout.BeliefReadout64.from_payload(bad)


if __name__ == '__main__':
    unittest.main()
