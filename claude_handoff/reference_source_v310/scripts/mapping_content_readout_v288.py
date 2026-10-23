"""One readout factorization, with unchanged source factors and equality radio.

Both arms execute TWO original decoder calls using the SAME parameters.
Knownness always receives the real posterior. Only mean_wall's wall call uses
the inherited TRAIN-normalization mean as log C, hence a zero extra feature.
This is a plug-in physical-field bias, not an exact likelihood/covariance claim.
Parent checkpoint file authentication remains the caller's responsibility.
"""
import copy

import torch

from scripts import mapping_common_map_v284 as original


MODEL_KIND = 'content_evidence_readout_v288'
ARMS = ('joint_belief', 'mean_wall')
PARENT_CHECKPOINT = 'D:/sheaf-gbp-artifacts/occupancy_codec_20261017/common-map-v284/equality/validated-004000.pt'
PARENT_CHECKPOINT_SHA256 = 'cca8fec345f7b6faf1d69010e5dd2529b6e30d6d1763a241d3e3485c07716e70'


class ContentReadout64(original.CommonMap64):
    """No new modules, tensors, trainable parameters, source or wire family."""

    def __init__(self, base, *, arm, parent_state_sha256=None):
        if type(base) is not original.CommonMap64 or arm not in ARMS:
            raise ValueError('Original CommonMap64 and registered readout arm required')
        provenance = base.weights_hash() if parent_state_sha256 is None else parent_state_sha256
        if (not isinstance(provenance, str) or len(provenance) != 64
                or any(c not in '0123456789abcdef' for c in provenance)):
            raise ValueError('Lowercase parent model-state SHA256 required')
        # Supply historical construction records explicitly: unused source D is
        # intentionally frozen, so do not call its all-trainable metadata API.
        super().__init__(copy.deepcopy(base.source), copy.deepcopy(base.readout),
                         check_parent=False,
                         source_construction=base.source_construction,
                         readout_construction=base.readout_construction)
        self.arm = arm
        self.parent_state_sha256 = provenance
        self.base_construction = copy.deepcopy(base.construction_metadata())

    @classmethod
    def from_parent_payload(cls, payload, *, arm):
        base = original.CommonMap64.from_payload(payload)
        if ('arm' in payload and payload['arm'] != 'equality') or (
                'step' in payload and payload['step'] != 4000):
            raise ValueError('Registered parent is equality4000, not another run or step')
        if 'model_sha256' in payload and payload['model_sha256'] != base.weights_hash():
            raise ValueError('Parent recorded state hash disagrees with its tensors')
        return cls(base, arm=arm)

    @classmethod
    def from_payload(cls, payload):
        if (not isinstance(payload, dict) or not isinstance(payload.get('construction'), dict)
                or not isinstance(payload.get('state_dict'), dict)):
            raise ValueError('Self-contained typed content-readout checkpoint required')
        meta = payload['construction']
        if (meta.get('model_kind') != MODEL_KIND or meta.get('arm') not in ARMS
                or not isinstance(meta.get('base_construction'), dict)):
            raise ValueError('Wrong content-readout family or missing base construction')
        # Names and shapes are identical to the original family. This validates
        # live trained tensors without loading ancestors or asserting that they
        # still equal the historical source/readout state hashes.
        base = original.CommonMap64.from_payload(dict(
            construction=meta['base_construction'], state_dict=payload['state_dict']))
        model = cls(base, arm=meta['arm'], parent_state_sha256=meta.get('parent_state_sha256'))
        if model.construction_metadata() != meta:
            raise ValueError('Content-readout metadata disagrees with actual construction')
        if 'model_sha256' in payload and payload['model_sha256'] != model.weights_hash():
            raise ValueError('Recorded content-readout state hash disagrees with tensors')
        return model

    def construction_metadata(self):
        meta = super().construction_metadata()
        meta.update(model_kind=MODEL_KIND, arm=self.arm,
                    base_construction=copy.deepcopy(self.base_construction),
                    parent_state_sha256=self.parent_state_sha256,
                    parent_file_pin_verification_required=True,
                    decoder_calls_per_belief=2, decoder_weights_shared=True,
                    new_parameter_count=0, source_or_radio_changed=False,
                    known_input='Actual hard64 mean and actual inherited TRAIN-normalized logdiagC',
                    wall_input=('Actual hard64 mean and actual logdiagC' if self.arm == 'joint_belief'
                                else 'Actual hard64 mean; fixed inherited TRAIN mean logdiagC gives normalized zero'),
                    neutral_covariance_is_runtime_observation=False,
                    wall_is_plugin_estimate=self.arm == 'mean_wall', exact_observed_likelihood_claim=False,
                    initial_parent_wall_function_preserved=self.arm == 'joint_belief')
        return meta

    def decode(self, features):
        # Original validation executes first for BOTH arms, rejecting malformed
        # shape/device/dtype/nonfinite features and non-hard8 mean values.
        actual = original.CommonMap64.decode(self, features)
        if self.arm == 'joint_belief':
            wall_features = features
        else:
            neutral = self.readout.normalization_mean[64:].expand_as(features[..., 64:])
            wall_features = torch.cat((features[..., :64], neutral), dim=-1)
        physical = original.CommonMap64.decode(self, wall_features)
        return dict(known_logits=actual['known_logits'], wall_logits=physical['wall_logits'])


# Existing strict isinstance guard accepts the subclass; no forked solver.
rollout = original.rollout
audit_rollout = original.audit_rollout

