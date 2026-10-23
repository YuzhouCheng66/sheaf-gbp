"""Minimal native-resolution continuation adapter for v299 joint_control.

No learned parameter or persistent buffer is added, renamed, or reinitialized.
At128 the exact original recognition and two-call content-readout paths run.
At256/512 the original CNN sees native ternary maps and native public xy; only
its final spatial features are average-pooled to the original4x4 head input.
The copied decoder's same four learned blocks run on H/8,H/4,H/2,H feature
grids. This interpolates intermediate FEATURES, never raw maps or final logits.
It changes the high-resolution receptive-field schedule, not the64+64 wire.

Resolution is explicit at decode time and never cached from a private input.
Use the unchanged v295 forest rollout with decode_rounds=(), then decode each
feature stage at its original H. Recipient slices are independent and can be
backpropagated separately; this module does not implement optimizer scheduling.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from scripts import mapping_common_map_v284 as common
from scripts import mapping_content_readout_v288 as content
from scripts import mapping_context_codec_v121 as context
from scripts import mapping_recognition_hops_v277 as recognition
from scripts.mapping_belief_readout_v282 import _hard_mean
from scripts.mapping_implicit_model_v192 import coordinate_grid, latent_quantize
from scripts.mapping_local_capacity_v269 import Q_MIN, Q_MAX

MODEL_KIND = 'multiscale_v299_continuation_v310'
RESOLUTIONS = (128,256,512)


def _resolution(value):
    if type(value) is not int or value not in RESOLUTIONS:
        raise ValueError('Explicit native square resolution128/256/512 required')
    return value


def decoder_extents(resolution):
    size=_resolution(resolution)
    return (size//8,size//4,size//2,size)


def _native_channels(capacity,private):
    """Original known/free/wall/xy order, at original native cell centres."""
    if (not isinstance(private,torch.Tensor) or private.ndim!=3 or private.shape[0]<1
            or private.shape[-2]!=private.shape[-1] or private.dtype==torch.bool
            or private.is_complex() or private.device!=capacity.public_xy.device
            or capacity.public_xy.dtype!=torch.float32
            or not bool(((private==-1)|(private==0)|(private==1)).all())):
        raise ValueError('Same-device native ternary [batch,H,H] required')
    size=_resolution(private.shape[-1])
    if size==128:
        return capacity._private_channels(private)
    xy=coordinate_grid(size,size,device=private.device).reshape(size,size,2).permute(2,0,1).contiguous()
    observed=torch.stack((private>=0,private==0,private==1),1).float()
    return torch.cat((observed,xy[None].expand(len(private),-1,-1,-1)),1)


class NativeRecognitionHops64(recognition.RecognitionHops64):
    """Same source module keys, head sizes, prior and source-only ownership."""
    @classmethod
    def from_source(cls,original):
        if type(original) is not recognition.RecognitionHops64:
            raise ValueError('Original typed recognition source required')
        result=cls(original.capacity,seed=original.seed,
            source_parent_construction=original.source_parent_construction)
        result.load_state_dict(original.state_dict(),strict=True)
        return result

    def factors(self,private):
        if (not isinstance(private,torch.Tensor) or private.ndim!=4
                or min(private.shape[:2])<1 or private.shape[-2]!=private.shape[-1]):
            raise ValueError('Original nonempty native private[B,N,H,H] required')
        size=_resolution(private.shape[-1])
        if size==128:
            return recognition.RecognitionHops64.factors(self,private)
        b,n=private.shape[:2]
        with torch.autocast(device_type=private.device.type,enabled=False):
            native=self.capacity.encoder(_native_channels(self.capacity,private.reshape(b*n,size,size)))
            if native.shape!=(b*n,128,size//32,size//32) or native.dtype!=torch.float32:
                raise ValueError('Original native final CNN feature shape/dtype required')
            # Integer nonoverlapping average pooling; no adaptive GPU pooling
            # implementation or modification of original input pixels.
            ratio=size//128
            features=F.avg_pool2d(native,kernel_size=ratio,stride=ratio).flatten(1)
            if features.shape!=(b*n,2048):
                raise AssertionError('Original2048 head boundary changed')
            mu=latent_quantize(self.capacity.global_head(features).tanh()).reshape(b,n,64)
            learned_q=(recognition.PRECISION_MIN+(recognition.PRECISION_MAX-recognition.PRECISION_MIN)
                *self.precision_head(features).sigmoid()).reshape(b,n,64)
        if not bool(torch.isfinite(mu).all() and torch.isfinite(learned_q).all()):
            raise FloatingPointError('Nonfinite native recognition factor')
        active=(private>=0).flatten(2).any(-1)
        evidence_q=active[...,None].double()*learned_q.double()
        prior=recognition.LATENT_PRIOR/(n*64)
        precision=evidence_q+prior
        return dict(q=torch.diag_embed(precision),h=evidence_q*mu.double(),mu=mu,
            learned_q=learned_q,active=active,prior_precision=prior)


def native_context_decode(decoder,q,h,resolution):
    """Original decoder equations/weights with a native FEATURE-grid schedule.

    The historical _geometry validator is restricted to128, so this function
    validates the new explicit sizes itself. The original public-coordinate
    generator and numeric q/h validator are reused without modification.
    """
    size=_resolution(resolution)
    if size==128:
        return decoder(q,h,128,128)
    context._numeric(q,h,decoder.config.channels)
    if q.ndim!=3 or min(q.shape[:2])<1:
        raise ValueError('Context q/h must have nonempty [B,N,K] shape')
    if q.device!=decoder.output.weight.device or q.dtype!=decoder.output.weight.dtype:
        raise ValueError('Decoder and q/h must share dtype/device')
    b,n=q.shape[:2];count=b*n;side=decoder.config.side
    mean=(h/torch.where(q>0,q,torch.ones_like(q))).reshape(count,decoder.config.planes,side,side)
    precision=q.log1p().reshape_as(mean)
    hidden=decoder.lift(torch.cat((mean,precision,
        context._public(count,side,side,q.device,q.dtype,size,size)),1))
    for extent,block in zip(decoder_extents(size),decoder.blocks):
        hidden=F.interpolate(hidden,(extent,extent),mode='bilinear',align_corners=False)
        hidden=block(torch.cat((hidden,
            context._public(count,extent,extent,q.device,q.dtype,size,size)),1))
    known,wall=decoder.output(hidden).reshape(b,n,2,size,size).unbind(2)
    result=torch.stack((F.logsigmoid(-known),F.logsigmoid(known)+F.logsigmoid(-wall),
        F.logsigmoid(known)+F.logsigmoid(wall)),-1)
    empty=torch.full_like(result,-80.);empty[...,0]=0.
    return torch.where((q==0).all(-1)[...,None,None,None],empty,result)


class MultiscaleV299(content.ContentReadout64):
    """Trainable v299 weights, native input/output, no new learned capacity."""
    def __init__(self,parent,*,continuation_parent_state_sha256=None):
        if type(parent) is not content.ContentReadout64 or parent.arm!='joint_belief':
            raise ValueError('Original v299 joint-belief ContentReadout64 required')
        provenance=parent.weights_hash() if continuation_parent_state_sha256 is None else continuation_parent_state_sha256
        if (not isinstance(provenance,str) or len(provenance)!=64
                or any(c not in '0123456789abcdef' for c in provenance)):
            raise ValueError('Lowercase continuation-parent state SHA256 required')
        expected_xy=coordinate_grid(128,128,device=parent.source.capacity.public_xy.device).reshape(128,128,2).permute(2,0,1).contiguous()
        if not torch.equal(expected_xy,parent.source.capacity.public_xy):
            raise ValueError('Original fixed public128 coordinate buffer changed')
        common.CommonMap64.__init__(self,NativeRecognitionHops64.from_source(parent.source),
            copy.deepcopy(parent.readout),check_parent=False,
            source_construction=parent.source_construction,readout_construction=parent.readout_construction)
        self.arm=parent.arm
        self.parent_state_sha256=parent.parent_state_sha256
        self.base_construction=copy.deepcopy(parent.base_construction)
        self.parent_content_construction=copy.deepcopy(parent.construction_metadata())
        self.continuation_parent_state_sha256=provenance

    @classmethod
    def from_parent_payload(cls,payload):
        """Caller authenticates the selected checkpoint file; no file I/O here."""
        original=content.ContentReadout64.from_payload(payload)
        _check_state_schema(payload['state_dict'],original.state_dict())
        return cls(original)

    def _decode_native_once(self,features,resolution):
        r=self.readout
        if (not isinstance(features,torch.Tensor) or features.ndim not in (2,3)
                or features.shape[-1]!=128 or min(features.shape[:-1])<1
                or features.dtype!=torch.float32 or features.device!=r.lift.weight.device
                or not bool(torch.isfinite(features).all())):
            raise ValueError('Live finite FP32 belief features required')
        _hard_mean(features[...,:64])
        shape,flat=features.shape[:-1],features.reshape(-1,128)
        code=flat[:,:64].contiguous()
        extra=(flat[:,64:]-r.normalization_mean[64:])/r.normalization_std[64:]
        with torch.autocast(device_type=features.device.type,enabled=False):
            lifted=(F.linear(code,r.lift.weight[:,:64].contiguous(),r.lift.bias)
                +F.linear(extra,r.lift.weight[:,64:].contiguous(),None))
            mean_raw,q_raw=lifted.chunk(2,-1)
            q=Q_MIN+(Q_MAX-Q_MIN)*q_raw.sigmoid()
            logp=native_context_decode(r.decoder,q[:,None],(q*mean_raw.tanh())[:,None],resolution)[:,0]
            known=(torch.logsumexp(logp[...,1:],-1)-logp[...,0]).flatten(1)
            wall=(logp[...,2]-logp[...,1]).flatten(1)
        result=dict(known_logits=known.reshape(*shape,resolution*resolution),
            wall_logits=wall.reshape(*shape,resolution*resolution))
        if any(not bool(torch.isfinite(value).all()) for value in result.values()):
            raise FloatingPointError('Nonfinite native decoded logits')
        return result

    def decode(self,features,*,resolution=128):
        size=_resolution(resolution)
        if size==128:
            return content.ContentReadout64.decode(self,features)
        # Preserve the original two shared-weight calls and head selection;
        # there is no shortcut changing the original joint-belief semantics.
        actual=self._decode_native_once(features,size)
        physical=self._decode_native_once(features,size)
        return dict(known_logits=actual['known_logits'],wall_logits=physical['wall_logits'])

    def decode_recipients(self,features,*,resolution,start,stop):
        if (not isinstance(features,torch.Tensor) or features.ndim!=3
                or type(start) is not int or type(stop) is not int
                or not 0<=start<stop<=features.shape[1]):
            raise ValueError('B,N,128 features and nonempty recipient range required')
        return self.decode(features[:,start:stop],resolution=resolution)

    def construction_metadata(self):
        meta=super().construction_metadata()
        meta.update(model_kind=MODEL_KIND,version=310,
            parent_content_construction=copy.deepcopy(self.parent_content_construction),
            continuation_parent_state_sha256=self.continuation_parent_state_sha256,
            continuation_parent_file_pin_required=True,checkpoint_keys_and_weights_reused=True,
            source_or_radio_changed=True,native_source_adapter_changed=True,radio_changed=False,
            native_resolutions=list(RESOLUTIONS),explicit_decode_resolution=True,
            decoder_resolution_cached_from_private=False,
            exact_original128_source_and_decoder_path=True,
            source_native_CNN_then_feature_pool='H128 identity; H256 average2x2/stride2; H512 average4x4/stride4; final128x4x4',
            source_private_channel_order=['known','free','wall','public_x','public_y'],
            input_raster_resized=False,decoder_feature_grid_schedule='H/8,H/4,H/2,H',
            decoder_lift_grid=[16,16],decoder_interpolation='bilinear intermediate hidden features; align_corners=False',
            decoder_public_coordinates='original endpoint-inclusive normalized y/x plus public1/H,1/H at every stage',
            final_logits_resized=False,final_output_convolution_at_native_resolution=True,
            high_resolution_receptive_field_schedule_changed=True,
            new_parameter_count=0,new_persistent_buffer_count=0,
            historical_nested128_metadata_is_checkpoint_provenance=True,
            native_resolution_support_is_not_fidelity_claim=True,
            original_unused_source_decoder_still_frozen=True,
            recipient_decoding_independent=True,full_native_pixels=True,
            registered_precision_profile=4,registered_precision_range=[1e-8,128.],
            intended_unchanged_rollout='mapping_multiroom_forest_v295.rollout with decode_rounds=(); explicit native decode afterward',
            transport_changed=False,communication_100x_claim=False)
        return meta

    def to_payload(self):
        return dict(construction=self.construction_metadata(),
            state_dict={k:v.detach().cpu().clone() for k,v in self.state_dict().items()},
            model_sha256=self.weights_hash())

    @classmethod
    def from_payload(cls,payload):
        if (not isinstance(payload,dict) or not isinstance(payload.get('construction'),dict)
                or not isinstance(payload.get('state_dict'),dict)):
            raise ValueError('Self-contained typed multiscale checkpoint required')
        meta=payload['construction']
        if meta.get('model_kind')!=MODEL_KIND or meta.get('version')!=310:
            raise ValueError('Wrong multiscale continuation family')
        original=content.ContentReadout64.from_payload(dict(
            construction=meta.get('parent_content_construction'),state_dict=payload['state_dict']))
        _check_state_schema(payload['state_dict'],original.state_dict())
        model=cls(original,continuation_parent_state_sha256=meta.get('continuation_parent_state_sha256'))
        if model.construction_metadata()!=meta:
            raise ValueError('Multiscale construction metadata disagrees with implementation')
        if payload.get('model_sha256')!=model.weights_hash():
            raise ValueError('Multiscale state hash mismatch or missing')
        return model


def _check_state_schema(state,expected):
    if set(state)!=set(expected) or any(not isinstance(v,torch.Tensor) or v.dtype!=expected[k].dtype
            or v.shape!=expected[k].shape or not bool(torch.isfinite(v).all()) for k,v in state.items()):
        raise ValueError('Exact original finite tensor names/shapes/dtypes required')
