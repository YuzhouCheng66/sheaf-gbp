"""Joint diagonal recognition / uncertainty readout with matched tree transport.

finite_beta uses .5*||z_i-z_j||^2; equality uses one common latent variable.
Both registered arms use synchronous UNDAMPED messages, profile3 [1e-8,16],
and actual 172-byte session packets. They are different Gaussian objectives.
The source is a learned surrogate, not an exact decoded observation likelihood.
No raw support, full covariance, or posterior-as-new-source is transmitted.
Only pair / three-node trees are supported. Loops, correlated original sensors,
session establishment, radio overhead and 100x compression are NOT solved here.
"""
import copy
import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.mapping_adaptive_wire_v207 import AdaptiveCodec64
from scripts.mapping_belief_readout_v282 import BeliefReadout64, _hard_mean
from scripts.mapping_implicit_model_v192 import latent_quantize
from scripts.mapping_local_capacity_v269 import _state_hash, Q_MIN, Q_MAX
from scripts.mapping_recognition_hops_v277 import RecognitionHops64, _edges, source_factor_hashes
from scripts import mapping_recurrent_gbp_v213 as recurrent
from scripts import mapping_session_bound_wire_v267 as transport

MODEL_KIND = 'common_map_recognition_v284'
ARMS = ('finite_beta', 'equality')
PROFILE = 3
PACKET_BYTES = 172
bind_context = transport.bind_context


class CommonMapCodec64(AdaptiveCodec64):
    RANGES = dict(AdaptiveCodec64.RANGES)
    RANGES.update({2: (1e-8, 1.), 3: (1e-8, 16.)})


class CommonMap64(nn.Module):
    """Active original E/mu/q plus copied lift/D; unused source lift/D frozen."""
    resolution, latent_dim = 128, 64

    def __init__(self, source, readout, *, check_parent=True,
                 source_construction=None, readout_construction=None):
        super().__init__()
        if not isinstance(source, RecognitionHops64) or not isinstance(readout, BeliefReadout64):
            raise ValueError('Typed source and belief readout required')
        if readout.arm != 'posterior_diagonal':
            raise ValueError('Both arms inherit the posterior_diagonal readout')
        self.source_construction = copy.deepcopy(source.construction_metadata()
            if source_construction is None else source_construction)
        self.readout_construction = copy.deepcopy(readout.construction_metadata()
            if readout_construction is None else readout_construction)
        if check_parent and (readout.source_state_sha256 != source.weights_hash()
                or readout.source_construction != self.source_construction):
            raise ValueError('Readout and frozen source parent provenance disagree')
        self.source, self.readout = source, readout
        self.requires_grad_(True)
        self.source.capacity.lift.requires_grad_(False)
        self.source.capacity.decoder.requires_grad_(False)
        for p in self.parameters():
            p.grad = None
        self.eval()

    @classmethod
    def from_parent_payloads(cls, source_payload, readout_payload):
        return cls(RecognitionHops64.from_payload(source_payload),
                   BeliefReadout64.from_payload(readout_payload))

    @classmethod
    def from_payload(cls, payload):
        if (not isinstance(payload, dict) or not isinstance(payload.get('construction'), dict)
                or not isinstance(payload.get('state_dict'), dict)):
            raise ValueError('Self-contained typed common-map checkpoint required')
        meta, state = payload['construction'], payload['state_dict']
        if meta.get('model_kind') != MODEL_KIND:
            raise ValueError('Wrong common-map family')
        def part(prefix):
            return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        source = RecognitionHops64.from_payload(dict(construction=meta['source_construction'], state_dict=part('source.')))
        readout = BeliefReadout64.from_payload(dict(construction=meta['readout_construction'], state_dict=part('readout.')))
        model = cls(source, readout, check_parent=False, source_construction=meta['source_construction'],
                    readout_construction=meta['readout_construction'])
        model.load_state_dict(state, strict=True)
        if model.construction_metadata() != meta or any(not bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
            raise ValueError('Invalid common-map state or metadata')
        return model

    def parameter_groups(self):
        return dict(encoder=list(self.source.capacity.encoder.parameters()),
            mean_head=list(self.source.capacity.global_head.parameters()),
            precision_head=list(self.source.precision_head.parameters()),
            lift=list(self.readout.lift.parameters()), decoder=list(self.readout.decoder.parameters()))

    def trainable_parameters(self):
        active = [p for group in self.parameter_groups().values() for p in group]
        if len({id(p) for p in active}) != len(active) or any(not p.requires_grad for p in active):
            raise AssertionError('All five disjoint active groups must train')
        if {id(p) for p in active} != {id(p) for p in self.parameters() if p.requires_grad}:
            raise AssertionError('Only the five declared active groups may train')
        return active

    def weights_hash(self):
        return _state_hash(self.state_dict())

    def frozen_unused_hash(self):
        return _state_hash({k: v for k, v in self.state_dict().items()
            if k.startswith(('source.capacity.lift.', 'source.capacity.decoder.'))})

    def construction_metadata(self):
        return dict(model_kind=MODEL_KIND, source_construction=copy.deepcopy(self.source_construction),
            readout_construction=copy.deepcopy(self.readout_construction),
            parameters={k: sum(p.numel() for p in v) for k, v in self.parameter_groups().items()},
            trainable_parameter_count=sum(p.numel() for p in self.trainable_parameters()),
            total_parameters=sum(p.numel() for p in self.parameters()),
            source_factor='Immutable own-only diagonal Gaussian learned surrogate, split public prior',
            decoder_input='hard8 mean64 and inherited TRAIN-normalized log marginal variance64',
            feature_gradients_stopped=False, covariance_is_provenance=False, precision_calibrated=False,
            restriction='identity sheaf subcase, no learned restriction map',
            modes=list(ARMS), registered_damping=1., registered_precision_profile=PROFILE,
            registered_precision_range=[1e-8,16.], packet_bytes=PACKET_BYTES,
            private_decoder_skip=False, original_source_decoder_used=False,
            session_establishment_measured=False, communication_100x_claim=False,
            loopy_or_correlated_evidence_correctness_claim=False, optimizer_inherited=False)

    def decode(self, features):
        """Same v282 numerical readout, now with live mean AND C gradients."""
        r = self.readout
        if (not isinstance(features, torch.Tensor) or features.ndim not in (2,3)
                or features.shape[-1] != 128 or min(features.shape[:-1]) < 1
                or features.dtype != torch.float32 or features.device != r.lift.weight.device
                or not bool(torch.isfinite(features).all())):
            raise ValueError('Live finite FP32 belief features required')
        _hard_mean(features[..., :64])
        shape, flat = features.shape[:-1], features.reshape(-1,128)
        code = flat[:,:64].contiguous()
        extra = (flat[:,64:] - r.normalization_mean[64:]) / r.normalization_std[64:]
        with torch.autocast(device_type=features.device.type, enabled=False):
            lifted = (F.linear(code, r.lift.weight[:,:64].contiguous(), r.lift.bias)
                + F.linear(extra, r.lift.weight[:,64:].contiguous(), None))
            mean_raw, q_raw = lifted.chunk(2,-1)
            q = Q_MIN + (Q_MAX-Q_MIN)*q_raw.sigmoid()
            logp = r.decoder(q[:,None], (q*mean_raw.tanh())[:,None],128,128)[:,0]
            known = (torch.logsumexp(logp[...,1:],-1)-logp[...,0]).flatten(1)
            wall = (logp[...,2]-logp[...,1]).flatten(1)
        return dict(known_logits=known.reshape(*shape,16384), wall_logits=wall.reshape(*shape,16384))


def initialize(q, h, edges, *, precision_profile=PROFILE):
    if (not isinstance(q,torch.Tensor) or not isinstance(h,torch.Tensor)
            or q.ndim!=4 or h.ndim!=3 or h.shape[-1]!=64 or q.shape!=(*h.shape,64)
            or min(h.shape[:2])<1 or h.shape[1] not in (2,3)
            or q.dtype!=torch.float64 or h.dtype!=torch.float64 or q.device!=h.device
            or not bool(torch.isfinite(q).all() and torch.isfinite(h).all())):
        raise ValueError('Finite FP64 diagonal source Q/h, native pair or line3 required')
    diagonal = q.diagonal(dim1=-2,dim2=-1)
    if not bool((diagonal>0).all()) or not torch.equal(q,torch.diag_embed(diagonal)):
        raise ValueError('Strictly positive diagonal original factors required')
    graph = _edges(edges,h.shape[1])
    if len(graph)!=h.shape[1]-1 or precision_profile not in (2,3):
        raise ValueError('Connected pair/line3 tree and declared codec profile required')
    return dict(recurrent.initialize(q,h,graph),precision_profile=precision_profile)


def belief(state):
    """Exact diagonal Gaussian local marginal, no dense inverse or clipping."""
    p = state['q'].diagonal(dim1=-2,dim2=-1)
    eta = state['h']
    incoming_p, incoming_h = [], []
    for node in range(eta.shape[1]):
        pp, hh = p[:,node], eta[:,node]
        for i,j,_ in state['directed']:
            if j==node:
                pp = pp + state['p'][(i,j)]
                hh = hh + state['eta'][(i,j)]
        incoming_p.append(pp); incoming_h.append(hh)
    precision, info = torch.stack(incoming_p,1), torch.stack(incoming_h,1)
    if not bool(torch.isfinite(precision).all() and torch.isfinite(info).all() and (precision>0).all()):
        raise FloatingPointError('Invalid diagonal posterior; no precision floor')
    return info/precision, torch.diag_embed(precision.reciprocal())


def quantize(p, eta, *, precision_profile=PROFILE):
    """Literal receiver FP32 codebook values, identity STE, no clipping."""
    if precision_profile not in (2,3) or p.shape!=eta.shape or p.shape[-1]!=64:
        raise ValueError('Registered scalar64 profile and matching tensors required')
    pp, hh = p.detach().double(), eta.detach().double()
    active = pp>0
    low,high = CommonMapCodec64.RANGES[precision_profile]
    if (not bool(torch.isfinite(pp).all() and torch.isfinite(hh).all()) or bool((pp<0).any())
            or bool(((pp[active]<low)|(pp[active]>high)).any()) or bool((hh[~active]!=0).any())):
        raise ValueError('Natural parameters outside public precision range; no clipping')
    safe = torch.where(active,pp,torch.ones_like(pp))
    value = torch.where(active,hh/safe.sqrt(),torch.zeros_like(hh))
    maximum = value.abs().amax(-1,keepdim=True)
    exponent = torch.ceil(torch.log2(maximum.clamp_min(2.**-8))).clamp_min(-8)
    if bool((exponent>8).any()):
        raise ValueError('Information block exponent overflow; no clipping')
    pc = torch.where(active,1+torch.round(torch.log(safe/low)/math.log(high/low)*254),torch.zeros_like(pp)).long()
    vc = torch.round(value/torch.pow(2.,exponent)*127)
    table = torch.as_tensor(np.concatenate(([0.],np.geomspace(low,high,255))),device=p.device,dtype=torch.float64)
    hp = table[pc]
    he = vc/127*torch.pow(2.,exponent)*hp.sqrt()
    hp,he = hp.float().to(p.dtype),he.float().to(eta.dtype)
    return hp+(p-p.detach()),he+(eta-eta.detach()),exponent.squeeze(-1).long()


def advance(state, rounds=1, *, mode='finite_beta', damping=1., quantized=True, trace=False):
    if mode not in ARMS or type(rounds) is not int or rounds<0 or type(damping) not in (int,float) or damping not in (1.,.5):
        raise ValueError('Registered mode and synchronous rounds required')
    if mode=='equality' and (damping!=1. or state['precision_profile']!=PROFILE):
        raise ValueError('Equality registered only undamped/profile3')
    result, history = state, []
    for _ in range(rounds):
        next_p,next_h = {},{}
        for i,j,w in result['directed']:
            cp = result['q'][:,i].diagonal(dim1=-2,dim2=-1)
            ch = result['h'][:,i]
            # Synchronous cavity: exclude the complete reverse edge, NEVER
            # replace an own source with a posterior or use newly sent values.
            for k,l,_ in result['directed']:
                if l==i and k!=j:
                    cp = cp + result['p'][(k,i)]
                    ch = ch + result['eta'][(k,i)]
            if not bool((cp>0).all() and torch.isfinite(cp).all() and torch.isfinite(ch).all()):
                raise FloatingPointError('Invalid original-evidence cavity')
            if mode=='finite_beta':
                p,h = w*cp/(w+cp),w*ch/(w+cp)
            else:
                p,h = cp,ch
            raw_p = damping*p+(1-damping)*result['p'][(i,j)]
            raw_h = damping*h+(1-damping)*result['eta'][(i,j)]
            if quantized:
                p,h,exponent = quantize(raw_p,raw_h,precision_profile=result['precision_profile'])
            else:
                p,h,exponent = raw_p,raw_h,None
            next_p[(i,j)],next_h[(i,j)] = p,h
            if trace:
                history.append(dict(sender=i,receiver=j,round=result['iteration'],
                    precision_profile=result['precision_profile'], raw_p=raw_p.detach().clone(),
                    raw_eta=raw_h.detach().clone(),decoded_p=p.detach().clone(),decoded_eta=h.detach().clone(),
                    exponent=None if exponent is None else exponent.detach().clone()))
        result = dict(result,p=next_p,eta=next_h,iteration=result['iteration']+1,
            message_versions={key:result['iteration'] for key in next_p})
    mean,covariance = belief(result)
    return dict(state=result,mean=mean,covariance=covariance,trace=history)


def wrap_packet(state, raw):
    binding = transport.verify_context(state)
    decoded = CommonMapCodec64.decode(raw)
    if decoded['mode']!='standardized' or decoded['precision_profile']!=state['precision_profile']:
        raise ValueError('Wrong packet profile/mode')
    links = {(i,j):(nonce,weight) for i,j,nonce,weight in binding.links}
    key = tuple(sorted((decoded['sender'],decoded['receiver'])))
    if key not in links:
        raise ValueError('Inactive packet edge')
    nonce,weight = links[key]
    return nonce+weight+raw


def receive(state, packet):
    """Actual cache replacement; caller must charge ALL attempted packet bytes."""
    binding = transport.verify_context(state)
    if not isinstance(packet,bytes) or len(packet)!=PACKET_BYTES:
        raise ValueError('Exactly172 bytes required')
    nonce,weight,raw = packet[:16],packet[16:24],packet[24:]
    header = CommonMapCodec64.HEADER.unpack_from(raw)
    i,j = header[-3:-1]
    links = {(a,b):(n,w) for a,b,n,w in binding.links}
    if (i,j) not in state['p'] or links.get(tuple(sorted((i,j))))!=(nonce,weight):
        return state,False
    d = CommonMapCodec64.decode(raw)
    if d['precision_profile']!=state['precision_profile'] or d['mode']!='standardized':
        raise ValueError('Wrong current precision profile/mode')
    key,sequence = (i,j),d['sequence']
    if sequence<state['message_versions'][key]:
        return state,False
    p = torch.from_numpy(d['precision'].astype(np.float32)).to(device=state['h'].device,dtype=state['h'].dtype)[None]
    h = torch.from_numpy(d['eta'].astype(np.float32)).to(device=state['h'].device,dtype=state['h'].dtype)[None]
    if sequence==state['message_versions'][key]:
        if not torch.equal(p,state['p'][key]) or not torch.equal(h,state['eta'][key]):
            raise ValueError('Conflicting same-sequence packet')
        return state,False
    pp,hh,vv = dict(state['p']),dict(state['eta']),dict(state['message_versions'])
    pp[key],hh[key],vv[key] = p,h,sequence
    return dict(state,p=pp,eta=hh,message_versions=vv,
                iteration=max(state['iteration'],sequence+1)),True


def serialize_trace(state, history):
    packets=[]
    for item in history:
        if item['raw_p'].shape!=(1,64):
            raise ValueError('Actual serialization requires one world')
        raw,_ = CommonMapCodec64.encode(item['raw_p'][0].cpu().numpy(),item['raw_eta'][0].cpu().numpy(),
            item['sender'],item['receiver'],item['round'],mode='standardized',precision_profile=state['precision_profile'])
        d = CommonMapCodec64.decode(raw)
        for key,value in (('decoded_p',d['precision']),('decoded_eta',d['eta'])):
            expected = torch.from_numpy(value.astype(np.float32)).to(dtype=item[key].dtype)
            if not torch.equal(expected,item[key][0].cpu()):
                raise AssertionError('Actual serializer differs from live hard8 receiver')
        packets.append(wrap_packet(state,raw))
    return packets


def belief_features(code,covariance):
    _hard_mean(code)
    if covariance.shape!=(*code.shape,64) or covariance.device!=code.device or covariance.dtype!=torch.float64:
        raise ValueError('Same-device FP64 actual posterior covariance required')
    diagonal = covariance.diagonal(dim1=-2,dim2=-1)
    if not bool(torch.isfinite(diagonal).all() and (diagonal>0).all()):
        raise FloatingPointError('No covariance flooring or imputation')
    return torch.cat((code,diagonal.log().float()),-1)


def rollout(model,private,edges,*,mode='finite_beta',rounds=2,damping=1.,precision_profile=PROFILE,
            quantized=True,serialize=False,edge_contexts=None,trace=True):
    if not isinstance(model,CommonMap64) or type(rounds) is not int or rounds not in (0,1,2):
        raise ValueError('Typed model and zero/one/two actual message sweeps required')
    if mode not in ARMS or (serialize and (not quantized or private.shape[0]!=1 or edge_contexts is None)):
        raise ValueError('Registered mode; serialization needs quantized single-world contexts')
    if mode=='equality' and (damping!=1. or precision_profile!=PROFILE):
        raise ValueError('Equality registered only undamped/profile3')
    factors = model.source.factors(private)
    state = initialize(factors['q'],factors['h'],edges,precision_profile=precision_profile)
    if serialize:
        state = bind_context(state,edge_contexts)
    codes,means,covariances,features,decoded,states,history,packets = [],[],[],[],[],[],[],[]
    for hop in range(rounds+1):
        if hop:
            update = advance(state,1,mode=mode,damping=damping,quantized=quantized,trace=trace or serialize)
            state = update['state']
            history.extend(update['trace'])
            if serialize:
                packets.extend(serialize_trace(state,update['trace']))
        mean,covariance = belief(state)
        code = latent_quantize(mean.float())
        feature = belief_features(code,covariance)
        codes.append(code); means.append(mean); covariances.append(covariance)
        features.append(feature); decoded.append(model.decode(feature)); states.append(state)
        if state['q'] is not factors['q'] or state['h'] is not factors['h']:
            raise AssertionError('Original source factors were replaced with a posterior')
    return dict(source_factors=factors,codes=codes,means=means,covariances=covariances,features=features,
        decoded=decoded,outputs=decoded,states=states,state=state,trace=history,packets=packets,
        mode=mode,damping=damping,precision_profile=precision_profile,quantized=quantized,
        actual_serialized_application_bytes=sum(map(len,packets)),
        wire=dict(packet_bytes=PACKET_BYTES,actual_data_bytes=sum(map(len,packets)),
            proposed_directed_messages=len(state['directed'])*rounds*private.shape[0],
            actual_packet_count=len(packets),session_establishment_measured=False,
            precision_range=list(CommonMapCodec64.RANGES[precision_profile]),
            communication_100x_claim=False,full_covariance_transmitted=False))


def audit_rollout(result,edges,contexts):
    """CPU receive-only audit of actual packets, NEVER recompute outgoing data."""
    factors = result['source_factors']
    q,h = factors['q'].detach().cpu().double(),factors['h'].detach().cpu().double()
    if h.shape[0]!=1 or not result['quantized']:
        raise ValueError('One actual quantized packet transcript required')
    state = bind_context(initialize(q,h,edges,precision_profile=result['precision_profile']),contexts)
    packets = result['packets']; width=len(state['directed']); rounds=len(result['codes'])-1
    if len(packets)!=width*rounds:
        raise ValueError('Complete fixed synchronous packet schedule required')
    max_mean_error,max_covariance_error=0.,0.
    def compare(hop):
        nonlocal max_mean_error,max_covariance_error
        mean,covariance=belief(state)
        expected_mean=result['means'][hop].detach().cpu().double()
        expected_cov=result['covariances'][hop].detach().cpu().double()
        max_mean_error=max(max_mean_error,float((mean-expected_mean).abs().max()))
        max_covariance_error=max(max_covariance_error,float((covariance-expected_cov).abs().max()))
        if not torch.equal(torch.round(latent_quantize(mean.float())*127).long(),
                           torch.round(result['codes'][hop].detach().cpu()*127).long()):
            raise AssertionError('Actual receive hard8 code mismatch')
        if not torch.allclose(mean,expected_mean,atol=1e-11,rtol=1e-11) or not torch.allclose(covariance,expected_cov,atol=1e-11,rtol=1e-11):
            raise AssertionError('Actual receive mean/covariance mismatch')
    compare(0)
    for hop in range(1,rounds+1):
        seen=set()
        for packet in packets[(hop-1)*width:hop*width]:
            d=CommonMapCodec64.decode(packet[24:]); key=(d['sender'],d['receiver'])
            if key in seen or key not in state['p'] or d['sequence']!=hop-1:
                raise ValueError('Wrong synchronous packet route or sequence')
            seen.add(key)
            state,accepted=receive(state,packet)
            if not accepted:
                raise AssertionError('Fresh actual packet was not accepted')
        state=dict(state,iteration=hop)
        compare(hop)
    replay_bytes=0
    # Last round duplicate and earlier stale attempts are separately charged.
    for packet in packets:
        before=state
        state,accepted=receive(state,packet)
        replay_bytes+=len(packet)
        if accepted or state is not before:
            raise AssertionError('Duplicate/stale reception mutated posterior state')
    source_fixed=all(s['q'] is factors['q'] and s['h'] is factors['h'] for s in result['states'])
    if not source_fixed:
        raise AssertionError('Recorded source factors changed')
    return dict(passed=True,actual_data_bytes=sum(map(len,packets)),actual_packet_count=len(packets),
        additional_forced_replay_bytes=replay_bytes,packet_sha256=hashlib.sha256(b''.join(packets)).hexdigest(),
        source_hashes=source_factor_hashes(factors),source_factors_fixed=True,
        actual_receive_parity=True,replay_idempotent=True,max_mean_abs_error=max_mean_error,
        max_covariance_abs_error=max_covariance_error,
        context_authentication_or_establishment_proven=False,confidence_calibrated=False)
