"""Static diagonal equality GBP on an offline, public BFS spanning forest.

Original own-only recognition factors and the original 64-mean/logC decoder are
unchanged. A new profile4 [1e-8,128] uses the existing 148+24 byte packet layout.
This is NOT distributed tree construction, loopy BP, new evidence acquisition,
calibrated confidence, measured session establishment, or a 100x claim. h is
source distance, R is paid synchronous transmissions, L is local computation.
On a forest a directed cavity contains each original source at most once;
messages REPLACE caches. No posterior is ever reinjected as a private source.
"""
import hashlib
import math
from numbers import Integral

import numpy as np
import torch

from scripts.mapping_adaptive_wire_v207 import AdaptiveCodec64
from scripts.mapping_common_map_v284 import CommonMap64, belief_features
from scripts.mapping_implicit_model_v192 import latent_quantize
from scripts.mapping_recognition_hops_v277 import LATENT_PRIOR, _edges, source_factor_hashes
from scripts import mapping_recurrent_gbp_v213 as recurrent
from scripts import mapping_session_bound_wire_v267 as transport

PROFILE = 4
PACKET_BYTES = 172
MAX_NODES = 16
MAX_ROUNDS = 16
DECODE_ROUNDS = (0, 1, 2, 4, 8)
bind_context = transport.bind_context


class ForestCodec64(AdaptiveCodec64):
    RANGES = dict(AdaptiveCodec64.RANGES, **{})
    RANGES[4] = (1e-8, 128.)


def _node_count(n):
    if isinstance(n, bool) or not isinstance(n, Integral) or not 1 <= n <= MAX_NODES:
        raise ValueError('Static forest supports one through sixteen nodes')
    return int(n)


def validate_forest(edges, n):
    """Reject loops/duplicates; retain isolated nodes and unit edge contexts."""
    n = _node_count(n)
    graph = sorted((min(i,j), max(i,j), w) for i,j,w in _edges(edges,n))
    parents = list(range(n))
    def root(i):
        while parents[i] != i:
            i = parents[i]
        return i
    for i,j,_ in graph:
        a,b = root(i),root(j)
        if a == b:
            raise ValueError('Cycles are forbidden: no repeated-source loopy equality BP')
        parents[b] = a
    return graph


def select_forest(positions_m, radius_m, adjacency=None):
    """Offline public schedule: minimum-ID component root, sorted-neighbor BFS.

    No map, target, decoder output, outcome or performance enters selection.
    Physical links and this centrally supplied schedule have unmeasured setup.
    """
    p = np.asarray(positions_m)
    if p.ndim != 2 or p.shape[1] != 2 or not np.issubdtype(p.dtype,np.floating) or not np.isfinite(p).all():
        raise ValueError('Finite metric positions N by 2 required')
    n = _node_count(len(p))
    if isinstance(radius_m,bool) or not np.isscalar(radius_m) or not math.isfinite(float(radius_m)) or radius_m <= 0:
        raise ValueError('Positive finite communication radius required')
    physical = ((p[:,None,:]-p[None,:,:])**2).sum(-1) <= float(radius_m)**2
    np.fill_diagonal(physical,False)
    if adjacency is not None:
        supplied = np.asarray(adjacency)
        if supplied.dtype != np.bool_ or supplied.shape != (n,n) or not np.array_equal(supplied,physical):
            raise ValueError('Supplied adjacency must equal physical radius adjacency')
    visited=set(); edges=[]; components=[]
    for start in range(n):
        if start in visited:
            continue
        visited.add(start); queue=[start]; component=[]
        for i in queue:
            component.append(i)
            for j in np.flatnonzero(physical[i]).tolist():
                if j not in visited:
                    visited.add(j); queue.append(j); edges.append((min(i,j),max(i,j),1.))
        components.append(sorted(component))
    edges=validate_forest(edges,n)
    selected=np.zeros((n,n),dtype=bool)
    for i,j,_ in edges:
        selected[i,j]=selected[j,i]=True
    return dict(edges=edges,physical_adjacency=physical,selected_adjacency=selected,
        components=components,component_roots=[c[0] for c in components],
        offline_public_schedule=True,distributed_forest_construction=False,
        schedule_setup_measured=False,uses_map_or_target=False)


def reachability(edges, n, rounds):
    n=_node_count(n)
    if type(rounds) is not int or not 0 <= rounds <= MAX_ROUNDS:
        raise ValueError('Zero through sixteen synchronous rounds required')
    graph=validate_forest(edges,n)
    adjacent=np.eye(n,dtype=bool)
    for i,j,_ in graph:
        adjacent[i,j]=adjacent[j,i]=True
    reached=np.eye(n,dtype=bool)
    for _ in range(rounds):
        reached=(adjacent.astype(np.int64)@reached.astype(np.int64))>0
    return reached


def precision_bound(n=16):
    """Conservative tree envelope; includes every quantizer and FP32 cast.

    Every outgoing direction contains <=N-1 sources, depth <=N-1. A source
    precision is <=4+lambda/(N64). Quantized rounding inflates it by at most
    sqrt(high/low)^(1/254) times (1+2^-24) per hop. Repeated rounds do not
    increase path depth or introduce new factors on the same finite forest.
    """
    n=_node_count(n); low,high=ForestCodec64.RANGES[PROFILE]
    multiplier=math.exp(math.log(high/low)/508)*(1+2.**-24)
    bound=(n-1)*(4.+LATENT_PRIOR/(n*64))*multiplier**(n-1)
    return dict(nodes=n,max_path_depth=n-1,source_max=4.+LATENT_PRIOR/(n*64),
        per_hop_precision_inflation=multiplier,outgoing_precision_upper_bound=bound,
        registered_upper=high,strictly_inside=bound<high,includes_fp32_cast=True,
        assumption='Static forest, immutable source-once cavity, no evidence reinjection')


def initialize(q, h, edges):
    if (not isinstance(q,torch.Tensor) or not isinstance(h,torch.Tensor)
            or q.ndim!=4 or h.ndim!=3 or h.shape[-1]!=64 or q.shape!=(*h.shape,64)
            or h.shape[0]<1 or q.dtype!=torch.float64 or h.dtype!=torch.float64
            or q.device!=h.device or not bool(torch.isfinite(q).all() and torch.isfinite(h).all())):
        raise ValueError('Finite same-device FP64 B,N,64 diagonal original Q/h required')
    n=_node_count(h.shape[1]); graph=validate_forest(edges,n)
    diagonal=q.diagonal(dim1=-2,dim2=-1)
    if not bool((diagonal>0).all()) or not torch.equal(q,torch.diag_embed(diagonal)):
        raise ValueError('Positive diagonal original source factors required')
    if bool((diagonal>4.+LATENT_PRIOR/(n*64)).any()):
        raise ValueError('Original precision exceeds the declared recognition/source envelope')
    state=recurrent.initialize(q,h,graph)
    directed=state['directed']; e=len(directed)
    incidence=q.new_zeros(n,e); cavity=q.new_zeros(e,e)
    for a,(i,j,_) in enumerate(directed):
        incidence[j,a]=1.
        for b,(k,l,_) in enumerate(directed):
            if l==i and k!=j:
                cavity[a,b]=1.
    return dict(state,precision_profile=PROFILE,edges=tuple(graph),
        incidence=incidence,cavity=cavity,
        senders=torch.tensor([i for i,_,_ in directed],device=q.device,dtype=torch.long))


def _messages(state, name):
    return (torch.stack([state[name][(i,j)] for i,j,_ in state['directed']],1)
            if state['directed'] else state['h'].new_zeros(state['h'].shape[0],0,64))


def scalar_belief(state):
    precision=state['q'].diagonal(dim1=-2,dim2=-1)+torch.matmul(state['incidence'],_messages(state,'p'))
    info=state['h']+torch.matmul(state['incidence'],_messages(state,'eta'))
    if not bool(torch.isfinite(precision).all() and torch.isfinite(info).all() and (precision>0).all()):
        raise FloatingPointError('Invalid diagonal belief; no floor/repair')
    return info/precision,precision.reciprocal()


def belief(state):
    mean,variance=scalar_belief(state)
    return mean,torch.diag_embed(variance)


def quantize(p, eta):
    """Vectorized literal codebook forward, identity STE, no saturation."""
    if p.shape!=eta.shape or p.shape[-1]!=64 or p.dtype!=torch.float64 or eta.dtype!=torch.float64:
        raise ValueError('Matching FP64 scalar64 natural parameters required')
    pp,hh=p.detach(),eta.detach(); active=pp>0
    low,high=ForestCodec64.RANGES[PROFILE]
    if (not bool(torch.isfinite(pp).all() and torch.isfinite(hh).all()) or bool((pp<0).any())
            or bool(((pp[active]<low)|(pp[active]>high)).any()) or bool((hh[~active]!=0).any())):
        raise ValueError('Precision outside declared profile4; overflow is fatal, no clipping')
    safe=torch.where(active,pp,torch.ones_like(pp))
    value=torch.where(active,hh/safe.sqrt(),torch.zeros_like(hh))
    maximum=value.abs().amax(-1,keepdim=True)
    exponent=torch.ceil(torch.log2(maximum.clamp_min(2.**-8))).clamp_min(-8)
    if bool((exponent>8).any()):
        raise ValueError('Information block exponent overflow; no clipping')
    pc=torch.where(active,1+torch.round(torch.log(safe/low)/math.log(high/low)*254),torch.zeros_like(pp)).long()
    vc=torch.round(value/torch.pow(2.,exponent)*127)
    table=torch.as_tensor(np.concatenate(([0.],np.geomspace(low,high,255))),device=p.device,dtype=torch.float64)
    hp=table[pc]; he=vc/127*torch.pow(2.,exponent)*hp.sqrt()
    hp,he=hp.float().double(),he.float().double()
    return hp+(p-p.detach()),he+(eta-eta.detach()),exponent.squeeze(-1).long()


def advance(state, rounds=1, *, quantized=True, trace=False):
    if type(rounds) is not int or not 0<=rounds<=MAX_ROUNDS or state['iteration']+rounds>MAX_ROUNDS:
        raise ValueError('Total paid synchronous rounds must remain in [0,16]')
    if type(quantized) is not bool or state['precision_profile']!=PROFILE:
        raise ValueError('Explicit float diagnostic or registered profile4 required')
    result=state; history=[]
    for _ in range(rounds):
        cp=result['q'].diagonal(dim1=-2,dim2=-1)[:,result['senders']]+torch.matmul(result['cavity'],_messages(result,'p'))
        ch=result['h'][:,result['senders']]+torch.matmul(result['cavity'],_messages(result,'eta'))
        if not bool(torch.isfinite(cp).all() and torch.isfinite(ch).all() and (cp>0).all()):
            raise FloatingPointError('Invalid immutable-source cavity')
        if quantized and result['directed']:
            pp,hh,exponent=quantize(cp,ch)
        else:
            pp,hh,exponent=cp,ch,None
        nextp={}; nexth={}
        for a,(i,j,_) in enumerate(result['directed']):
            nextp[i,j]=pp[:,a]; nexth[i,j]=hh[:,a]
            if trace:
                history.append(dict(sender=i,receiver=j,round=result['iteration'],precision_profile=PROFILE,
                    raw_p=cp[:,a].detach().clone(),raw_eta=ch[:,a].detach().clone(),
                    decoded_p=pp[:,a].detach().clone(),decoded_eta=hh[:,a].detach().clone(),
                    exponent=None if exponent is None else exponent[:,a].detach().clone()))
        result=dict(result,p=nextp,eta=nexth,iteration=result['iteration']+1,
            message_versions={key:result['iteration'] for key in nextp})
    mean,covariance=belief(result)
    return dict(state=result,mean=mean,covariance=covariance,trace=history)


def wrap_packet(state, raw):
    binding=transport.verify_context(state); d=ForestCodec64.decode(raw)
    if d['mode']!='standardized' or d['precision_profile']!=PROFILE:
        raise ValueError('Profile4 standardized packet required')
    links={(i,j):(nonce,w) for i,j,nonce,w in binding.links}
    key=tuple(sorted((d['sender'],d['receiver'])))
    if key not in links:
        raise ValueError('Packet edge is not in the selected forest')
    nonce,w=links[key]
    return nonce+w+raw


def receive(state, packet):
    """Actual receiver bytes; all attempts, including rejected ones, are billed."""
    binding=transport.verify_context(state)
    if not isinstance(packet,bytes) or len(packet)!=PACKET_BYTES:
        raise ValueError('Exactly 172 bytes required')
    nonce,weight,raw=packet[:16],packet[16:24],packet[24:]
    header=ForestCodec64.HEADER.unpack_from(raw); i,j=header[-3:-1]
    links={(a,b):(n,w) for a,b,n,w in binding.links}
    if (i,j) not in state['p'] or links.get(tuple(sorted((i,j))))!=(nonce,weight):
        return state,False
    d=ForestCodec64.decode(raw)
    if d['mode']!='standardized' or d['precision_profile']!=PROFILE:
        raise ValueError('Unexpected codec profile/mode')
    key=(i,j); sequence=d['sequence']
    if sequence>=MAX_ROUNDS:
        raise ValueError('Sequence outside registered static episode')
    if sequence<state['message_versions'][key]:
        return state,False
    p=torch.from_numpy(d['precision'].astype(np.float32)).to(device=state['h'].device,dtype=torch.float64)[None]
    h=torch.from_numpy(d['eta'].astype(np.float32)).to(device=state['h'].device,dtype=torch.float64)[None]
    if sequence==state['message_versions'][key]:
        if not torch.equal(p,state['p'][key]) or not torch.equal(h,state['eta'][key]):
            raise ValueError('Conflicting same-sequence packet')
        return state,False
    pp,hh,vv=dict(state['p']),dict(state['eta']),dict(state['message_versions'])
    pp[key],hh[key],vv[key]=p,h,sequence
    return dict(state,p=pp,eta=hh,message_versions=vv,iteration=max(state['iteration'],sequence+1)),True


def serialize_trace(state, history):
    packets=[]
    for item in history:
        if item['raw_p'].shape!=(1,64):
            raise ValueError('Literal serialization requires a single world')
        raw,_=ForestCodec64.encode(item['raw_p'][0].cpu().numpy(),item['raw_eta'][0].cpu().numpy(),
            item['sender'],item['receiver'],item['round'],mode='standardized',precision_profile=PROFILE)
        d=ForestCodec64.decode(raw)
        for key,value in (('decoded_p',d['precision']),('decoded_eta',d['eta'])):
            if not torch.equal(torch.from_numpy(value.astype(np.float32)).double(),item[key][0].cpu()):
                raise AssertionError('Vectorized quantizer differs from literal received bytes')
        packets.append(wrap_packet(state,raw))
    return packets


def central_reference(q,h,edges,rounds):
    """FLOAT reference for the ACTUALLY reached original-source subset.

    The caller's Q already has lambda/(N*64) per source. Thus a reached set S
    contains |S|*lambda/(N*64), not a fresh complete prior at every receiver.
    Reference does not enter inference, messages, training labels or decoder.
    """
    state=initialize(q,h,edges)
    mask=torch.as_tensor(reachability(edges,h.shape[1],rounds),device=q.device,dtype=torch.float64)
    p=torch.matmul(mask,q.diagonal(dim1=-2,dim2=-1)); info=torch.matmul(mask,h)
    return dict(mean=info/p,variance=p.reciprocal(),precision=p,information=info,
        reached_sources=mask.bool(),reached_source_counts=mask.sum(-1).long())


def rollout(model,private,edges,*,rounds=8,decode_rounds=DECODE_ROUNDS,
            quantized=True,serialize=False,edge_contexts=None,trace=True):
    if not isinstance(model,CommonMap64) or type(rounds) is not int or not 0<=rounds<=MAX_ROUNDS:
        raise ValueError('Original CommonMap64 subclass and registered static rounds required')
    requested=tuple(decode_rounds)
    if (len(set(requested))!=len(requested) or any(type(k) is not int or not 0<=k<=rounds for k in requested)):
        raise ValueError('Unique explicit decoded round indices within the paid schedule required')
    if serialize and (quantized is not True or private.shape[0]!=1 or edge_contexts is None):
        raise ValueError('Actual serialization needs quantized B1 and pre-established contexts')
    if not serialize and edge_contexts is not None:
        raise ValueError('Contexts apply only to actual serialization')
    factors=model.source.factors(private)
    state=initialize(factors['q'],factors['h'],edges)
    if serialize:
        state=bind_context(state,edge_contexts)
    codes=[]; means=[]; variances=[]; features=[]; states=[]; history=[]; packets=[]; decoded={}
    for r in range(rounds+1):
        if r:
            update=advance(state,1,quantized=quantized,trace=trace or serialize)
            state=update['state']; history.extend(update['trace'])
            if serialize:
                packets.extend(serialize_trace(state,update['trace']))
        mean,variance=scalar_belief(state)
        code=latent_quantize(mean.float())
        feature=belief_features(code,torch.diag_embed(variance))
        codes.append(code);means.append(mean);variances.append(variance);features.append(feature);states.append(state)
        if r in requested:
            decoded[r]=model.decode(feature)
        if state['q'] is not factors['q'] or state['h'] is not factors['h']:
            raise AssertionError('Source posterior was reinjected as new evidence')
    return dict(source_factors=factors,codes=codes,means=means,variances=variances,
        features=features,decoded=decoded,outputs=decoded,decoded_rounds=requested,
        states=states,state=state,trace=history,packets=packets,rounds=rounds,
        quantized=quantized,mode='equality',damping=1.,precision_profile=PROFILE,
        source_versions_unchanged=True,additional_local_refinement_iterations=0,
        wire=dict(packet_bytes=PACKET_BYTES,actual_data_bytes=sum(map(len,packets)),
            actual_packet_count=len(packets),proposed_directed_messages=len(state['directed'])*rounds*private.shape[0],
            protocol_budget_bytes=len(state['directed'])*rounds*private.shape[0]*PACKET_BYTES,
            precision_range=list(ForestCodec64.RANGES[PROFILE]),offline_public_schedule=True,
            session_establishment_measured=False,schedule_setup_measured=False,
            physical_transport_measured=False,communication_100x_claim=False))


def audit_rollout(result,edges,contexts):
    """Independent CPU receive-only replay, including all duplicate/stale costs."""
    factors=result['source_factors']; q=factors['q'].detach().cpu().double(); h=factors['h'].detach().cpu().double()
    if h.shape[0]!=1 or not result['quantized']:
        raise ValueError('One quantized episode required')
    state=bind_context(initialize(q,h,edges),contexts); packets=result['packets']
    width=len(state['directed']); rounds=len(result['codes'])-1
    if len(packets)!=width*rounds:
        raise ValueError('Incomplete paid packet schedule')
    mean_error=0.; variance_error=0.
    def compare(r):
        nonlocal mean_error,variance_error
        m,v=scalar_belief(state)
        wantm=result['means'][r].detach().cpu().double(); wantv=result['variances'][r].detach().cpu().double()
        mean_error=max(mean_error,float((m-wantm).abs().max())); variance_error=max(variance_error,float((v-wantv).abs().max()))
        if not torch.equal(torch.round(latent_quantize(m.float())*127).long(),torch.round(result['codes'][r].detach().cpu()*127).long()):
            raise AssertionError('Actual packet hard64 receiver mismatch')
        if not torch.allclose(m,wantm,atol=1e-11,rtol=1e-11) or not torch.allclose(v,wantv,atol=1e-11,rtol=1e-11):
            raise AssertionError('Actual packet belief mismatch')
    compare(0)
    for r in range(1,rounds+1):
        seen=set()
        for packet in packets[(r-1)*width:r*width]:
            d=ForestCodec64.decode(packet[24:]); key=(d['sender'],d['receiver'])
            if key in seen or key not in state['p'] or d['sequence']!=r-1:
                raise ValueError('Wrong route/sequence in synchronous schedule')
            seen.add(key); state,accepted=receive(state,packet)
            if not accepted:
                raise AssertionError('Fresh frame rejected')
        state=dict(state,iteration=r); compare(r)
    for packet in packets:
        before=state; state,accepted=receive(state,packet)
        if accepted or state is not before:
            raise AssertionError('Duplicate/stale changed fixed evidence')
    if any(s['q'] is not factors['q'] or s['h'] is not factors['h'] for s in result['states']):
        raise AssertionError('Recorded original source factors changed')
    return dict(passed=True,actual_packet_count=len(packets),actual_data_bytes=sum(map(len,packets)),
        additional_forced_replay_bytes=sum(map(len,packets)),
        packet_sha256=hashlib.sha256(b''.join(packets)).hexdigest(),source_hashes=source_factor_hashes(factors),
        max_mean_abs_error=mean_error,max_variance_abs_error=variance_error,
        source_factors_fixed=True,replay_idempotent=True,actual_receive_parity=True,
        profile=PROFILE,forest_only=True,session_setup_measured=False,neural_provenance_recomputed=False)
