"""Differentiable persistent scalar-radio GBP, source replacement not feedback.

Independent extension of v208; the original sealed solver is untouched.
State carries immutable-version private factors and directed replacement
messages. No posterior is ever used as a new unary observation.
"""
import numpy as np
import torch
from scripts.mapping_evidence_gbp_v208 import gaussian_belief,quantize_standardized
from scripts.mapping_adaptive_wire_v207 import AdaptiveCodec64


def initialize(q,h,edges,versions=None):
    if q.ndim!=4 or q.shape[:3]!=h.shape or q.shape[-1]!=q.shape[-2] or q.shape[-1]!=64:raise ValueError('Batched64D factors required')
    b,n,d=h.shape;directed=[];seen=set()
    for i,j,w in edges:
        key=tuple(sorted((i,j)))
        if i==j or not 0<=i<n or not 0<=j<n or key in seen or w<=0:raise ValueError('Invalid edge')
        seen.add(key);directed.extend([(i,j,float(w)),(j,i,float(w))])
    versions=torch.zeros((b,n),device=q.device,dtype=torch.long) if versions is None else versions.clone()
    if versions.shape!=(b,n) or versions.dtype!=torch.long or (versions<0).any():raise ValueError('Invalid source versions')
    return dict(q=q,h=h,source_versions=versions,directed=tuple(directed),iteration=0,
        p={(i,j):q.new_zeros(b,d) for i,j,_ in directed},eta={(i,j):q.new_zeros(b,d) for i,j,_ in directed},
        message_versions={(i,j):-1 for i,j,_ in directed})


def replace_sources(state,q,h,versions):
    if q.shape!=state['q'].shape or h.shape!=state['h'].shape or versions.shape!=state['source_versions'].shape or versions.dtype!=torch.long or (versions<0).any():raise ValueError('Bad replacement source')
    equal=versions==state['source_versions'];changed=versions>state['source_versions']
    if not torch.equal(q.detach()[equal],state['q'].detach()[equal]) or not torch.equal(h.detach()[equal],state['h'].detach()[equal]):raise ValueError('Conflicting equal source version')
    if not changed.any():return state
    # Retain the actual old differentiable source for unchanged/stale entries.
    # New observations REPLACE Q/h, never add to it or add a posterior.
    return dict(state,q=torch.where(changed[...,None,None],q,state['q']),h=torch.where(changed[...,None],h,state['h']),
        source_versions=torch.where(changed,versions,state['source_versions']))


def _stack(state,cache):
    b,n,d=state['h'].shape;zero=state['q'].new_zeros(b,d)
    return torch.stack([torch.stack([cache.get((i,j),zero) for j in range(n)],1) for i in range(n)],1)


def belief(state):
    return gaussian_belief(state['q'],state['h'],_stack(state,state['p']),_stack(state,state['eta']))[:2]


def advance(state,rounds,*,damping=.5,quantized=True,trace=False,check=True):
    if rounds<0 or not 0<damping<=1:raise ValueError('Invalid rounds or damping')
    q,h=state['q'],state['h'];cache_p=state['p'];cache_eta=state['eta'];packets=[]
    negatives=q.new_zeros((),dtype=torch.long)
    for step in range(rounds):
        mean,covariance,a,b=gaussian_belief(q,h,_stack(state,cache_p),_stack(state,cache_eta))
        diagonal=torch.diagonal(covariance,dim1=-2,dim2=-1)
        alpha=diagonal.reciprocal()-a;beta=mean/diagonal-b
        new_p={};new_eta={}
        for i,j,w in state['directed']:
            cp=alpha[:,i]+a[:,i]-cache_p[(j,i)];ch=beta[:,i]+b[:,i]-cache_eta[(j,i)]
            negatives=negatives+(cp<0).sum()
            if check and (cp<-1e-7).any():raise FloatingPointError('Negative cavity precision')
            cp=cp.clamp_min(0)
            raw_p=damping*w*cp/(w+cp)+(1-damping)*cache_p[(i,j)]
            raw_eta=damping*w*ch/(w+cp)+(1-damping)*cache_eta[(i,j)]
            if quantized:out_p,out_eta,exponent=quantize_standardized(raw_p,raw_eta,check=check)
            else:out_p,out_eta,exponent=raw_p,raw_eta,None
            new_p[(i,j)]=out_p;new_eta[(i,j)]=out_eta
            if trace:packets.append(dict(sender=i,receiver=j,round=state['iteration']+step,raw_p=raw_p.detach(),raw_eta=raw_eta.detach(),
                decoded_p=out_p.detach(),decoded_eta=out_eta.detach(),exponent=exponent))
        cache_p,cache_eta=new_p,new_eta
    end=state['iteration']+rounds
    result=dict(state,p=cache_p,eta=cache_eta,iteration=end,
        message_versions={key:end-1 if rounds else version for key,version in state['message_versions'].items()})
    mean,covariance=belief(result)
    return dict(state=result,mean=mean,covariance=covariance,trace=packets,negative_cavity_count=negatives,
        application_byte_budget_per_world=148*len(state['directed'])*rounds)


def redeliver_packet(state,packet):
    """B=1 wire ingestion used for actual serialized duplicate/stale audits.

    Caller counts EVERY delivered packet, accepted or not. This function does
    not silently fabricate a source update from incoming information.
    """
    if state['h'].shape[0]!=1:raise ValueError('Wire ingress represents one world')
    decoded=AdaptiveCodec64.decode(packet);key=(decoded['sender'],decoded['receiver'])
    if key not in state['p'] or decoded['mode']!='standardized' or decoded['precision_profile']!=1:raise ValueError('Wrong edge/profile')
    version=decoded['sequence'];old=state['message_versions'][key]
    if version<old:return state,False
    p=torch.as_tensor(decoded['precision'][None],device=state['q'].device,dtype=state['q'].dtype)
    eta=torch.as_tensor(decoded['eta'][None],device=state['q'].device,dtype=state['q'].dtype)
    if version==old:
        if not torch.equal(p,state['p'][key]) or not torch.equal(eta,state['eta'][key]):raise ValueError('Conflicting equal packet version')
        return state,False
    cp=dict(state['p']);ch=dict(state['eta']);cv=dict(state['message_versions'])
    cp[key]=p;ch[key]=eta;cv[key]=version
    return dict(state,p=cp,eta=ch,message_versions=cv,iteration=max(state['iteration'],version+1)),True
