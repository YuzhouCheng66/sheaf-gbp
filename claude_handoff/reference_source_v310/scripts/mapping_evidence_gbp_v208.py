"""Differentiable dense-local, scalar-radio GBP with v207 hard wire forward.

Source Q/h never absorb previous posteriors. Every synchronous directed cache
is replaced. Straight-through derivatives are a TRAIN surrogate, not a claim
that lossy packet quantization is differentiable or Gaussian marginals exact.
"""
import math
import numpy as np
import torch


def quantize_standardized(precision,eta,*,check=True):
    """v207 standardized/profile1: hard148B packet values, STE derivatives.

    Hard code decisions use detached FP64, matching the actual NumPy serializer.
    Gaussian compute and stored decoded natural parameters remain caller dtype
    (normally FP32). The scale exponent is real charged header information.
    """
    p=precision.detach().double();h=eta.detach().double();active=p>0
    if check:
        if not torch.isfinite(p).all() or not torch.isfinite(h).all():raise ValueError('Nonfinite natural parameters')
        if ((active & ((p<1e-3)|(p>1.))) | ((~active)&(h!=0))).any():raise ValueError('Outside registered precision profile')
    safe_p=torch.where(active,p,torch.ones_like(p))
    value=torch.where(active,h/safe_p.sqrt(),torch.zeros_like(h))
    maximum=value.abs().amax(-1,keepdim=True)
    exponent=torch.ceil(torch.log2(maximum.clamp_min(2.**-8))).clamp_min(-8)
    if check and (exponent>8).any():raise ValueError('Outside block-floating vector range')
    scale=torch.pow(2.,exponent)
    pc=torch.where(active,1+torch.round(torch.log(safe_p/1e-3)/math.log(1000.)*254),torch.zeros_like(p)).long()
    if check and ((pc<0)|(pc>255)).any():raise ValueError('Out of precision codes')
    table=torch.as_tensor(np.concatenate(([0.],np.geomspace(1e-3,1.,255))),device=p.device,dtype=torch.float64)
    hard_p=table[pc]
    vc=torch.round(value/scale*127)
    if check and (vc.abs()>127).any():raise ValueError('No silent clipping')
    hard_value=vc/127*scale
    hard_eta=hard_value*hard_p.sqrt()
    hard_p=hard_p.to(precision.dtype);hard_eta=hard_eta.to(eta.dtype)
    p_out=hard_p+(precision-precision.detach())
    # Identity STE in natural coordinates. The forward is the complete codec,
    # not independently rounded eta with an inconsistent precision.
    eta_out=hard_eta+(eta-eta.detach())
    return p_out,eta_out,exponent.squeeze(-1).long()


def gaussian_belief(q,h,precision,eta):
    incoming_p=precision.sum(1);incoming_eta=eta.sum(1)
    matrix=q+torch.diag_embed(incoming_p)
    chol=torch.linalg.cholesky(matrix)
    covariance=torch.cholesky_inverse(chol)
    mean=torch.cholesky_solve((h+incoming_eta).unsqueeze(-1),chol).squeeze(-1)
    return mean,covariance,incoming_p,incoming_eta


def scalar_gbp(q,h,edges,*,rounds=2,damping=.5,quantized=True,trace=False,check=True):
    """q[B,N,D,D],h[B,N,D]; unique isotropic weighted undirected edges."""
    if q.ndim!=4 or q.shape[:3]!=h.shape or q.shape[-1]!=q.shape[-2]:raise ValueError('Invalid batched Gaussian shapes')
    b,n,d=h.shape
    if rounds<0 or not 0<damping<=1 or (quantized and d!=64):raise ValueError('Invalid rounds/damping/dimension')
    directed=[];seen=set()
    for i,j,w in edges:
        key=tuple(sorted((i,j)))
        if i==j or not 0<=i<n or not 0<=j<n or key in seen or w<=0:raise ValueError('Invalid edge')
        seen.add(key);directed.extend([(i,j,w),(j,i,w)])
    cache_p={(i,j):q.new_zeros(b,d) for i,j,_ in directed}
    cache_h={(i,j):q.new_zeros(b,d) for i,j,_ in directed}
    packets=[];negative_count=q.new_zeros((),dtype=torch.int64)
    def stack_cache(cache):
        zero=q.new_zeros(b,d)
        return torch.stack([torch.stack([cache.get((i,j),zero) for j in range(n)],1) for i in range(n)],1)
    for step in range(rounds):
        precision=stack_cache(cache_p);information=stack_cache(cache_h)
        mean,covariance,a,bb=gaussian_belief(q,h,precision,information)
        diagonal=torch.diagonal(covariance,dim1=-2,dim2=-1)
        alpha=diagonal.reciprocal()-a;beta=mean/diagonal-bb
        proposed_p={};proposed_h={}
        for i,j,w in directed:
            cp=alpha[:,i]+a[:,i]-cache_p[(j,i)]
            ch=beta[:,i]+bb[:,i]-cache_h[(j,i)]
            negative_count=negative_count+(cp<0).sum()
            if check and (cp<-1e-7).any():raise FloatingPointError('Negative cavity precision')
            # Theoretical cp>=source prior. Count ANY negative value; training
            # requires this count to stayzero, rather than hiding a ridge.
            cp=cp.clamp_min(0)
            raw_p=damping*w*cp/(w+cp)+(1-damping)*cache_p[(i,j)]
            raw_h=damping*w*ch/(w+cp)+(1-damping)*cache_h[(i,j)]
            if quantized:out_p,out_h,exponent=quantize_standardized(raw_p,raw_h,check=check)
            else:out_p,out_h,exponent=raw_p,raw_h,None
            proposed_p[(i,j)]=out_p;proposed_h[(i,j)]=out_h
            if trace:packets.append(dict(sender=i,receiver=j,round=step,raw_p=raw_p.detach(),raw_eta=raw_h.detach(),decoded_p=out_p.detach(),decoded_eta=out_h.detach(),exponent=exponent))
        cache_p,cache_h=proposed_p,proposed_h
    mean,covariance,_,_=gaussian_belief(q,h,stack_cache(cache_p),stack_cache(cache_h))
    return dict(mean=mean,covariance=covariance,trace=packets,negative_cavity_count=negative_count,
        directed_packet_budget_per_world=2*len(edges)*rounds,application_byte_budget_per_world=148*2*len(edges)*rounds)
