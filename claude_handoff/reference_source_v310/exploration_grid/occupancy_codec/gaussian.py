"""Exact projected GBP for coisometric restrictions and chosen Gaussian unaries.

R_i R_i^T = I and Lambda_i = q_i I imply isotropic projected cavities.
Every directed edge transmits one precision scalar and K natural parameters.
No precision diagonal/matrix approximation is introduced into this factor model.
Alternatively choose Lambda_i = R_i.T diag(q_i) R_i + beta (I-R_i.T R_i).
Its projected unary is genuinely diagonal, so K precisions plus K naturals are
exact for that chosen model. This is not diagonalizing an arbitrary dense unary.
"""
import struct
from functools import lru_cache
import numpy as np
import torch

HEADER=struct.Struct('<IHHHHHH')  # tick, sender, receiver, K, H, W, round: 16 bytes
PRECISION_LUT=np.r_[0.,np.exp(np.linspace(np.log(.001),np.log(20.),255))].astype(np.float32)


@lru_cache(maxsize=8)
def mean_lut(stalk):
    return (np.arange(-127,128,dtype=np.float64)*(5*stalk**.5)/127).astype(np.float32)


def fp16_wire(value):
    quantized=value.to(torch.float16).to(value.dtype)
    return quantized.detach()+(value-value.detach())


def quantize_message(q,h,stalk,wire='fp16'):
    if wire=='fp16':return fp16_wire(q),fp16_wire(h)
    if wire!='int8_bitmap':raise ValueError('Unknown Gaussian wire codec')
    # For gamma=.05, the GBP Schur complement guarantees 0 < q_out < 20.
    # All projected unary means obey |mean_k| <= 5 sqrt(D); Gaussian cavity
    # means are convex combinations of such means. Use these declared bounds.
    qmin=.001;qmax=20.;bound=5*stalk**.5
    log_code=((q.clamp(qmin,qmax).log()-np.log(qmin))/(np.log(qmax)-np.log(qmin))*254).round()
    # One shared float32 decode table avoids CPU/GPU exp/div roundoff changing
    # a subsequent half-bin rounding decision. The wire and tensor use exactly
    # the same code->float bits, not merely approximately equal exponentials.
    qtable=torch.as_tensor(PRECISION_LUT,device=q.device,dtype=q.dtype)
    mtable=torch.as_tensor(mean_lut(stalk),device=h.device,dtype=h.dtype)
    qvalue=qtable[log_code.long()+1]
    qvalue=torch.where(q>0,qvalue,torch.zeros_like(qvalue))
    mean=(h/q.clamp_min(1e-12)).clamp(-bound,bound)
    mvalue=mtable[(mean/bound*127).round().long()+127]
    hvalue=qvalue*mvalue
    return qvalue.detach()+(q-q.detach()),hvalue.detach()+(h-h.detach())


def gbp(mu,q,r,adjacency,rounds=2,edge_variance=.05,quantize=True,trace=False,wire='fp16'):
    """Slots are [B,sender,receiver]. Reverse messages are excluded from cavities."""
    b,n,d=mu.shape;k=r.shape[-2]
    projected=(r@mu.unsqueeze(-1)).squeeze(-1)
    h0=q*projected
    aq=mu.new_zeros(b,n,n,q.shape[-1]);ah=mu.new_zeros(b,n,n,k)
    mask=adjacency.to(mu.dtype).unsqueeze(-1)
    records=[]
    for _ in range(rounds):
        cq=q[:,:,None]+aq.sum(1)[:,:,None]-aq.transpose(1,2)
        ch=h0[:,:,None]+ah.sum(1)[:,:,None]-ah.transpose(1,2)
        denominator=1.+edge_variance*cq
        oq=cq/denominator;oh=ch/denominator
        if quantize:oq,oh=quantize_message(oq,oh,d,wire)
        aq=oq*mask;ah=oh*mask
        if trace:records.append((aq,ah))
    belief_q=q+aq.sum(1)
    belief_h=h0+ah.sum(1)
    z=belief_h/belief_q.clamp_min(1e-12)
    # Unchanged private nullspace; only the shared projected section changes.
    section=mu+(r.transpose(-1,-2)@(z-projected).unsqueeze(-1)).squeeze(-1)
    section=torch.where((belief_q>0).any(-1,keepdim=True),section,mu)
    return section,belief_q,records


def serialize(tick,sender,receiver,precision,natural,height,width,round_id):
    natural=np.asarray(natural,np.float32)
    values=np.r_[float(precision),natural].astype('<f2')
    if natural.ndim!=1 or not np.isfinite(values).all() or values[0]<=0:
        raise ValueError('Invalid compact Gaussian message')
    return HEADER.pack(tick,sender,receiver,len(natural),height,width,round_id)+values.tobytes()


def deserialize(payload):
    if len(payload)<HEADER.size:raise ValueError('Truncated header')
    meta=HEADER.unpack_from(payload);k=meta[3]
    if len(payload)!=HEADER.size+2*(k+1):raise ValueError('Wrong Gaussian length')
    values=np.frombuffer(payload,dtype='<f2',offset=HEADER.size).astype(np.float32)
    if not np.isfinite(values).all() or values[0]<=0:raise ValueError('Invalid precision / natural vector')
    return meta,float(values[0]),values[1:]


def packet_bytes(k,diagonal=False):return HEADER.size+2*(k+(k if diagonal else 1))


def gossip(mu,q,r,adjacency,rounds=2,edge_variance=None,quantize=True,trace=False):
    """Budget-matched confidence-weighted latent gossip, without GBP cavities.

    Restrictions here must be the public fixed coordinate selector. Learned
    encoder codes are averaged after two synchronous broadcasts. Original local
    evidence is retained at each update; repeated evidence is not called a
    calibrated Gaussian posterior. This is an independently retrained control.
    """
    b,n,d=mu.shape;k=r.shape[-2];projected=mu[...,:k];h0=q*projected
    aq=mu.new_zeros(b,n,n,q.shape[-1]);ah=mu.new_zeros(b,n,n,k)
    mask=adjacency.to(mu.dtype).unsqueeze(-1);records=[]
    for _ in range(rounds):
        oq=(q+aq.sum(1))[:,:,None].expand(b,n,n,q.shape[-1])
        oh=(h0+ah.sum(1))[:,:,None].expand(b,n,n,k)
        if quantize:oq=fp16_wire(oq);oh=fp16_wire(oh)
        aq=oq*mask;ah=oh*mask
        if trace:records.append((aq,ah))
    belief_q=q+aq.sum(1);z=(h0+ah.sum(1))/belief_q.clamp_min(1e-12)
    shared=torch.where((belief_q>0).any(-1,keepdim=True),z,projected)
    return torch.cat((shared,mu[...,k:]),-1),belief_q,records
