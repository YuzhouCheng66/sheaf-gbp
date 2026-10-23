"""A finite Gaussian natural-parameter vocabulary, not a map decoder.

For two accurate/static cell facts with learned positive weights a and typed
natural atoms b, q=sum(a*n), h=sum(b[type]*n). At an outgoing second-round
hard-GBP cavity, n<=m<=N (the posterior's N^2 mass is NOT transmitted).
This finite family admits point indices instead of two FP16 scalars. Receiving
code only reconstructs q/h; it does not return occupancy to a neural decoder.

This does NOT apply to the old cross-key-floor encoder, arbitrary Gaussian
factors, noisy/conflicting cell facts, unknown coordinates, soft edges or more
than two rounds. Close/aliased points and floating residuals must be audited.
The four closed-form candidate types implement a quantizer, not an iterative
occupancy inverse or a universal nearest-neighbor/injectivity certificate.
"""
import math
import struct
import numpy as np
import torch
from torch import nn
from occupancy_codec.gaussian import HEADER

COUNT=struct.Struct('<H');ENTRY=struct.Struct('<HB');CODEC_VERSION=3


class PairVocabulary(nn.Module):
    def __init__(self,precision_atoms,natural_atoms,max_mass=32):
        super().__init__();a=torch.as_tensor(precision_atoms).detach().clone().float()
        b=torch.as_tensor(natural_atoms).detach().clone().float()
        if a.ndim!=2 or a.shape[-1]!=2 or b.shape!=(*a.shape,2):raise ValueError('Two-cell atom shapes required')
        if not torch.isfinite(a).all() or not torch.isfinite(b).all() or not (a>0).all():raise ValueError('Finite positive precision atoms required')
        if not 1<=max_mass<=255:raise ValueError('Declared uint8 mass limit required')
        self.groups=len(a);self.max_mass=max_mass
        modes=torch.tensor([[0,0],[0,1],[1,0],[1,1]],device=a.device)
        group=torch.arange(len(a),device=a.device)[:,None,None];slots=torch.arange(2,device=a.device)[None,None]
        typed=b[group,slots,modes[None]].double()
        matrix=torch.stack((a.double()[:,None,:].expand(-1,4,-1),typed),-2)
        if torch.linalg.det(matrix).abs().min()<1e-9:raise ValueError('Singular pair atom coordinates')
        self.register_buffer('precision_atoms',a);self.register_buffer('natural_atoms',b)
        self.register_buffer('modes',modes);self.register_buffer('inverse',torch.linalg.inv(matrix))
        self.register_buffer('typed',typed)

    def _validate(self,codes,mass):
        if codes.shape[-1]!=self.groups or mass.shape!=codes.shape[:-1]:raise ValueError('Code/mass dimensions disagree')
        if not torch.isfinite(codes.float()).all() or not torch.equal(codes,codes.round()):raise ValueError('Integer Gaussian indices required')
        if not torch.isfinite(mass.float()).all() or not torch.equal(mass,mass.round()):raise ValueError('Integer factor mass required')
        if (mass<0).any() or (mass>self.max_mass).any():raise ValueError('Out of declared mass domain')
        base=2*mass.long()+1
        if (codes<0).any() or (codes>=base[...,None].square()).any():raise ValueError('Invalid Gaussian codepoint index')
        return base

    def decode(self,codes,mass):
        base=self._validate(codes,mass);first=codes%base[...,None];second=codes//base[...,None]
        symbols=torch.stack((first,second),-1);counts=((symbols+1)//2).float();types=(symbols-1)%2
        values=self.natural_atoms.expand(*codes.shape[:-1],self.groups,2,2).gather(-1,types[...,None]).squeeze(-1)
        return (counts*self.precision_atoms).sum(-1),(counts*values).sum(-1)

    @torch.no_grad()
    def encode(self,q,h,mass):
        if q.shape!=h.shape or q.shape[-1]!=self.groups or mass.shape!=q.shape[:-1]:raise ValueError('Natural-parameter dimensions disagree')
        if not torch.isfinite(q).all() or not torch.isfinite(h).all() or (q<0).any():raise ValueError('Invalid Gaussian statistics')
        if not torch.isfinite(mass.float()).all() or not torch.equal(mass,mass.round()) or (mass<0).any() or (mass>self.max_mass).any():
            raise ValueError('Mass outside registered integer two-round domain')
        value=torch.stack((q,h),-1).double()
        count=torch.einsum('gsij,...gj->...gsi',self.inverse,value).round().clamp_min(0)
        count=torch.minimum(count,mass[...,None,None,None].double())
        cq=(count*self.precision_atoms[:,None].double()).sum(-1);ch=(count*self.typed).sum(-1)
        scale_q=self.precision_atoms.sum(-1).clamp_min(1.).double()
        scale_h=self.natural_atoms.abs().amax((-1,-2)).clamp_min(1.).double()
        score=((cq-q[...,None])/scale_q[:,None]).square()+((ch-h[...,None])/scale_h[:,None]).square()
        selected=score.argmin(-1)
        chosen=count.gather(-2,selected[...,None,None].expand(*selected.shape,1,2)).squeeze(-2).long()
        types=self.modes[selected]
        symbols=torch.where(chosen>0,2*chosen-1+types,0)
        base=2*mass.long()+1;codes=symbols[...,0]+base[...,None]*symbols[...,1]
        reconstructed_q,reconstructed_h=self.decode(codes,mass)
        return codes,reconstructed_q,reconstructed_h


def bit_width(mass):return math.ceil(math.log2((2*mass+1)**2))


def pack_indices(values,width):
    values=np.asarray(values,np.uint32).reshape(-1)
    bits=((values[:,None]>>np.arange(width,dtype=np.uint32))&1).astype(np.uint8).reshape(-1)
    return np.packbits(bits,bitorder='little').tobytes()


def unpack_indices(body,count,width):
    if len(body)!=(count*width+7)//8:raise ValueError('Wrong packed-index byte length')
    bits=np.unpackbits(np.frombuffer(body,np.uint8),bitorder='little')
    if bits[count*width:].any():raise ValueError('Nonzero padding bits')
    return (bits[:count*width].reshape(count,width).astype(np.uint32)<<np.arange(width,dtype=np.uint32)).sum(-1,dtype=np.uint32)


def serialize_vocabulary(tick,sender,receiver,codes,mass,height,width,round_id,max_mass=32):
    codes=np.asarray(codes);mass=np.asarray(mass)
    if codes.ndim!=2 or mass.shape!=(len(codes),):raise ValueError('Tile arrays required')
    if not np.isfinite(mass).all() or not np.equal(mass,np.round(mass)).all() or (mass<0).any() or (mass>max_mass).any():
        raise ValueError('Mass outside vocabulary domain')
    active=np.flatnonzero(mass>0)
    if not len(active):return None
    if len(codes)>65536 or len(active)>65535:raise ValueError('Tile addressing/count limit')
    entries=[]
    for tile in active:
        m=int(mass[tile]);v=codes[tile];maximum=(2*m+1)**2
        if not np.equal(v,np.round(v)).all() or (v<0).any() or (v>=maximum).any():raise ValueError('Invalid Gaussian codepoint')
        entries.append(ENTRY.pack(int(tile),m)+pack_indices(v,bit_width(m)))
    return HEADER.pack(tick,sender,receiver,codes.shape[1]+1,height,width,round_id)+bytes([CODEC_VERSION])+COUNT.pack(len(active))+b''.join(entries)


def deserialize_vocabulary(payload,tiles,groups,max_mass=32):
    if len(payload)<HEADER.size+1+COUNT.size:raise ValueError('Truncated vocabulary header')
    meta=HEADER.unpack_from(payload)
    if meta[3]!=groups+1 or payload[HEADER.size]!=CODEC_VERSION:raise ValueError('Gaussian dimension/codec mismatch')
    count=COUNT.unpack_from(payload,HEADER.size+1)[0];at=HEADER.size+1+COUNT.size;seen=set();entries=[]
    for _ in range(count):
        if at+ENTRY.size>len(payload):raise ValueError('Truncated vocabulary tile')
        tile,mass=ENTRY.unpack_from(payload,at);at+=ENTRY.size
        if tile>=tiles or tile in seen or not 1<=mass<=max_mass:raise ValueError('Invalid tile ID/mass')
        seen.add(tile);size=(groups*bit_width(mass)+7)//8
        if at+size>len(payload):raise ValueError('Truncated Gaussian indices')
        code=unpack_indices(payload[at:at+size],groups,bit_width(mass));at+=size
        if (code>=(2*mass+1)**2).any():raise ValueError('Out-of-vocabulary point')
        entries.append((tile,mass,code))
    if at!=len(payload):raise ValueError('Trailing bytes in Gaussian packet')
    return meta,entries
