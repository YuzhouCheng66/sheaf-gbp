"""Whole-map learned spatial Gaussian source autoencoder, version14 wire.

ONE8x8 lattice of64 learned means represents the ENTIRE variable-size map.
It is not64 independent input sites or a raw occupancy carrier. The dense
convolutional decoder sees only q/h and public H/W: no private skip, own-map
overwrite, geometry oracle or peer map. Source fidelity, Gaussian composition
and eventual rate advantages are separate hypotheses; this implements SOURCE
ONLY. Scalarq is chosen isotropic confidence, not per-cell calibrated variance.
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from occupancy_codec.gaussian import HEADER
from occupancy_codec.isotropic_patch_wire import _integer, _numpy_codes
from occupancy_codec.site_gaussian_wire import SiteGaussianQuantizer


MODEL_KIND = 'global-spatial-source-v84'
VERSION = 14
CHANNELS = 64
PLAIN_BYTES = HEADER.size+1+1+CHANNELS  # 16B header, version, q,64means


@dataclass
class GlobalSpatialSourceConfig:
    side: int = 8
    channels: int = CHANNELS
    bits: int = 8
    width: int = 128
    seed: int = 108401


def _configuration(config):
    if (config.side,config.channels,config.bits,config.width)!=(8,64,8,128):
        raise ValueError('Registered global8x8/C64/b8/width128 source family required')


def _geometry(height,width):
    height,width=_integer(height,'height',128),_integer(width,'width',128)
    if min(height,width)<8:raise ValueError('Whole-map public dimensions8..128 required')
    return height,width


def _public(batch,height,width,device,dtype,full_height,full_width):
    """Only public coordinate chart and grid resolution, never observed facts."""
    yy,xx=torch.meshgrid(torch.linspace(-1,1,height,device=device,dtype=dtype),
                        torch.linspace(-1,1,width,device=device,dtype=dtype),indexing='ij')
    fields=torch.stack((yy,xx,torch.full_like(yy,1/full_height),torch.full_like(xx,1/full_width)),0)
    return fields[None].expand(batch,4,height,width)


class GlobalSpatialEncoder(nn.Module):
    def __init__(self,config):
        super().__init__();_configuration(config)
        incoming=7;layers=[]
        for outgoing,stride in ((48,1),(64,2),(96,2),(128,1),(128,1)):
            layers.extend((nn.Conv2d(incoming,outgoing,3,stride=stride,padding=1),nn.SiLU()))
            incoming=outgoing
        self.layers=nn.Sequential(*layers)
        self.pool=nn.AdaptiveAvgPool2d((8,8))
        self.mean=nn.Conv2d(128,1,1)
        self.precision=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,1))

    def forward(self,private,positions,active=None):
        if not isinstance(private,torch.Tensor) or private.ndim!=4:
            raise ValueError('Own private[B,N,H,W] required')
        b,n,height,width=private.shape;_geometry(height,width)
        if (private.dtype==torch.bool or private.is_complex() or not torch.isfinite(private).all()
                or not ((private==-1)|(private==0)|(private==1)).all()):
            raise ValueError('Own maps must be finite unknown-1/free0/wall1')
        if (not isinstance(positions,torch.Tensor) or positions.shape!=(b,n,2)
                or positions.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64)
                or positions.device!=private.device):
            raise ValueError('Own integer positions[B,N,2] on matching device required')
        if active is not None and (not isinstance(active,torch.Tensor) or active.shape!=(b,n)
                or active.dtype!=torch.bool or active.device!=private.device):
            raise ValueError('Public activity must be bool[B,N] on matching device')
        marker=torch.zeros(b,n,height*width,device=private.device,dtype=torch.float32)
        inside=((positions[...,0]>=0)&(positions[...,0]<height)&(positions[...,1]>=0)&(positions[...,1]<width))
        index=(positions.long()[...,0]*width+positions.long()[...,1]).clamp(0,height*width-1)
        marker.scatter_(-1,index[...,None],inside.float()[...,None])
        own=torch.stack(((private==0).float(),(private==1).float(),marker.reshape(b,n,height,width)),2)
        public=_public(b*n,height,width,private.device,own.dtype,height,width)
        features=torch.cat((own.flatten(0,1),public),1)
        hidden=self.pool(self.layers(features))
        mean=self.mean(hidden).flatten(1).tanh().reshape(b,n,64)
        nonempty=(private>=0).any((-1,-2))
        if active is not None:nonempty=nonempty&active
        q=(.05+3.95*self.precision(hidden.mean((-1,-2))).sigmoid()).reshape(b,n,1)*nonempty[...,None]
        return q,q*mean


class GlobalSpatialDecoder(nn.Module):
    def __init__(self,config):
        super().__init__();_configuration(config)
        self.lift=nn.Sequential(nn.Conv2d(6,128,3,padding=1),nn.SiLU(),
                                nn.Conv2d(128,128,3,padding=1),nn.SiLU())
        blocks=[];incoming=128
        for outgoing in (96,64,48,32):
            blocks.append(nn.Sequential(nn.Conv2d(incoming+4,outgoing,3,padding=1),nn.SiLU(),
                                        nn.Conv2d(outgoing,outgoing,3,padding=1),nn.SiLU()))
            incoming=outgoing
        self.blocks=nn.ModuleList(blocks)
        self.output=nn.Conv2d(32,2,1)

    def forward(self,q,h,height,width):
        height,width=_geometry(height,width)
        if (not isinstance(q,torch.Tensor) or not isinstance(h,torch.Tensor) or q.ndim!=3
                or q.shape[-1]!=1 or h.shape!=(*q.shape[:-1],64)):
            raise ValueError('Whole-map scalarq[B,N,1] and naturalh[B,N,64] required')
        if (q.device!=h.device or q.dtype!=h.dtype or not q.is_floating_point()
                or not torch.isfinite(q).all() or not torch.isfinite(h).all()
                or (q<0).any() or ((q==0).expand_as(h)&(h!=0)).any()):
            raise ValueError('Finite consistent Gaussian q/h on matching dtype/device required')
        b,n=q.shape[:2];batch=b*n
        mean=(h/q.clamp_min(1e-12)).reshape(batch,1,8,8)
        precision=q.log1p().reshape(batch,1,1,1).expand(batch,1,8,8)
        hidden=self.lift(torch.cat((mean,precision,_public(batch,8,8,q.device,q.dtype,height,width)),1))
        for size,block in zip((16,32,64,128),self.blocks):
            sy,sx=min(size,height),min(size,width)
            hidden=F.interpolate(hidden,(sy,sx),mode='bilinear',align_corners=False)
            hidden=block(torch.cat((hidden,_public(batch,sy,sx,q.device,q.dtype,height,width)),1))
        known,wall=self.output(hidden).reshape(b,n,2,height,width).unbind(2)
        logprob=torch.stack((F.logsigmoid(-known),F.logsigmoid(known)+F.logsigmoid(-wall),
                             F.logsigmoid(known)+F.logsigmoid(wall)),-1)
        empty=torch.full_like(logprob,-80.);empty[...,0]=0.
        return torch.where((q[...,0]<=0)[...,None,None,None],empty,logprob)


class GlobalSpatialSource(nn.Module):
    def __init__(self,config=GlobalSpatialSourceConfig()):
        super().__init__();_configuration(config);self.config=config
        self.encoder=GlobalSpatialEncoder(config);self.decoder=GlobalSpatialDecoder(config)
        self.quantizer=SiteGaussianQuantizer()

    def source_views(self,private,positions,active=None):
        q,h=self.encoder(private,positions,active)
        _,_,wq,wh=self.quantizer.encode(q,h)
        height,width=private.shape[-2:]
        return self.decoder(q,h,height,width),self.decoder(wq,wh,height,width)

    def counts(self):
        encoder=sum(p.numel() for p in self.encoder.parameters())
        decoder=sum(p.numel() for p in self.decoder.parameters())
        return dict(encoder=encoder,decoder=decoder,total=encoder+decoder,
            global_mean_channels=64,global_scalar_precisions=1,global_q_mean_scalar_fields=65,
            bits_per_scalar=8,latent_mean_lattice=[8,8],one_code_for_entire_map=True,
            global_wire_version=VERSION,plain_nonempty_packet_bytes=PLAIN_BYTES,
            generic_frame_overhead_bytes=9,uncompressed_framed_nonempty_bytes=PLAIN_BYTES+9,
            source_only=True,gbp_fusion_trained=False,learned_restriction_claim=False,
            no_private_decoder_context=True,no_own_projection=True,no_raw_map_carrier=True,
            no_rate_or_losslessness_claim=True,model_parameters_not_radio_traffic=True)


def serialize_global_source(tick,sender,receiver,qcode,means,height,width,round_id=0):
    """Whole-map version14, exact82B nonempty; no site count/address/mass."""
    height,width=_geometry(height,width)
    tick=_integer(tick,'tick',(1<<32)-1)
    sender,receiver=_integer(sender,'sender',65535),_integer(receiver,'receiver',65535)
    round_id=_integer(round_id,'source round',0)
    precision=_numpy_codes(qcode,(1,),255,'precision code')
    mean=_numpy_codes(means,(64,),254,'mean codes')
    if precision[0]==0:return None
    return HEADER.pack(tick,sender,receiver,64,height,width,round_id)+bytes((VERSION,int(precision[0])))+mean.astype(np.uint8).tobytes()


def deserialize_global_source(packet):
    """Explicit global semantics; cannot parse version13 site0 as a map."""
    if not isinstance(packet,(bytes,bytearray,memoryview)) or len(packet)!=PLAIN_BYTES:
        raise ValueError('Whole-map global-source packet must be exactly82 bytes')
    meta=HEADER.unpack_from(packet);_geometry(meta[4],meta[5])
    if meta[3]!=64 or meta[6]!=0 or packet[HEADER.size]!=VERSION:
        raise ValueError('Whole-map channels/source-round/version mismatch')
    qcode=np.frombuffer(packet,dtype=np.uint8,count=1,offset=HEADER.size+1).astype(np.int64)
    means=np.frombuffer(packet,dtype=np.uint8,count=64,offset=HEADER.size+2).astype(np.int64)
    if qcode[0]==0 or (means==255).any():raise ValueError('Empty/reserved Gaussian code in global source packet')
    return meta,qcode,means


def save_global_spatial_source(model,step,registration,validation):
    return dict(model_kind=MODEL_KIND,stage='source',config=asdict(model.config),
        state_dict=model.state_dict(),step=step,registration=registration,validation=validation,
        global_wire_version=VERSION,latent_mean_lattice=[8,8],one_code_for_entire_map=True,
        gbp_fusion_trained=False,no_own_projection=True)


def load_global_spatial_source(path,device='cpu'):
    saved=torch.load(path,map_location=device,weights_only=False)
    if (saved.get('model_kind')!=MODEL_KIND or saved.get('stage')!='source'
            or saved.get('global_wire_version')!=VERSION or saved.get('latent_mean_lattice')!=[8,8]
            or saved.get('one_code_for_entire_map') is not True or saved.get('gbp_fusion_trained') is not False):
        raise ValueError('Explicit whole-map spatial-source family/layout required')
    model=GlobalSpatialSource(GlobalSpatialSourceConfig(**saved['config'])).to(device)
    model.load_state_dict(saved['state_dict']);model.eval()
    return model,saved
