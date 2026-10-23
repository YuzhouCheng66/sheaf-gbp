"""A TOTAL64 mean/64 diagonal support-local Gaussian recognition family.

The arms differ only in permission for off-address connections. Both keep a
source's empty-address data q/h exactly zero. Locality is a model/FLOAT
property at FIXED N, reached-source set and prior allocation, not a claim
about the existing wire's shared eta exponent or changing public priors.
No transport, loss, teacher, optimization or sensor generation is implemented.
"""
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from scripts.mapping_implicit_model_v192 import latent_quantize, latent_codes, codes_to_latent


MODEL_KIND = 'support_local64_v306'
ARMS = ('support_local', 'cross_address')
RESOLUTIONS = (128, 256, 512)
GRID, ADDRESS_COUNT, PER_ADDRESS, LATENT_DIM = 4, 16, 4, 64
LATENT_PRIOR, Q_MIN, Q_MAX = 1e-4, 1e-4, 4.
DEFAULT_SEED = 306101
FOURIER_FREQUENCIES = (1., 2., 4., 8.)
QUERY_CHUNK = 1024


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)]).encode()
        digest.update(len(header).to_bytes(8, 'little')); digest.update(header)
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _resolution(resolution):
    if type(resolution) is not int or resolution not in RESOLUTIONS:
        raise ValueError('Explicit native square resolution128/256/512 required')
    return resolution


def patches(value):
    """[B,N,H,H] -> [B,N,16,H/4,H/4], row-major public addresses."""
    if not isinstance(value, torch.Tensor) or value.ndim != 4 or value.shape[-2] != value.shape[-1]:
        raise ValueError('Square batched node rasters required')
    size = _resolution(value.shape[-1]); b, n = value.shape[:2]; side = size//GRID
    if min(b,n) < 1:
        raise ValueError('Empty batch/node dimensions forbidden')
    return value.reshape(b,n,GRID,side,GRID,side).permute(0,1,2,4,3,5).reshape(b,n,16,side,side)


def assemble_patches(value):
    """Inverse spatial rearrangement; no interpolation or private bypass."""
    if (not isinstance(value,torch.Tensor) or value.ndim != 5 or value.shape[2] != 16
            or value.shape[-2] != value.shape[-1] or min(value.shape[:2]) < 1):
        raise ValueError('B,N,16,square patch values required')
    b,n,_,side,_ = value.shape; size = _resolution(side*GRID)
    return value.reshape(b,n,GRID,GRID,side,side).permute(0,1,2,4,3,5).reshape(b,n,size,size)


def _hard_mean(code):
    # Exact two documented FP32 realizations of integer/127, not allclose.
    if (not isinstance(code,torch.Tensor) or code.ndim < 2 or code.shape[-1] != 64
            or min(code.shape[:-1]) < 1 or code.dtype != torch.float32
            or not bool(torch.isfinite(code).all()) or bool((code.abs()>1).any())):
        raise ValueError('Finite hard8 FP32 mean64 required')
    integer = (code*127.).round()
    direct = (integer.double()/127.).float()
    reciprocal = integer*torch.tensor(1/127,dtype=torch.float32,device=code.device)
    if not bool(((code==direct)|(code==reciprocal)).all()):
        raise ValueError('Mean is not the original hard8 alphabet')


def belief_features(mean, variance):
    """Actual continuous mean/positive marginal variance -> live FP32128."""
    if (not isinstance(mean,torch.Tensor) or mean.ndim < 2 or mean.shape[-1] != 64
            or min(mean.shape[:-1]) < 1 or mean.dtype not in (torch.float32,torch.float64)
            or not isinstance(variance,torch.Tensor) or variance.shape != mean.shape
            or variance.device != mean.device or variance.dtype != mean.dtype
            or not bool(torch.isfinite(mean).all() and torch.isfinite(variance).all())
            or not bool((variance>0).all())):
        raise ValueError('Finite same-shape mean64 and positive variance64 required; no floor')
    result = torch.cat((latent_quantize(mean.float()),variance.log().float()),-1)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError('Nonfinite belief feature')
    return result


def public_queries(resolution, *, device):
    """[16,patch_pixels,20]: localxy, fixed local Fourier16, address-centrexy."""
    side = _resolution(resolution)//GRID
    axis = (torch.arange(side,device=device,dtype=torch.float32)+.5)*(2./side)-1.
    yy,xx = torch.meshgrid(axis,axis,indexing='ij')
    local = torch.stack((xx,yy),-1).reshape(-1,2)
    frequencies = torch.tensor(FOURIER_FREQUENCIES,device=device,dtype=torch.float32)
    angle = math.pi*local[:,:,None]*frequencies[None,None,:]
    coordinate = torch.cat((local,angle.sin().flatten(1),angle.cos().flatten(1)),-1)
    address_axis = (torch.arange(GRID,device=device,dtype=torch.float32)+.5)*(2./GRID)-1.
    ay,ax = torch.meshgrid(address_axis,address_axis,indexing='ij')
    address = torch.stack((ax,ay),-1).reshape(16,1,2).expand(-1,len(local),-1)
    return torch.cat((coordinate[None].expand(16,-1,-1),address),-1)


class AddressMixer(nn.Module):
    """Identity plus per-channel address mixing; no sample/agent mixing.

    Both store identical parameters. Masked local off-diagonals are ineffective
    (zero gradients), not extra trainable freedom. Metadata counts them apart.
    """
    def __init__(self, channels, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Registered arm required')
        self.channels, self.arm = channels, arm
        self.weight = nn.Parameter(torch.zeros(16,16,channels,dtype=torch.float32,device='cpu'))

    def forward(self, value):
        if value.shape[-2:] != (16,self.channels):
            raise ValueError('Address mixer boundary shape mismatch')
        weight = self.weight
        if self.arm == 'support_local':
            weight = weight*torch.eye(16,device=weight.device,dtype=weight.dtype)[:,:,None]
        return value+torch.einsum('...ic,oic->...oc',value,weight)

    def effective_parameters(self):
        return (16 if self.arm=='support_local' else 16*16)*self.channels


class SupportLocal64(nn.Module):
    """Standalone model. Original-map inputs are exclusively known/occupied.

    ``source_factors``/``factors`` accepts [B,N,H,H]; ``decode`` accepts ONLY
    hardmean64+actual logdiagC64. The wrapper does not call a Gaussian solver.
    Native images are convolved before feature aggregation; no raster is
    resized to128. Public addresses do not carry independent private bits.
    """
    latent_dim = 64

    def __init__(self, *, arm='support_local', seed=DEFAULT_SEED):
        super().__init__()
        if arm not in ARMS or type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('Registered arm and nonnegative CPU seed required')
        self.arm, self.seed = arm, seed
        # Restore caller CPU RNG; constructing this model does not touch CUDA RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            factory=dict(device='cpu',dtype=torch.float32)
            self.encoder = nn.Sequential(nn.Conv2d(2,32,3,2,1,**factory),nn.SiLU(),
                nn.Conv2d(32,64,3,2,1,**factory),nn.SiLU(),nn.Conv2d(64,64,3,2,1,**factory),nn.SiLU(),
                nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(1024,64,**factory),nn.SiLU())
            self.source_mixer = AddressMixer(64,arm)
            self.mean_head = nn.Linear(64,4,**factory)
            self.precision_head = nn.Linear(64,4,**factory)
            nn.init.zeros_(self.precision_head.weight)
            probability = (1.-Q_MIN)/(Q_MAX-Q_MIN)
            nn.init.constant_(self.precision_head.bias,math.log(probability/(1.-probability)))
            self.readout_mixer = AddressMixer(8,arm)
            self.decoder = nn.Sequential(nn.Linear(28,64,**factory),nn.SiLU(),
                nn.Linear(64,64,**factory),nn.SiLU(),nn.Linear(64,2,**factory))
        self.eval()

    def train(self, mode=True):
        if type(mode) is not bool:
            raise ValueError('Boolean train mode required')
        # No stochastic or batch-coupled modules; mode never changes arithmetic.
        return nn.Module.train(self,False)

    def encode(self, private):
        if (not isinstance(private,torch.Tensor) or private.ndim != 4
                or min(private.shape[:2])<1 or private.shape[1]>16
                or private.device!=self.mean_head.weight.device
                or private.dtype==torch.bool or private.is_complex()
                or private.shape[-2]!=private.shape[-1]
                or not bool(((private==-1)|(private==0)|(private==1)).all())):
            raise ValueError('Original ternary private[B,N,H,H],1<=N<=16 required')
        _resolution(private.shape[-1]); b,n = private.shape[:2]
        patch = patches(private); side = patch.shape[-1]
        channels = torch.stack((patch>=0,patch==1),3).float().reshape(b*n*16,2,side,side)
        with torch.autocast(device_type=private.device.type,enabled=False):
            feature = self.encoder(channels).reshape(b,n,16,64)
            feature = self.source_mixer(feature)
            continuous = self.mean_head(feature).tanh()
            mu = latent_quantize(continuous.reshape(b,n,64))
            q = Q_MIN+(Q_MAX-Q_MIN)*self.precision_head(feature).sigmoid()
        active = (patch>=0).flatten(-2).any(-1)
        if not bool(torch.isfinite(mu).all() and torch.isfinite(q).all()):
            raise FloatingPointError('Nonfinite recognition output; no repair')
        return dict(mu=mu,learned_q=q.reshape(b,n,64),active_per_address=active,
            code_bytes=latent_codes(mu),native_resolution=private.shape[-1])

    def source_factors(self, private):
        encoded = self.encode(private); n=private.shape[1]
        mask = encoded['active_per_address'][...,None].expand(-1,-1,-1,4).reshape_as(encoded['mu'])
        data_q = torch.where(mask,encoded['learned_q'].double(),0.)
        prior = LATENT_PRIOR/(n*64)
        precision = data_q+prior
        h = data_q*encoded['mu'].double()
        return dict(encoded,q=torch.diag_embed(precision),h=h,data_q=data_q,
            prior_precision=prior,active=encoded['active_per_address'].any(-1),
            source_factor_family='learned_diagonal_recognition_surrogate')

    factors = source_factors

    def decode(self, features, *, resolution=128):
        _resolution(resolution)
        if (not isinstance(features,torch.Tensor) or features.ndim not in (2,3)
                or features.shape[-1]!=128 or min(features.shape[:-1])<1
                or features.dtype!=torch.float32 or features.device!=self.mean_head.weight.device
                or not bool(torch.isfinite(features).all())):
            raise ValueError('Only finite FP32 hardmean64+logdiagC64 accepted')
        _hard_mean(features[...,:64])
        shape=features.shape[:-1]; batch=math.prod(shape); side=resolution//GRID
        with torch.autocast(device_type=features.device.type,enabled=False):
            tile = torch.cat((features[...,:64].reshape(batch,16,4),
                features[...,64:].reshape(batch,16,4)),-1)
            tile = self.readout_mixer(tile)
            coordinates = public_queries(resolution,device=features.device)
            parts=[]
            for start in range(0,side*side,QUERY_CHUNK):
                coord=coordinates[:,start:start+QUERY_CHUNK]; count=coord.shape[1]
                query=torch.cat((tile[:,:,None,:].expand(-1,-1,count,-1),
                    coord[None].expand(batch,-1,-1,-1)),-1)
                parts.append(self.decoder(query))
            logits=torch.cat(parts,2).reshape(batch,16,side,side,2)
            heads={name:assemble_patches(logits[...,index][:,None]).reshape(*shape,resolution*resolution)
                for index,name in enumerate(('known_logits','wall_logits'))}
        if any(not bool(torch.isfinite(value).all()) for value in heads.values()):
            raise FloatingPointError('Nonfinite decoder output')
        return heads

    def forward(self, private):
        factors=self.source_factors(private)
        precision=factors['q'].diagonal(dim1=-2,dim2=-1)
        features=belief_features(factors['h']/precision,precision.reciprocal())
        return dict(decoded=self.decode(features,resolution=private.shape[-1]),
            features=features,code=features[...,:64],source_factors=factors)

    def parameter_groups(self):
        return {name:list(getattr(self,name).parameters()) for name in
            ('encoder','source_mixer','mean_head','precision_head','readout_mixer','decoder')}

    def trainable_parameters(self):
        parameters=[p for group in self.parameter_groups().values() for p in group]
        if any(not p.requires_grad for p in parameters):
            raise AssertionError('This standalone family expects all declared parameters enabled')
        return parameters

    def weights_hash(self):
        return state_hash(self.state_dict())

    def construction_metadata(self):
        counts={name:sum(p.numel() for p in group) for name,group in self.parameter_groups().items()}
        effective=dict(counts,source_mixer=self.source_mixer.effective_parameters(),
            readout_mixer=self.readout_mixer.effective_parameters())
        return dict(model_kind=MODEL_KIND,arm=self.arm,seed=self.seed,grid=[4,4],
            dimensions_per_address=dict(mean=4,diagonal=4),total_dimensions=dict(mean=64,diagonal=64),
            native_resolutions=list(RESOLUTIONS),encoder_channels=['known','occupied'],
            encoder_widths=[32,64,64],encoder_feature_pool=[4,4],encoder_embedding=64,
            input_raster_resized=False,decoder_widths=[28,64,64,2],
            decoder_coordinates='local cell-centre xy; sin/cos(pi*localxy*frequencies); public address-centre xy',
            Fourier_frequencies=list(FOURIER_FREQUENCIES),coordinate_private_information=False,
            q_bounds=[Q_MIN,Q_MAX],latent_prior=LATENT_PRIOR,prior_per_source='lambda/(N*64)',
            empty_address_data_qh_exact_zero=True,decoder_private_mask_or_skip=False,
            parameter_counts=counts,effective_parameter_counts=effective,
            stored_parameter_count=sum(counts.values()),effective_parameter_count=sum(effective.values()),
            ineffective_masked_parameter_count=sum(counts.values())-sum(effective.values()),
            common_initial_function=True,identity_mixer_initialization=True,
            exact_observed_likelihood_claim=False,precision_calibrated=False,
            locality_scope='Fixed N/reached sources/prior; model FLOAT/hard8 per-coordinate only',
            wire_end_to_end_noninterference_claim=False,shared_eta_exponent_couples_wire_tiles=True,
            transport_implemented=False,proposed_unchanged_transport_profile=4,proposed_packet_bytes=172,
            no_support_stream=True,loopy_deduplication_claim=False,communication_100x_claim=False,
            old_parent_checkpoint_or_optimizer_loaded=False)

    def to_payload(self):
        return dict(construction=self.construction_metadata(),
            state_dict={key:value.detach().cpu().clone() for key,value in self.state_dict().items()},
            model_sha256=self.weights_hash())

    @classmethod
    def from_payload(cls,payload):
        if (not isinstance(payload,dict) or not isinstance(payload.get('construction'),dict)
                or not isinstance(payload.get('state_dict'),dict)):
            raise ValueError('Self-contained construction/state required')
        meta,state=payload['construction'],payload['state_dict']
        if meta.get('model_kind')!=MODEL_KIND:
            raise ValueError('Wrong model family')
        model=cls(arm=meta.get('arm'),seed=meta.get('seed'))
        expected=model.state_dict()
        if (meta!=model.construction_metadata() or set(state)!=set(expected)
                or any(not isinstance(value,torch.Tensor) or value.dtype!=expected[key].dtype
                       or value.shape!=expected[key].shape or not bool(torch.isfinite(value).all())
                       for key,value in state.items())):
            raise ValueError('Construction or state schema mismatch')
        model.load_state_dict(state,strict=True)
        if payload.get('model_sha256')!=model.weights_hash():
            raise ValueError('Typed state hash mismatch')
        return model
