"""64-byte bottleneck of frozen spatial-codec features; no map skip.

PCA statistics are fitted exclusively on TRAIN and thereafter shared model
weights, not a per-map side channel. The nonlinear residual starts at the
PCA function. This is a source-capacity experiment, not distributed fusion.
"""
import torch
from torch import nn
from scripts.mapping_implicit_model_v192 import latent_codes, codes_to_latent, latent_quantize, validate_codes


def pack_features(q,h):
    mean=h/torch.where(q>0,q,torch.ones_like(q))
    return torch.cat((mean,q.log1p()),-1)


def unpack_features(value):
    mean,logq=value.split(value.shape[-1]//2,-1)
    mean=mean.clamp(-1,1);q=logq.clamp(0,torch.log(value.new_tensor(5.))).expm1()
    return q,q*mean


@torch.no_grad()
def fit_pca(features,ids,latent_dim=64):
    if len(ids)!=len(features) or not all(str(v).lower().startswith('train') for v in ids):
        raise ValueError('TRAIN-only PCA/normalization required')
    if features.ndim!=2 or len(features)<2 or not 0<latent_dim<=features.shape[1]:raise ValueError('Shape')
    center=features.mean(0);scale=features.std(0,unbiased=False).clamp_min(.05)
    normalized=(features-center)/scale
    covariance=normalized.T@normalized/len(features)
    eigenvalues,eigenvectors=torch.linalg.eigh(covariance.double())
    basis=eigenvectors[:,-latent_dim:].flip(-1).to(features.dtype)
    scores=normalized@basis
    limits=scores.abs().amax(0).clamp_min(.01)*1.05
    return dict(center=center,scale=scale,basis=basis,limits=limits),dict(
        explained_variance=float(eigenvalues[-latent_dim:].sum()/eigenvalues.clamp_min(0).sum()),
        dimensions=latent_dim,train_views=len(features),scale_floor=.05,code_bound_margin=1.05)


class SpatialBottleneck(nn.Module):
    def __init__(self,stats):
        super().__init__()
        for key,value in stats.items():self.register_buffer(key,value.detach().clone())
        dimension=self.center.numel();code=self.limits.numel()
        self.encoder_residual=nn.Sequential(nn.Linear(dimension,256),nn.SiLU(),nn.Linear(256,code))
        self.decoder_residual=nn.Sequential(nn.Linear(code,256),nn.SiLU(),nn.Linear(256,dimension))
        for layer in (self.encoder_residual[-1],self.decoder_residual[-1]):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def normalize(self,value):return (value-self.center)/self.scale

    def encode(self,value):
        features=self.normalize(value)
        return (features@self.basis/self.limits + self.encoder_residual(features)).clamp(-1,1)

    def decode_normalized(self,z):
        return (z*self.limits)@self.basis.T+self.decoder_residual(z)

    def decode(self,z):return self.decode_normalized(z)*self.scale+self.center

    def forward(self,value):return self.decode(latent_quantize(self.encode(value)))

    def to_codes(self,value):return latent_codes(self.encode(value))

    def from_codes(self,codes):
        validate_codes(codes)
        if codes.shape[-1]!=self.limits.numel():raise ValueError('Wrong payload size')
        return self.decode(codes_to_latent(codes))

