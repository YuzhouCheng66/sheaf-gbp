"""Exact first-order full-native objective with stage-streamed readout backward.

Only scheduling differs from monolithic_objective: each decoder graph is freed
before decoding the next stage, while the original source/GBP graph stays live.
No optimizer, grad clearing, pixel sampling, AMP, private decoder input or
feature approximation. Existing parameter grads are accumulated, not replaced.
"""
import torch
from torch.nn import functional as F

STAGES=(0,1,2,4,8)
STAGE_WEIGHT=.2
PARITY_TOLERANCE={'rtol':2e-5,'atol':2e-7}


def proper_terms(heads,target):
    """All-pixel known BCE + observed-count wall BCE, then mean recipients."""
    if (not isinstance(target,torch.Tensor) or target.ndim!=4 or target.dtype!=torch.int8
            or min(target.shape[:2])<1 or target.shape[-1] not in (128,256,512)
            or target.shape[-2]!=target.shape[-1]
            or not bool(((target==-1)|(target==0)|(target==1)).all())):
        raise ValueError('Native int8 ternary target[B,N,H,H] required')
    b,n,h,w=target.shape
    if not isinstance(heads,dict) or set(heads)!={'known_logits','wall_logits'}:
        raise ValueError('Exactly original known/wall decoder heads required')
    for value in heads.values():
        if (not isinstance(value,torch.Tensor) or value.shape!=(b,n,h*w) or value.dtype!=torch.float32
                or value.device!=target.device or not bool(torch.isfinite(value).all())):
            raise ValueError('Finite same-device native FP32 heads[B,N,H*H] required')
    flat=target.reshape(b,n,h*w);known=flat>=0
    known_bce=F.binary_cross_entropy_with_logits(heads['known_logits'],known.float(),reduction='none')
    wall_bce=F.binary_cross_entropy_with_logits(heads['wall_logits'],(flat==1).float(),reduction='none')
    counts=known.sum(-1)
    known_per=known_bce.mean(-1)
    wall_per=torch.where(known,wall_bce,torch.zeros_like(wall_bce)).sum(-1)/counts.clamp_min(1)
    a,b=known_per.mean(),wall_per.mean()
    return dict(total=a+b,known=a,wall=b,known_per_recipient=known_per,wall_per_recipient=wall_per,
        observed_counts=counts,wall_counts=(flat==1).sum(-1),free_counts=(flat==0).sum(-1))


def validate_stages(features,targets):
    if not torch.is_grad_enabled():raise ValueError('Backward executor requires gradient-enabled context')
    if (not isinstance(features,dict) or not isinstance(targets,dict) or set(features)!=set(STAGES)
            or set(targets)!=set(STAGES) or any(type(k) is not int for k in (*features,*targets))):
        raise ValueError('Exactly stage keys0/1/2/4/8 required')
    reference=features[0];shape=None;target_shape=None
    for k in STAGES:
        f,t=features[k],targets[k]
        if (not isinstance(f,torch.Tensor) or f.ndim!=3 or f.shape[-1]!=128 or min(f.shape[:2])<1
                or f.dtype!=torch.float32 or not f.requires_grad or not bool(torch.isfinite(f).all())):
            raise ValueError('Live finite FP32 feature[B,N,128] required at every stage')
        if (not isinstance(t,torch.Tensor) or t.ndim!=4 or t.dtype!=torch.int8 or t.shape[:2]!=f.shape[:2]
                or t.shape[-1] not in (128,256,512) or t.shape[-2]!=t.shape[-1]
                or t.device!=f.device or not bool(((t==-1)|(t==0)|(t==1)).all())):
            raise ValueError('Matching full-native int8 targets required')
        if shape is None:shape=f.shape;target_shape=t.shape
        if f.shape!=shape or t.shape!=target_shape or f.device!=reference.device:
            raise ValueError('All stages must share scene shape, resolution and device')


def loss_record(terms):
    return dict(total=float(terms['total'].detach()),known=float(terms['known'].detach()),wall=float(terms['wall'].detach()),
        observed_counts=terms['observed_counts'].detach().cpu().tolist(),wall_counts=terms['wall_counts'].detach().cpu().tolist(),
        free_counts=terms['free_counts'].detach().cpu().tolist())


def report(records):
    return dict(total=sum(STAGE_WEIGHT*records[k]['total'] for k in STAGES),stages=records,
        stage_weights={k:STAGE_WEIGHT for k in STAGES},full_native_pixels=True,feature_sampling=False,
        optimizer_step_performed=False,gradients_cleared=False,AMP_used=False)


def monolithic_objective(model,stage_features,stage_targets):
    """Reference returns (live scalar loss, detached scalar/count report)."""
    validate_stages(stage_features,stage_targets);total=None;records={}
    for k in STAGES:
        f,t=stage_features[k],stage_targets[k]
        with torch.autocast(device_type=f.device.type,enabled=False):
            terms=proper_terms(model.decode(f,resolution=t.shape[-1]),t)
            weighted=STAGE_WEIGHT*terms['total']
        total=weighted if total is None else total+weighted;records[k]=loss_record(terms)
    return total,report(records)


def backward_staged(model,stage_features,stage_targets):
    """Accumulate decoder grads stagewise, then E/source/GBP grads once.

    The detached leaf changes only where the chain rule is evaluated. Its exact
    first-order cotangent is sent to the corresponding original LIVE feature.
    The source graph may itself contain higher-order derivative operations.
    Caller owns zero_grad, clipping, optimizer step and failed-update recovery.
    """
    validate_stages(stage_features,stage_targets);cotangents=[];records={}
    for k in STAGES:
        original,target=stage_features[k],stage_targets[k]
        leaf=original.detach().requires_grad_(True)
        with torch.autocast(device_type=leaf.device.type,enabled=False):
            heads=model.decode(leaf,resolution=target.shape[-1])
            terms=proper_terms(heads,target);weighted=STAGE_WEIGHT*terms['total']
        records[k]=loss_record(terms)
        weighted.backward()
        if leaf.grad is None or not bool(torch.isfinite(leaf.grad).all()):
            raise FloatingPointError('Readout must yield a finite feature cotangent')
        cotangents.append(leaf.grad.detach())
        del weighted,terms,heads,leaf
    torch.autograd.backward([stage_features[k] for k in STAGES],cotangents)
    result=report(records)
    result['source_backward_calls']=1
    result['decoder_backward_calls']=len(STAGES)
    result['detached_feature_leaf_does_not_detach_source_gradients']=True
    return result
