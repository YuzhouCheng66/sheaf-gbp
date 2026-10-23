"""Independent weak-prior scalar radio; existing sealed codecs are untouched.

Profile2 has 255 logarithmic positive precision levels in [1e-8,1], plus
zero. 64 precision bytes +64 standardized information bytes +20 header bytes.
No clipping or precision floor. Finite scalar-cavity GBP is NOT exact block
Gaussian inference. All directed sweeps count, including identical messages.
"""
import hashlib
import math
import numpy as np
import torch
from scripts.mapping_adaptive_wire_v207 import AdaptiveCodec64
from scripts.mapping_evidence_gbp_v208 import gaussian_belief
from scripts.mapping_recurrent_gbp_v213 import _stack, belief


class ObservedCodec64(AdaptiveCodec64):
    RANGES = dict(AdaptiveCodec64.RANGES, **{})
    RANGES[2] = (1e-8, 1.)


def quantize(precision, eta):
    p, h = precision.detach().double(), eta.detach().double()
    active = p > 0
    if not torch.isfinite(p).all() or not torch.isfinite(h).all():
        raise FloatingPointError('Nonfinite radio proposal')
    if (p < 0).any() or ((active & ((p < 1e-8) | (p > 1.))) | ((~active) & (h != 0))).any():
        raise ValueError('Outside declared precision profile2; no floor/clipping')
    safe = torch.where(active, p, torch.ones_like(p))
    value = torch.where(active, h / safe.sqrt(), torch.zeros_like(h))
    maximum = value.abs().amax(-1, keepdim=True)
    exponent = torch.ceil(torch.log2(maximum.clamp_min(2.**-8))).clamp_min(-8)
    if (exponent > 8).any():
        raise ValueError('Outside declared block floating range')
    pc = torch.where(active, 1+torch.round(torch.log(safe/1e-8)/math.log(1e8)*254), torch.zeros_like(p)).long()
    vc = torch.round(value/torch.pow(2., exponent)*127)
    if ((pc < 0) | (pc > 255)).any() or (vc.abs() > 127).any():
        raise ValueError('Invalid actual wire code')
    table = torch.as_tensor(np.concatenate(([0.], np.geomspace(1e-8, 1., 255))), device=p.device, dtype=torch.float64)
    hard_p = table[pc]
    hard_eta = vc/127*torch.pow(2., exponent)*hard_p.sqrt()
    # Receiver numerical storage is FP32, then promoted for FP64 Cholesky.
    # This matches NumPy packet decoding cast to FP32; not extra wire bits.
    hard_p = hard_p.float().to(precision.dtype)
    hard_eta = hard_eta.float().to(eta.dtype)
    return (hard_p+(precision-precision.detach()), hard_eta+(eta-eta.detach()), exponent.squeeze(-1).long())


def advance(state, rounds=1, *, damping=.5, quantized=True, trace=False):
    if rounds < 0 or not 0 < damping <= 1:
        raise ValueError('Rounds/damping')
    q, h = state['q'], state['h']
    p, eta = state['p'], state['eta']
    packets = []
    for step in range(rounds):
        mean, covariance, a, b = gaussian_belief(q, h, _stack(state, p), _stack(state, eta))
        diagonal = torch.diagonal(covariance, dim1=-2, dim2=-1)
        alpha, beta = diagonal.reciprocal()-a, mean/diagonal-b
        next_p, next_eta = {}, {}
        for i, j, w in state['directed']:
            cp = alpha[:,i]+a[:,i]-p[(j,i)]
            ch = beta[:,i]+b[:,i]-eta[(j,i)]
            # No inherited clamp hiding a negative numerical cavity.
            if (cp < 0).any() or not torch.isfinite(cp).all() or not torch.isfinite(ch).all():
                raise FloatingPointError('Negative/nonfinite cavity')
            raw_p = damping*w*cp/(w+cp)+(1-damping)*p[(i,j)]
            raw_eta = damping*w*ch/(w+cp)+(1-damping)*eta[(i,j)]
            op, oh, exponent = quantize(raw_p, raw_eta) if quantized else (raw_p, raw_eta, None)
            next_p[(i,j)], next_eta[(i,j)] = op, oh
            if trace:
                packets.append(dict(sender=i, receiver=j, round=state['iteration']+step,
                    raw_p=raw_p.detach(), raw_eta=raw_eta.detach(), decoded_p=op.detach(), decoded_eta=oh.detach(), exponent=exponent))
        p, eta = next_p, next_eta
    end = state['iteration']+rounds
    result = dict(state, p=p, eta=eta, iteration=end,
        message_versions={key:end-1 if rounds else version for key,version in state['message_versions'].items()})
    mean, covariance = belief(result)
    return dict(state=result, mean=mean, covariance=covariance, trace=packets,
        application_byte_budget_per_world=148*len(state['directed'])*rounds,
        negative_cavity_count=0)


def audit_packets(trace):
    packets = []
    for item in trace:
        if item['raw_p'].shape[0] != 1:
            raise ValueError('Audit one actual world at a time')
        packet,_ = ObservedCodec64.encode(item['raw_p'][0].cpu().numpy(), item['raw_eta'][0].cpu().numpy(),
            item['sender'], item['receiver'], item['round'], mode='standardized', precision_profile=2)
        decoded = ObservedCodec64.decode(packet)
        for name, key in (('precision','decoded_p'),('eta','decoded_eta')):
            expected = decoded[name].astype(np.float32).astype(item[key][0].cpu().numpy().dtype)
            np.testing.assert_array_equal(expected,item[key][0].cpu().numpy())
        packets.append(packet)
    return dict(actual_serialized_application_bytes=sum(map(len,packets)), directed_packets=len(packets),
        header_bytes=20*len(packets), precision_profile=2, precision_range=[1e-8,1.],
        receiver_storage='FP32 promoted to FP64 solve', SHA256=hashlib.sha256(b''.join(packets)).hexdigest(),
        packets=packets, physical_network_headers_measured=False)


def redeliver(state, packet):
    """Replacement-cache ingestion. Caller counts stale/duplicate bytes too."""
    if state['h'].shape[0] != 1:
        raise ValueError('One world')
    decoded = ObservedCodec64.decode(packet)
    key = (decoded['sender'], decoded['receiver'])
    if key not in state['p'] or decoded['mode']!='standardized' or decoded['precision_profile']!=2:
        raise ValueError('Wrong edge/profile')
    version, old = decoded['sequence'], state['message_versions'][key]
    if version < old:
        return state,False
    p = torch.as_tensor(decoded['precision'].astype(np.float32)[None],device=state['q'].device,dtype=state['q'].dtype)
    eta = torch.as_tensor(decoded['eta'].astype(np.float32)[None],device=state['q'].device,dtype=state['q'].dtype)
    if version == old:
        if not torch.equal(p,state['p'][key]) or not torch.equal(eta,state['eta'][key]):
            raise ValueError('Conflicting same message sequence')
        return state,False
    ps,hs,vs = dict(state['p']),dict(state['eta']),dict(state['message_versions'])
    ps[key],hs[key],vs[key] = p,eta,version
    return dict(state,p=ps,eta=hs,message_versions=vs,iteration=max(state['iteration'],version+1)),True
