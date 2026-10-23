"""Explicit support-local model adapter to the UNCHANGED v295 forest codec.

This does not remove the old CommonMap64 guard or modify any solver/codec.
FLOAT source/address locality is not wire locality: the original shared eta
exponent couples coordinates. New reached sources also bring their fixed
public-prior shares, even where their data evidence is exactly zero.
"""
import torch

from scripts import mapping_multiroom_forest_v295 as forest
from scripts.mapping_support_local_v306 import SupportLocal64, belief_features


PROFILE = forest.PROFILE
PACKET_BYTES = forest.PACKET_BYTES
DECODE_ROUNDS = forest.DECODE_ROUNDS
MAX_ROUNDS = forest.MAX_ROUNDS


def rollout(model, private, edges, *, rounds=8, decode_rounds=DECODE_ROUNDS,
            features_only=False, quantized=True, serialize=False,
            edge_contexts=None, trace=True):
    """One immutable source epoch; live features at every actual round.

    ``features_only=True`` skips D entirely, retaining the full autograd path
    through source/precision/messages/variance. Callers can decode recipient
    chunks with model.decode(..., resolution=result['native_resolution']).
    They must retain the shared feature graph until their last backward and
    normalize their loss identically; this function defines no loss or update.

    Serializing checks every outgoing value against the literal codec. The
    unchanged v295 audit performs independent receive-only replay, including
    duplicate/stale attempts. Setup, radio framing and tree discovery remain
    unmeasured; no loops, general deduplication or100x claim is introduced.
    """
    if (not isinstance(model,SupportLocal64) or type(rounds) is not int
            or not 0<=rounds<=MAX_ROUNDS):
        raise ValueError('Typed SupportLocal64 and registered0..16 rounds required')
    if any(type(value) is not bool for value in (features_only,quantized,serialize,trace)):
        raise ValueError('Explicit boolean execution flags required')
    if not isinstance(decode_rounds,(tuple,list)):
        raise ValueError('Explicit ordered decode round sequence required')
    requested=tuple(decode_rounds)
    if (len(set(requested))!=len(requested)
            or any(type(k) is not int or not 0<=k<=rounds for k in requested)
            or tuple(sorted(requested))!=requested):
        raise ValueError('Unique increasing decoded rounds inside the paid schedule required')
    if not isinstance(private,torch.Tensor) or private.ndim!=4:
        raise ValueError('Original private[B,N,H,H] required')
    if serialize and (not quantized or private.shape[0]!=1 or edge_contexts is None):
        raise ValueError('Literal frames require quantized B1 and established contexts')
    if not serialize and edge_contexts is not None:
        raise ValueError('Contexts apply only to literal serialization')

    factors=model.source_factors(private)
    state=forest.initialize(factors['q'],factors['h'],edges)
    if serialize:
        state=forest.bind_context(state,edge_contexts)
    means=[];variances=[];codes=[];features=[];states=[]
    history=[];packets=[];decoded={}
    for step in range(rounds+1):
        if step:
            update=forest.advance(state,1,quantized=quantized,trace=trace or serialize)
            state=update['state'];history.extend(update['trace'])
            if serialize:
                packets.extend(forest.serialize_trace(state,update['trace']))
        if state['q'] is not factors['q'] or state['h'] is not factors['h']:
            raise AssertionError('Original source factor was replaced or reinjected')
        mean,variance=forest.scalar_belief(state)
        feature=belief_features(mean,variance)
        means.append(mean);variances.append(variance);codes.append(feature[...,:64])
        features.append(feature);states.append(state)
        if not features_only and step in requested:
            decoded[step]=model.decode(feature,resolution=private.shape[-1])

    return dict(source_factors=factors,means=means,variances=variances,codes=codes,
        features=features,states=states,state=state,decoded=decoded,outputs=decoded,
        decoded_rounds=tuple(decoded),requested_decode_rounds=requested,
        features_only=features_only,features_detached=False,native_resolution=private.shape[-1],
        trace=history,packets=packets,rounds=rounds,quantized=quantized,
        mode='equality',damping=1.,precision_profile=PROFILE,
        adapter='mapping_support_forest_v306',model_arm=model.arm,
        source_builds_per_episode=1,source_versions_unchanged=True,
        additional_local_refinement_iterations=0,
        address_locality_scope='Model/FLOAT only at fixed N, reached sources and public-prior allocation',
        shared_exponent_wire_cross_address_coupling=True,
        wire=dict(packet_bytes=PACKET_BYTES,actual_data_bytes=sum(map(len,packets)),
            actual_packet_count=len(packets),
            proposed_directed_messages=len(state['directed'])*rounds*private.shape[0],
            protocol_budget_bytes=len(state['directed'])*rounds*private.shape[0]*PACKET_BYTES,
            precision_range=list(forest.ForestCodec64.RANGES[PROFILE]),
            offline_public_schedule=True,session_establishment_measured=False,
            schedule_setup_measured=False,physical_transport_measured=False,
            communication_100x_claim=False))


def audit_rollout(result, edges, contexts):
    """Delegate to the original independent CPU literal-receiver audit."""
    if not isinstance(result,dict) or result.get('adapter')!='mapping_support_forest_v306':
        raise ValueError('Explicit v306 adapter result required')
    return forest.audit_rollout(result,edges,contexts)
