"""Fixed 148-byte, block-floating scalar Gaussian messages.

No model changes: natural parameters (p,eta) decode into the same solver.
Modes encode mean=eta/p, information=eta, or standardized=eta/sqrt(p).
Header includes the per-packet power-of-two block exponent and mode. Precision
tables are public profiles: broad (1e-4,1e4) or the registered experiment's
theory-safe narrow (1e-3,1). No free scaling information or silent saturation.
The narrow profile is NOT claimed suitable for arbitrary priors/edge weights.
"""
import struct
import numpy as np
from scripts.mapping_scalar_gbp_v192 import ScalarFactorGBP,local_factor_messages,_clean_scalar_factor


class AdaptiveCodec64:
    HEADER=struct.Struct('!4sBBBBIII')
    MODES={'mean':0,'information':1,'standardized':2}
    RANGES={0:(1e-4,1e4),1:(1e-3,1.)}
    EXPONENT_MIN=-8;EXPONENT_MAX=8

    @classmethod
    def encode(cls,precision,eta,sender,receiver,sequence,*,mode='mean',precision_profile=1):
        precision=np.asarray(precision,dtype=np.float64);eta=np.asarray(eta,dtype=np.float64)
        if precision.shape!=(64,) or eta.shape!=(64,) or not np.isfinite(precision).all() or not np.isfinite(eta).all():raise ValueError('Finite64D natural parameters required')
        if mode not in cls.MODES or precision_profile not in cls.RANGES or (precision<0).any():raise ValueError('Invalid mode/precision')
        if any(not isinstance(x,(int,np.integer)) or not 0<=x<=0xffffffff for x in (sender,receiver,sequence)) or sender==receiver:raise ValueError('Invalid header')
        active=precision>0
        if (eta[~active]!=0).any():raise ValueError('Empty precision with information')
        low,high=cls.RANGES[precision_profile]
        if ((precision[active]<low)|(precision[active]>high)).any():raise ValueError('Precision outside declared codebook; no clipping')
        encoded=np.zeros(64)
        if mode=='mean':encoded[active]=eta[active]/precision[active]
        elif mode=='information':encoded[active]=eta[active]
        else:encoded[active]=eta[active]/np.sqrt(precision[active])
        maximum=float(np.abs(encoded).max())
        exponent=max(cls.EXPONENT_MIN,int(np.ceil(np.log2(maximum)))) if maximum else cls.EXPONENT_MIN
        if exponent>cls.EXPONENT_MAX:raise ValueError('Vector outside declared block-floating range; no clipping')
        scale=float(2.**exponent);codes=np.rint(encoded/scale*127).astype(np.int16)+127
        if ((codes<0)|(codes>254)).any():raise ValueError('No silent vector clipping')
        pc=np.zeros(64,dtype=np.uint8)
        pc[active]=(1+np.rint(np.log(precision[active]/low)/np.log(high/low)*254)).astype(np.uint8)
        flags=cls.MODES[mode]+4*precision_profile
        packet=cls.HEADER.pack(b'SG27',1,exponent+128,64,flags,int(sender),int(receiver),int(sequence))+pc.tobytes()+codes.astype(np.uint8).tobytes()
        return packet,dict(block_exponent=exponent,active_coordinates=int(active.sum()))

    @classmethod
    def decode(cls,packet):
        if not isinstance(packet,bytes) or len(packet)!=148:raise ValueError('Exactly148 bytes required')
        magic,version,exponent_code,dim,flags,sender,receiver,sequence=cls.HEADER.unpack_from(packet)
        mode_id=flags%4;profile=flags//4;exponent=exponent_code-128
        if magic!=b'SG27' or version!=1 or dim!=64 or sender==receiver or profile not in cls.RANGES or mode_id not in cls.MODES.values() or not cls.EXPONENT_MIN<=exponent<=cls.EXPONENT_MAX:raise ValueError('Invalid adaptive header')
        pc=np.frombuffer(packet,dtype=np.uint8,count=64,offset=20);vc=np.frombuffer(packet,dtype=np.uint8,count=64,offset=84)
        if (vc==255).any() or ((pc==0)&(vc!=127)).any():raise ValueError('Noncanonical code')
        low,high=cls.RANGES[profile];table=np.concatenate(([0.],np.geomspace(low,high,255)))
        precision=table[pc];value=(vc.astype(np.float64)-127)/127*(2.**exponent)
        if mode_id==0:eta=value*precision
        elif mode_id==1:eta=value
        else:eta=value*np.sqrt(precision)
        eta[pc==0]=0.
        return dict(sender=sender,receiver=receiver,sequence=sequence,precision=precision,eta=eta,
            block_exponent=exponent,mode=list(cls.MODES)[mode_id],precision_profile=profile)

    @staticmethod
    def signature(packet):
        # Equal code bytes with a DIFFERENT exponent are different messages.
        return packet[4:8]+packet[20:]


class AdaptiveWireGBP(ScalarFactorGBP):
    def __init__(self,q,h,edges,*,mode='mean',precision_profile=1,damping=.5,event_suppression=True,allow_clip=False,profile=None):
        super().__init__(q,h,edges,damping=damping)
        if self.d!=64 or mode not in AdaptiveCodec64.MODES or precision_profile not in AdaptiveCodec64.RANGES or allow_clip:raise ValueError('Declared64D no-clipping protocol required')
        self.mode=mode;self.precision_profile=precision_profile;self.event_suppression=event_suppression
        self.packets=[];self.delivery_log=[];self.last_sent_signature={};self.proposals=0;self.suppressed=0;self.exponents=[]

    def receive_packet(self,packet):
        d=AdaptiveCodec64.decode(packet);key=(d['sender'],d['receiver'])
        if key not in self.messages or d['mode']!=self.mode or d['precision_profile']!=self.precision_profile:raise ValueError('Wrong edge/mode/profile')
        previous=self.message_versions[key];self.packets.append(packet)
        try:accepted=super().receive_message(*key,d['precision'],d['eta'],d['sequence'])
        except ValueError:
            self.delivery_log.append(dict(sender=key[0],receiver=key[1],sequence=d['sequence'],bytes=148,result='rejected_conflict'));raise
        self.delivery_log.append(dict(sender=key[0],receiver=key[1],sequence=d['sequence'],bytes=148,result='accepted' if accepted else 'stale' if d['sequence']<previous else 'duplicate'))
        return accepted

    def step(self):
        proposed={}
        for i in range(self.n):
            a,b=self.incoming(i);alpha,beta=local_factor_messages(self.q[i],self.h[i],a,b)
            for j in self.neighbors[i]:
                reverse_p,reverse_eta=self.messages[(j,i)]
                cp,ch=_clean_scalar_factor(alpha+a-reverse_p,beta+b-reverse_eta)
                weight=self.edges[tuple(sorted((i,j)))];p=weight*cp/(weight+cp);eta=weight*ch/(weight+cp)
                old_p,old_eta=self.messages[(i,j)]
                p=self.damping*p+(1-self.damping)*old_p;eta=self.damping*eta+(1-self.damping)*old_eta
                packet,meta=AdaptiveCodec64.encode(p,eta,i,j,self.iteration,mode=self.mode,precision_profile=self.precision_profile)
                self.proposals+=1;self.exponents.append(meta['block_exponent'])
                signature=AdaptiveCodec64.signature(packet)
                if self.event_suppression and self.last_sent_signature.get((i,j))==signature:self.suppressed+=1;continue
                proposed[(i,j)]=packet
        delta=0.
        for key,packet in proposed.items():
            old_p,old_eta=self.messages[key];old_p=old_p.copy();old_eta=old_eta.copy()
            self.receive_packet(packet);self.last_sent_signature[key]=AdaptiveCodec64.signature(packet)
            now_p,now_eta=self.messages[key]
            delta=max(delta,float(np.abs(now_p-old_p).max()),float(np.abs(now_eta-old_eta).max()))
        self.iteration+=1;return delta

    def accounting(self):
        return dict(actual_serialized_application_bytes=sum(map(len,self.packets)),directed_delivery_attempts=len(self.packets),
            vector_payload_bytes=64*len(self.packets),precision_payload_bytes=64*len(self.packets),header_bytes=20*len(self.packets),
            proposed_messages=self.proposals,identical_payload_suppressed=self.suppressed,mode=self.mode,precision_profile=self.precision_profile,
            smallest_proposed_block_exponent=min(self.exponents) if self.exponents else None,
            largest_proposed_block_exponent=max(self.exponents) if self.exponents else None,
            scale_and_mode_metadata_in_charged_header=True,silent_clipping=False,physical_network_headers_measured=False,compression_ratio_claim=False)
