"""Procedural partial observations and actual one-hop raw reference packets."""
import struct
import zlib
import numpy as np


def paired_private(sample,seed,mode='left_right'):
    if mode=='left_right':return sample['private'].copy()
    if mode!='rectangles':raise ValueError('Registered footprint mode required')
    walls=np.asarray(sample['walls']);size=walls.shape[0];rng=np.random.default_rng(seed)
    result=np.full((2,size,size),-1,dtype=np.int8)
    for agent in range(2):
        mask=np.zeros_like(walls,dtype=bool)
        for _ in range(2):
            x,y=rng.uniform(.0,.65,2);width,height=rng.uniform(.25,.6,2)
            x0,y0=int(x*size),int(y*size);x1,y1=min(size,int((x+width)*size)),min(size,int((y+height)*size))
            mask[y0:y1,x0:x1]=True
        # GT supplies only the declared exact sensor observation; never return
        # wall labels at masked cells or a shared peer footprint to a model.
        result[agent][mask]=walls[mask].astype(np.int8)
    return result


RAW_HEADER=struct.Struct('!4sBBBBIII')


def raw_packet(private,sender,receiver,*,compressed=False):
    private=np.asarray(private)
    if private.ndim!=2 or not np.isin(private,[-1,0,1]).all() or sender==receiver:raise ValueError('Invalid ternary map/endpoints')
    values=(private.reshape(-1)+1).astype(np.uint8)
    padded=np.pad(values,(0,(-len(values))%4)).reshape(-1,4)
    payload=(padded[:,0]|(padded[:,1]<<2)|(padded[:,2]<<4)|(padded[:,3]<<6)).tobytes()
    if compressed:payload=zlib.compress(payload,9)
    return RAW_HEADER.pack(b'RW28',1,int(compressed),2,0,int(sender),int(receiver),0)+payload


def decode_raw(packet,shape):
    magic,version,compression,bits,flags,sender,receiver,sequence=RAW_HEADER.unpack_from(packet)
    if (magic,version,bits,flags)!=(b'RW28',1,2,0) or compression not in (0,1) or sender==receiver:raise ValueError('Bad raw header')
    payload=zlib.decompress(packet[20:]) if compression else packet[20:]
    count=int(np.prod(shape))
    if len(payload)!=(count+3)//4:raise ValueError('Wrong payload length')
    packed=np.frombuffer(payload,dtype=np.uint8)
    values=np.stack([(packed>>(2*i))&3 for i in range(4)],1).reshape(-1)[:count]
    if (values==3).any():raise ValueError('Reserved raw symbol')
    return (values.astype(np.int8)-1).reshape(shape)


def raw_pair_accounting(private):
    if private.ndim!=3 or len(private)!=2:raise ValueError('Only two-agent direct exchange reference')
    counts={};union=private.max(axis=0)
    for compressed,name in [(False,'full_2bit_raw'),(True,'zlib_2bit_raw')]:
        packets=[raw_packet(private[i],i,1-i,compressed=compressed) for i in range(2)]
        for i in range(2):
            received=decode_raw(packets[1-i],private.shape[1:])
            if not np.array_equal(np.maximum(private[i],received),union):raise AssertionError('Raw exchange did not recover union')
        counts[name]=dict(application_bytes=sum(map(len,packets)),directed_packets=2,header_bytes=40,exact_union_verified=True)
    counts['public_setup']='Grid dimensions/frame predeclared for both raw and neural decoder; headers charged, physicalnetwork headers notmeasured.'
    return counts
