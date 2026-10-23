"""Same generic lossless wrapper for Gaussian and raw packets.

Keep the16-byte application header, then mode:uint8, original size:uint32,
body, CRC32:uint32. Adaptive choice: plain / zlib / 16-bit bitplane+zlib.
All9 new framing bytes are charged. No rounding, model or inference change.
"""
import struct
import zlib
import numpy as np
from occupancy_codec.gaussian import HEADER

FRAME=struct.Struct('<BI');CRC=struct.Struct('<I');MAX_BODY=4*1024*1024


def shuffle16(body,inverse=False):
    if len(body)%2:raise ValueError('16-bit shuffle requires an even body')
    bits=np.unpackbits(np.frombuffer(body,np.uint8),bitorder='little')
    bits=bits.reshape(16,-1).T.flatten() if inverse else bits.reshape(-1,16).T.flatten()
    return np.packbits(bits,bitorder='little').tobytes()


def frame_packet(packet):
    if len(packet)<HEADER.size:raise ValueError('Truncated application header')
    head,body=packet[:HEADER.size],packet[HEADER.size:]
    if len(body)>MAX_BODY:raise ValueError('Oversized body')
    choices=[(0,body),(1,zlib.compress(body,6))]
    if body and len(body)%2==0:choices.append((2,zlib.compress(shuffle16(body),6)))
    mode,encoded=min(choices,key=lambda value:len(value[1]))
    return head+FRAME.pack(mode,len(body))+encoded+CRC.pack(zlib.crc32(body))


def unframe_packet(packet):
    if len(packet)<HEADER.size+FRAME.size+CRC.size:raise ValueError('Truncated frame')
    mode,size=FRAME.unpack_from(packet,HEADER.size)
    if mode not in (0,1,2) or size>MAX_BODY:raise ValueError('Invalid mode / size')
    encoded=packet[HEADER.size+FRAME.size:-CRC.size]
    if mode:
        decoder=zlib.decompressobj()
        try:body=decoder.decompress(encoded,size+1)
        except zlib.error as error:raise ValueError('Malformed zlib frame') from error
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:raise ValueError('Invalid compressed boundary')
        if mode==2:body=shuffle16(body,True)
    else:body=encoded
    if len(body)!=size or zlib.crc32(body)!=CRC.unpack_from(packet,len(packet)-CRC.size)[0]:
        raise ValueError('Length / CRC mismatch')
    return packet[:HEADER.size]+body
