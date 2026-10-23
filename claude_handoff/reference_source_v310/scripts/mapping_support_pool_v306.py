"""Execution-only deterministic native pooling adapter; no model-source edits.

For encoder output sides4/8/16, adaptive4x4 bins are identical nonoverlapping
blocks of side1/2/4. Fixed avg_pool2d evaluates that same mean operator. This
does not claim CUDA parity before the separately registered device admission.
"""
import torch
from torch import nn
from torch.nn import functional as F

from scripts.mapping_support_local_v306 import SupportLocal64

INPUT_SIDES=(4,8,16)
PARITY_TOLERANCE={'rtol':2e-5,'atol':2e-7}


class FixedNativePool4(nn.Module):
    output_size=(4,4)

    def forward(self,value):
        if (not isinstance(value,torch.Tensor) or value.ndim!=4 or min(value.shape[:2])<1
                or value.shape[-1] not in INPUT_SIDES or value.shape[-2]!=value.shape[-1]
                or not value.is_floating_point()):
            raise ValueError('Native floating CNN output[B,C,S,S], S4/8/16 required')
        block=value.shape[-1]//4
        return F.avg_pool2d(value,kernel_size=block,stride=block,padding=0,ceil_mode=False,count_include_pad=False)


def enable_native_pool(model):
    """Replace ONLY the declared parameter-free encoder pool; idempotent.

    Typed payloads remain the unchanged model schema and restore the original
    constructor. Call this adapter again after restore; record this returned
    execution declaration separately from model construction metadata.
    """
    if not isinstance(model,SupportLocal64) or not isinstance(model.encoder,nn.Sequential):
        raise ValueError('Existing typed SupportLocal64 required')
    pools=[(i,module) for i,module in enumerate(model.encoder)
           if isinstance(module,(nn.AdaptiveAvgPool2d,FixedNativePool4))]
    if len(pools)!=1 or pools[0][0]!=6 or pools[0][1].output_size!=(4,4):
        raise ValueError('Exact original encoder.6 adaptive4x4 boundary required')
    index,old=pools[0]
    if tuple(old.parameters()) or tuple(old.buffers()):raise ValueError('Pool must have no parameter/buffer state')
    replaced=not isinstance(old,FixedNativePool4)
    if replaced:model.encoder[index]=FixedNativePool4().train(old.training)
    return dict(execution_adapter='mapping_support_pool_v306.FixedNativePool4',module_path='encoder.6',
        replaced_this_call=replaced,native_resolutions=[128,256,512],accepted_CNN_sides=list(INPUT_SIDES),
        kernel_equals_stride='input_side//4',output_size=[4,4],parameter_changes=0,buffer_changes=0,
        mathematical_operator='Same disjoint block means as adaptive4x4 at the registered divisible input sizes',
        construction_metadata_modified=False,constructor_silently_changed=False,reapply_after_typed_restore=True,
        CUDA_parity_requires_separate_native_admission=True)
