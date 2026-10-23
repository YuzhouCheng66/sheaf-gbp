"""CPU-only equivalence; does not establish actual CUDA execution admission."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import unittest

os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import torch
from torch import nn

from scripts.mapping_support_local_v306 import SupportLocal64
from scripts.mapping_support_training_v306 import proper_terms

spec=importlib.util.spec_from_file_location('support_pool306_test',Path(__file__).with_name('mapping_support_pool_v306.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def private(size,n):
    labels=torch.full((1,n,size,size),-1,dtype=torch.int8)
    labels[...,:size//2,:3*size//4]=0
    labels[...,size//4:size//2,size//3:size//3+2]=1
    return labels


class PoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_all_native_CNN_sides_forward_and_input_grad(self):
        generator=torch.Generator(device='cpu').manual_seed(306101)
        for side in (4,8,16):
            for dtype in (torch.float32,torch.float64):
                a=torch.randn((2,7,side,side),dtype=dtype,generator=generator,requires_grad=True)
                b=a.detach().clone().requires_grad_()
                ref=nn.AdaptiveAvgPool2d((4,4))(a);got=m.FixedNativePool4()(b)
                torch.testing.assert_close(ref,got,**m.PARITY_TOLERANCE)
                cotangent=torch.randn(ref.shape,dtype=dtype,generator=generator)
                ref.backward(cotangent);got.backward(cotangent)
                torch.testing.assert_close(a.grad,b.grad,**m.PARITY_TOLERANCE)

    def test_both_full_models_128N2_256N1_loss_and_all_parameter_grads(self):
        for arm in ('support_local','cross_address'):
            for size,n in ((128,2),(256,1)):
                original=SupportLocal64(arm=arm);adapted=copy.deepcopy(original);m.enable_native_pool(adapted)
                labels=private(size,n);a=original(labels);b=adapted(labels)
                for key in ('q','h','mu','learned_q'):
                    torch.testing.assert_close(a['source_factors'][key],b['source_factors'][key],**m.PARITY_TOLERANCE)
                for key in ('known_logits','wall_logits'):
                    torch.testing.assert_close(a['decoded'][key],b['decoded'][key],**m.PARITY_TOLERANCE)
                la=proper_terms(a['decoded'],labels)['total'];lb=proper_terms(b['decoded'],labels)['total']
                torch.testing.assert_close(la,lb,**m.PARITY_TOLERANCE);la.backward();lb.backward()
                for (name,p),(other,q) in zip(original.named_parameters(),adapted.named_parameters()):
                    self.assertEqual(name,other);self.assertIsNotNone(p.grad,name);self.assertIsNotNone(q.grad,name)
                    self.assertTrue(torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all(),name)
                    torch.testing.assert_close(p.grad,q.grad,**m.PARITY_TOLERANCE,msg=f'{arm}/{size}/{name}')

    def test_state_parameter_identity_metadata_rng_and_typed_restore_unchanged(self):
        model=SupportLocal64();state={k:v.clone() for k,v in model.state_dict().items()}
        meta=model.construction_metadata();digest=model.weights_hash();parameters=list(model.parameters())
        rng=torch.get_rng_state().clone();decl=m.enable_native_pool(model)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue(decl['replaced_this_call'])
        self.assertEqual(json.loads(json.dumps(decl))['parameter_changes'],0)
        self.assertFalse(decl['construction_metadata_modified']);self.assertTrue(decl['reapply_after_typed_restore'])
        self.assertEqual(model.weights_hash(),digest);self.assertEqual(model.construction_metadata(),meta)
        self.assertEqual(set(model.state_dict()),set(state))
        for name,value in model.state_dict().items():self.assertTrue(torch.equal(value,state[name]))
        self.assertTrue(all(p is q for p,q in zip(parameters,model.parameters())))
        self.assertFalse(m.enable_native_pool(model)['replaced_this_call'])
        restored=SupportLocal64.from_payload(model.to_payload())
        self.assertIsInstance(restored.encoder[6],nn.AdaptiveAvgPool2d)
        self.assertEqual(restored.weights_hash(),digest);self.assertEqual(restored.construction_metadata(),meta)
        m.enable_native_pool(restored);self.assertEqual(restored.weights_hash(),digest)

    def test_invalid_input_and_wrong_pool_rejected_without_extra_mutation(self):
        pool=m.FixedNativePool4()
        for shape in ((1,1,5,5),(1,1,12,12),(1,1,8,4),(1,4,4),(0,1,4,4)):
            with self.assertRaises(ValueError):pool(torch.zeros(shape))
        with self.assertRaises(ValueError):pool(torch.zeros((1,1,4,4),dtype=torch.int64))
        with self.assertRaises(ValueError):m.enable_native_pool(nn.Sequential(nn.AdaptiveAvgPool2d((4,4))))
        model=SupportLocal64();model.encoder[6]=nn.AdaptiveAvgPool2d((2,2));old=model.encoder[6]
        with self.assertRaises(ValueError):m.enable_native_pool(model)
        self.assertIs(model.encoder[6],old)


if __name__=='__main__':unittest.main()
