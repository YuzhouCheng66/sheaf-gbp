"""CPU-only synthetic contracts; no checkpoint, GPU, training, or TEST data."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from scripts import train_mapping_multiroom_v295 as run


def endpoint(arm):
    metrics=dict(own_typed_accuracy=.96,own_wall_f1=.91,own_wall_precision=.92,own_wall_recall=.90,
        known_recall=.97,own_known_recall=.97,union_accuracy=.90,peer_wall_f1=.65,
        door_free_recall=.94,pair_topology_accuracy=.96,unknown_known_fpr=.01)
    if arm=='causal_joint':metrics.update(union_accuracy=.95,peer_wall_f1=.80)
    profiles={str(k):{f'rooms{r}/'+('allN' if n is None else f'N{n}'):
        dict(all=dict(metrics=copy.deepcopy(metrics)),h_ge2=dict(eligible_recipient_counts=[2]*8),h_gt2=dict(eligible_recipient_counts=[2]*8))
        for r in (6,12,20) for n in (None,8,16)} for k in run.HOPS}
    return dict(arm=arm,step=4000,split='VALIDATION',TEST_opened=False,actual_wire_audits_passed=True,
        profiles=profiles,records=[dict(identity=run.scene_id(spec),spec=spec) for spec in run.scene_plan('VALIDATION')])


class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_registration_and_no_unregistered_native_size(self):
        path=Path(run.__file__).resolve().parents[1]/'configs'/run.CONFIG
        p=run.registration(path)
        for field,value in (('native_size',512),('TRAIN_scene_tuples',1535),('room_counts',[6,12])):
            bad=copy.deepcopy(p);bad['data'][field]=value
            with tempfile.TemporaryDirectory() as folder:
                target=Path(folder)/'bad.json';target.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):run.registration(target)
        bad=copy.deepcopy(p);bad['execution']['gpu_uuid']='GPU-other'
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'bad.json';target.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):run.registration(target)

    def test_explicit_registered_device_rejects_wrong_physical_gpu(self):
        p=run.registration(Path(run.__file__).resolve().parents[1]/'configs'/run.CONFIG)
        with mock.patch.object(run,'check',return_value=p),mock.patch.dict(run.os.environ,{'CUDA_VISIBLE_DEVICES':run.GPU_UUID}),\
                mock.patch.object(run.subprocess,'check_output',return_value=run.GPU_UUID+', RTX 4080\n'):
            with self.assertRaises(RuntimeError):run.device_setup(Path('unused'),'cuda:0',run.GPU_UUID)

    def test_scene_identity_and_shared_deterministic_draws(self):
        train=run.scene_plan('TRAIN');val=run.scene_plan('VALIDATION')
        self.assertEqual(len(train),1536);self.assertEqual(len(val),48)
        self.assertEqual(len({(s['rooms'],s['index']) for s in train}),768)
        self.assertEqual(len({(s['rooms'],s['index']) for s in val}),24)
        self.assertTrue(np.array_equal(run.draw_plan(),run.draw_plan()))
        self.assertEqual(run.draw_plan().shape,(4000,));self.assertLess(int(run.draw_plan().max()),1536)
        with self.assertRaises(ValueError):run.scene_plan('TEST')
        with self.assertRaises(ValueError):run.scene_id(dict(split='TRAIN',index=0,rooms=6,agents=8,size=256))

    def test_cosine_schedule_no_hidden_extension(self):
        self.assertAlmostEqual(run.scheduled_lr(0),5e-5,15);self.assertAlmostEqual(run.scheduled_lr(3999),5e-6,15)
        values=[run.scheduled_lr(i) for i in range(4000)]
        self.assertTrue(all(a>=b for a,b in zip(values,values[1:])))
        for value in (-1,4000,True):
            with self.assertRaises(ValueError):run.scheduled_lr(value)

    def test_forest_targets_use_actual_routes_not_radius_shortcuts(self):
        private=np.full((4,3,3),-1,np.int8)
        for i in range(4):private[i,i//3,i%3]=i%2
        targets,reached=run.forest_labels(private,[(0,1,1.),(1,2,1.)])
        self.assertEqual(targets[1][0,0,2],-1);self.assertEqual(targets[2][0,0,2],0)
        self.assertFalse(reached[8][0,3]);self.assertTrue(np.array_equal(targets[8][3],private[3]))
        self.assertTrue(np.array_equal(targets[0],private))
        self.assertEqual(targets[8][0,1,0],-1)

    def test_loss_weights_recipient_mean_unknown_wall_gradient_zero(self):
        labels=torch.tensor([[[[0,-1],[1,-1]],[[0,0],[0,1]]]],dtype=torch.int8)
        outputs={};targets={}
        for k in run.HOPS:
            outputs[k]=dict(known_logits=torch.full((1,2,4),float(k)/10,requires_grad=True),
                            wall_logits=torch.full((1,2,4),float(k)/20,requires_grad=True))
            targets[k]=labels
        result=dict(decoded=outputs)
        own,_=run.objective(result,targets,'source_only');own.backward(retain_graph=True)
        self.assertIsNone(outputs[8]['wall_logits'].grad)
        self.assertEqual(float(outputs[0]['wall_logits'].grad[0,0,1]),0.)
        self.assertEqual(float(outputs[0]['wall_logits'].grad[0,0,3]),0.)
        expected=sum(run.stage_loss(outputs[k],labels)[0]*weight for k,weight in ((0,.5),(1,.125),(2,.125),(4,.125),(8,.125)))
        actual,_=run.objective(result,targets,'causal_joint');self.assertTrue(torch.equal(actual,expected))
        empty=torch.full_like(labels,-1);value,terms=run.stage_loss(outputs[0],empty)
        self.assertEqual(float(terms['wall'].detach()),0.);self.assertTrue(torch.isfinite(value))
        with self.assertRaises(ValueError):run.stage_loss(outputs[0],labels.expand(2,-1,-1,-1))
        with self.assertRaises(ValueError):run.objective(result,targets,'unknown')
        bad=dict(known_logits=torch.full((1,2,4),float('nan')),wall_logits=torch.zeros(1,2,4))
        with self.assertRaises(FloatingPointError):run.stage_loss(bad,labels)

    def test_source_only_computes_no_peer_rollout_or_extra_decoder(self):
        with mock.patch.object(run.core,'rollout',return_value={}) as fn:
            run.train_rollout(None,None,[],'source_only')
            self.assertEqual(fn.call_args.kwargs['rounds'],0);self.assertEqual(fn.call_args.kwargs['decode_rounds'],(0,))
            run.train_rollout(None,None,[],'causal_joint')
            self.assertEqual(fn.call_args.kwargs['rounds'],8);self.assertEqual(fn.call_args.kwargs['decode_rounds'],run.HOPS)

    def test_original_scene_cache_roundtrip_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            bank=run.SceneBank(folder);spec=dict(split='TRAIN',index=0,rooms=6,agents=8,size=128)
            first=bank.get(spec);second=bank.get(spec)
            self.assertTrue(np.array_equal(first['private'],second['private']))
            self.assertTrue(np.array_equal(first['targets'][0],first['private']))
            self.assertEqual(first['forest']['edges'],second['forest']['edges'])
            self.assertEqual(set(first['metadata']['arrays_sha256']),{'private','positions_m','adjacency','trajectory_m','door_mask'})
            path=Path(folder)/(run.scene_id(spec)+'.npz')
            path.write_bytes(path.read_bytes()+b'tamper')
            with self.assertRaises(ValueError):bank.get(spec)

    def test_empty_and_mixed_cohorts_preserve_eligible_scene_counts(self):
        self.assertEqual(run.mean_rows([])['recipients'],0)
        merged=run.mean_rows([{},dict(peer_wall_f1=.6),dict(peer_wall_f1=.8)])
        self.assertAlmostEqual(merged['metrics']['peer_wall_f1'],.7);self.assertEqual(merged['eligible']['peer_wall_f1'],2)
        records=[]
        for i,eligible in enumerate((False,True)):
            scopes={cohort:run.mean_rows([] if cohort!='all' and not eligible else [dict(union_accuracy=.8+i*.1)])
                    for cohort in run.COHORTS}
            records.append(dict(spec=dict(rooms=6,agents=8),hops={str(k):copy.deepcopy(scopes) for k in run.HOPS}))
        result=run.aggregate(records)['0']['rooms6/N8']
        self.assertEqual(result['h_ge2']['eligible_recipient_counts'],[0,1])
        self.assertEqual(result['h_ge2']['eligible']['union_accuracy'],1)
        self.assertAlmostEqual(result['h_ge2']['metrics']['union_accuracy'],.9)

    def test_physical_neighbor_routed_two_rounds_not_called_physical_hop2(self):
        own=np.full((3,8,8),-1,np.int8)
        for i in range(3):own[i,3,i+2]=i%2
        edges=[(0,1,1.),(1,2,1.)];targets,reached=run.forest_labels(own,edges)
        scene=dict(private=own,forest=dict(edges=edges),targets=targets,reached=reached,
            adjacency=~np.eye(3,dtype=bool),door_mask=np.zeros((8,8),bool))
        report=run.scene_metrics(scene,targets[2],2)
        self.assertEqual(report['nodes'][0]['h_ge2_sources'],0)
        self.assertEqual(report['nodes'][0]['h_ge2_novel_cells'],0)
        self.assertEqual(report['nodes'][0]['route_ge2_sources'],1)
        self.assertEqual(report['nodes'][0]['route_ge2_novel_cells'],1)
        self.assertEqual(report['h_ge2']['recipients'],0)
        self.assertEqual(report['route_ge2']['recipients'],2)

    def test_gate_material_fail_closed_and_protection(self):
        control=endpoint('source_only');candidate=endpoint('causal_joint')
        self.assertTrue(run.paired_gate(control,candidate)['passed'])
        for kind in ('missing','nan','erasewall','FPR','no_multihop','source_limited'):
            bad=copy.deepcopy(candidate)
            if kind=='missing':bad['records'].pop()
            elif kind=='nan':bad['profiles']['2']['rooms6/allN']['all']['metrics']['peer_wall_f1']=float('nan')
            elif kind=='erasewall':bad['profiles']['8']['rooms6/N16']['all']['metrics']['own_wall_recall']=.1
            elif kind=='FPR':bad['profiles']['8']['rooms12/N8']['all']['metrics']['unknown_known_fpr']=.05
            elif kind=='no_multihop':bad['profiles']['2']['rooms20/allN']['h_ge2']['eligible_recipient_counts']=[0]*8
            else:bad['profiles']['0']['rooms20/allN']['all']['metrics']['own_wall_f1']=.6
            self.assertFalse(run.paired_gate(control,bad)['passed'],kind)

    def test_atomic_json_and_npz_and_fixed_buffers(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'status.json';run.write(path,dict(state='one'));run.write(path,dict(state='two'))
            self.assertEqual(run.read(path)['state'],'two');self.assertFalse(path.with_name('status.json.tmp').exists())
            archive=Path(folder)/'a.npz';run.save_npz(archive,x=np.arange(4))
            with np.load(archive) as z:self.assertTrue(np.array_equal(z['x'],np.arange(4)))
        model=torch.nn.Module();model.register_buffer('fixed',torch.ones(2));snapshot=run.state_buffers(model)
        run.assert_buffers(model,snapshot);model.fixed[0]=2.
        with self.assertRaises(AssertionError):run.assert_buffers(model,snapshot)


if __name__=='__main__':unittest.main()
