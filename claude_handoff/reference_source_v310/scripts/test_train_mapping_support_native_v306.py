"""Bounded synthetic CPU trainer contracts; no bank preparation or GPU jobs."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
import numpy as np
import torch
from scripts import mapping_native_bank_v306 as bankmod

_spec=importlib.util.spec_from_file_location('support_native_trainer_test',Path(__file__).with_name('train_mapping_support_native_v306.py'))
m=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(m)
torch.set_num_threads(1)
HOPS=(0,1,2,4,8)
SIZES=(128,256,512)
ROOMS=(6,12,20)
COHORTS=('all','h_ge2','h_gt2','route_ge2','route_gt2','no_new_evidence')
CONFIG=Path(__file__).parents[1]/'configs'/'mapping_support_native_pair_v306.json'


def protocol():
    return json.loads(CONFIG.read_text(encoding='utf-8'))


def metric_values():
    return dict(own_wall_f1=.85,own_wall_precision=.88,own_wall_recall=.82,
        own_known_recall=.99,peer_wall_f1=.70,unknown_known_fpr=.01,
        pair_topology_accuracy=.90,wall_precision=.90,door_free_recall=.88)


def rows():
    result=[]
    for spec in m.scene_plan('VALIDATION'):
        metrics=metric_values()
        result.append(dict(spec=spec,identity=m.scene_key(spec),hops={str(k):{
            c:dict(metrics=copy.deepcopy(metrics),eligible={a:spec['agents'] for a in metrics},
                recipients=spec['agents']) for c in COHORTS} for k in HOPS}))
    return result


def endpoint(arm):
    data=rows()
    if arm=='support_local':
        for row in data:
            for cohort in row['hops']['8'].values():
                cohort['metrics'].update(own_wall_f1=.92,peer_wall_f1=.77)
    return dict(arm=arm,step=2160,split='VALIDATION',TEST_opened=False,
        actual_wire_audits_passed=True,records=data,profiles=m.aggregate(data))


def compare_tree(test,a,b):
    if isinstance(a,torch.Tensor):
        test.assertTrue(torch.equal(a,b))
    elif isinstance(a,dict):
        test.assertEqual(set(a),set(b))
        for key in a:compare_tree(test,a[key],b[key])
    elif isinstance(a,(tuple,list)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):compare_tree(test,x,y)
    else:test.assertEqual(a,b)


class SupportNativeTrainerTests(unittest.TestCase):
    def test_registered_identity_and_fairness_mutations_rejected(self):
        p=protocol()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'protocol.json'
            path.write_text(json.dumps(p),encoding='utf-8')
            self.assertEqual(m.registration(path),p)
            mutations=[
                lambda v:v.update(version=307),
                lambda v:v['data'].update(TEST_opened=True),
                lambda v:v['data'].update(draw_seed=306102),
                lambda v:v['data'].update(native_sizes=[128,256]),
                lambda v:v['training'].update(updates_per_arm=2161),
                lambda v:v['training'].update(stage_weights=[.5,.125,.125,.125,.125]),
                lambda v:v['training'].update(lr_start=.002),
                lambda v:v['training'].update(automatic_extension=True),
                lambda v:v['communication'].update(profile=3),
                lambda v:v['gate']['primary'].update(R8_macro_peer_wall_f1_gain_min=.01),
                lambda v:v['gate'].update(unknown_metric_or_incomplete_is_pass=True),
            ]
            for mutate in mutations:
                bad=copy.deepcopy(p);mutate(bad)
                path.write_text(json.dumps(bad),encoding='utf-8')
                with self.subTest(mutation=bad),self.assertRaises(ValueError):m.registration(path)

    def test_exact_fixed_draws_full_macrobalance_and_closed_validation(self):
        state=np.random.get_state();a=m.draw_plan();b=m.draw_plan()
        self.assertEqual(a,b);self.assertEqual(len(a),2160)
        self.assertEqual(a,bankmod.draw_plan(2160,seed=306101))
        self.assertEqual(len({bankmod.scene_key(s) for s in a}),1776)
        self.assertEqual(len({(s['rooms'],s['index']) for s in a}),296)
        for start in range(0,2160,18):
            block=a[start:start+18]
            self.assertEqual({(s['size'],s['rooms'],s['agents']) for s in block},
                {(h,r,n) for h in SIZES for r in ROOMS for n in (8,16)})
            self.assertTrue(all(s['split']=='TRAIN' and 0<=s['index']<256 for s in block))
            for r in ROOMS:self.assertEqual(len({s['index'] for s in block if s['rooms']==r}),1)
        after=np.random.get_state();np.testing.assert_array_equal(state[1],after[1]);self.assertEqual(state[2:],after[2:])
        val=m.scene_plan('VALIDATION');self.assertEqual(len(val),144)
        self.assertEqual(len({bankmod.scene_key(s) for s in val}),144)
        self.assertTrue(all(s['split']=='VALIDATION' and 0<=s['index']<8 for s in val))
        with self.assertRaises(ValueError):m.scene_plan('TEST')

    def test_cosine_endpoints_and_registered_elapsed_budgets(self):
        actual=[m.scheduled_lr(i) for i in (0,1,1079,1080,2159)]
        expected=[.0001+.5*(.001-.0001)*(1+math.cos(math.pi*i/2159)) for i in (0,1,1079,1080,2159)]
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-15)
        self.assertTrue(all(a>=b for a,b in zip(actual,actual[1:])))
        for step in (-1,2160,True,1.5):
            with self.assertRaises(ValueError):m.scheduled_lr(step)
        self.assertEqual(m.budget_check(100.,7200,now=7299.),7199.)
        for cap in (3600,7200):
            with self.assertRaises(TimeoutError):m.budget_check(100.,cap,now=100.+cap)
        with mock.patch.object(m,'STOP_REQUESTED',True):
            with self.assertRaises((InterruptedError,RuntimeError)):m.budget_check(0.,7200,now=1.)

    def test_aggregation_equal_worlds_equal_N_and_separate_native_cells(self):
        data=rows()
        for row in data:
            s=row['spec'];value=SIZES.index(s['size'])*.2+ROOMS.index(s['rooms'])*.04+(0. if s['agents']==8 else .1)+s['index']*.001
            for hop in row['hops'].values():
                for cohort in hop.values():cohort['metrics']['own_wall_f1']=value
        got=m.aggregate(data)
        self.assertEqual(set(got),{'128','256','512'})
        for h in SIZES:
            for r in ROOMS:
                expected=SIZES.index(h)*.2+ROOMS.index(r)*.04+.05+.0035
                for k in HOPS:
                    group=got[str(h)][str(k)][f'rooms{r}/allN']['all']
                    self.assertAlmostEqual(group['metrics']['own_wall_f1'],expected)
                    self.assertEqual(group['scenes'],16)
                    self.assertEqual(group['eligible']['own_wall_f1'],16)
                    for n in (8,16):
                        part=got[str(h)][str(k)][f'rooms{r}/N{n}']['all']
                        self.assertEqual(part['scenes'],8)
                        self.assertEqual(part['eligible_recipient_counts'],[n]*8)

    def test_complete_gate_direction_nine_equal_cells_and_own_retention(self):
        local,cross=endpoint('support_local'),endpoint('cross_address')
        got=m.paired_gate(local,cross,protocol())
        self.assertTrue(got['valid']);self.assertTrue(got['passed'])
        self.assertFalse(got['full_goal_complete']);self.assertFalse(got['whole_process_100x_claim'])
        by_name={r['name']:r for r in got['checks']}
        self.assertAlmostEqual(by_name['gain/R8/peerF1']['delta'],.07)
        self.assertAlmostEqual(by_name['gain/ownRetention']['delta'],.07)
        self.assertFalse(m.paired_gate(cross,local,protocol())['valid'])
        # One high-resolution cell must have exactly 1/9 macro weight, not pixel weight.
        for n in (8,16):
            local['profiles']['512']['8'][f'rooms20/N{n}']['all']['metrics']['peer_wall_f1']=.89
        got=m.paired_gate(local,cross,protocol())
        peer=next(c for c in got['checks'] if c['name']=='gain/R8/peerF1')
        self.assertAlmostEqual(peer['delta'],.07+.12/9)
        # Peer gains alone cannot pass if own-detail retention does not improve.
        for h in SIZES:
            for r in ROOMS:
                for n in (8,16):local['profiles'][str(h)]['8'][f'rooms{r}/N{n}']['all']['metrics']['own_wall_f1']=.85
        got=m.paired_gate(local,cross,protocol());self.assertFalse(got['passed'])
        self.assertFalse(next(c for c in got['checks'] if c['name']=='gain/ownRetention')['passed'])

    def test_gate_missing_metrics_denominators_identity_incomplete_fail_closed(self):
        cross=endpoint('cross_address')
        mutations=(
            lambda e:e.update(step=1080),lambda e:e.update(TEST_opened=True),
            lambda e:e.update(actual_wire_audits_passed=False),lambda e:e['records'].pop(),
            lambda e:e['records'].__setitem__(1,copy.deepcopy(e['records'][0])),
            lambda e:e['records'][0]['spec'].update(split='TEST'),
        )
        for mutate in mutations:
            e=endpoint('support_local');mutate(e);got=m.paired_gate(e,cross)
            self.assertFalse(got['valid']);self.assertFalse(got['passed'])
        for defect in ('none','nan','bool','missing','eligible','scenes','count_missing','zero_recipient','wrong_positive_recipient'):
            e=endpoint('support_local');cell=e['profiles']['256']['8']['rooms12/N8']['all']
            if defect=='none':cell['metrics']['peer_wall_f1']=None
            elif defect=='nan':cell['metrics']['peer_wall_f1']=float('nan')
            elif defect=='bool':cell['metrics']['peer_wall_f1']=True
            elif defect=='missing':del cell['metrics']['peer_wall_f1']
            elif defect=='eligible':cell['eligible']['peer_wall_f1']=7
            elif defect=='scenes':cell['scenes']=7
            elif defect=='count_missing':cell['eligible_recipient_counts'].pop()
            elif defect=='zero_recipient':cell['eligible_recipient_counts'][0]=0
            else:cell['eligible_recipient_counts'][0]=7
            with self.subTest(defect=defect):
                got=m.paired_gate(e,cross);self.assertFalse(got['valid']);self.assertFalse(got['passed'])
        # Finite global mean does not excuse a failed worst-H/rooms protection.
        e=endpoint('support_local')
        for n in (8,16):e['profiles']['256']['8'][f'rooms12/N{n}']['all']['metrics']['pair_topology_accuracy']=.849
        got=m.paired_gate(e,cross);self.assertTrue(got['valid']);self.assertFalse(got['passed'])
        self.assertTrue(any(not c['passed'] and c['name'].endswith('/pair_topology_accuracy') for c in got['checks']))

    def test_aggregation_cannot_replace_missing_world_with_duplicate(self):
        for defect in ('missing','duplicate','bad_identity','TEST'):
            data=rows()
            if defect=='missing':data.pop()
            elif defect=='duplicate':data[1]=copy.deepcopy(data[0])
            elif defect=='bad_identity':data[0]['identity']=data[1]['identity']
            else:data[0]['spec']['split']='TEST'
            with self.subTest(defect=defect),self.assertRaises(ValueError):m.aggregate(data)

    def test_fresh_only_preparation_and_run_guard_before_model_or_GPU(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p=protocol();p['output_root']=str(root.resolve())
            with mock.patch.object(m,'registration',return_value=p),mock.patch.object(m,'sha',side_effect=AssertionError('No external reads')):
                with self.assertRaises(FileExistsError):m.prepare(root,'unused.json')
            (root/'started.json').write_text('{}',encoding='utf-8')
            with mock.patch.object(m,'check',return_value=p),mock.patch.object(m,'admitted_bank',return_value=object()),\
                    mock.patch.object(m,'device_setup',side_effect=AssertionError('No GPU')):
                with self.assertRaises(FileExistsError):m.run(root,device='cpu')
            with mock.patch.object(m,'check',return_value=p),mock.patch.object(m,'helpers',return_value=(None,None,None,None)):
                with self.assertRaises(FileExistsError):m.prepare_data(root)

    def test_small_real_CPU_step_typed_pool_and_Adam_state_roundtrip(self):
        from scripts.mapping_support_pool_v306 import FixedNativePool4
        own=np.full((2,128,128),-1,np.int8)
        own[0,:,:70]=0;own[1,:,58:]=0;own[0,15:112,30]=1;own[1,15:112,90]=1
        scene=dict(private=own,forest={'edges':[(0,1,1.)]},targets={0:own.copy(),**{k:np.broadcast_to(own.max(0),own.shape).copy() for k in HOPS if k}})
        local=m.make_model('support_local');cross=m.make_model('cross_address')
        self.assertEqual(local.weights_hash(),cross.weights_hash())
        for arm,model in [('support_local',local),('cross_address',cross)]:
            opt=torch.optim.AdamW(model.trainable_parameters(),lr=.001,weight_decay=0)
            before=model.weights_hash();record=m.train_step(model,opt,scene,'cpu',0)
            json.dumps(record,allow_nan=False)
            self.assertNotEqual(before,model.weights_hash());self.assertTrue(all(v>0 and math.isfinite(v) for v in record['gradient_groups'].values()))
            self.assertEqual(set(record['stages']),set(HOPS));self.assertEqual(record['lr'],.001)
            digest=m.json_hash(m.draw_plan()[:1])
            with tempfile.TemporaryDirectory() as temp:
                saved=m.save_checkpoint(temp,model,opt,1,arm,protocol(),digest)
                restored,payload=m.load_checkpoint(saved['path'])
                self.assertIsInstance(restored.encoder[6],FixedNativePool4)
                self.assertEqual(restored.weights_hash(),model.weights_hash())
                self.assertEqual(restored.construction_metadata(),model.construction_metadata())
                self.assertEqual(payload['completed_updates'],1);self.assertEqual(payload['actual_draw_sha256'],digest)
                self.assertFalse(payload['recovery_is_resume']);self.assertEqual(payload['cuda_rng_state'],[])
                other=torch.optim.AdamW(restored.trainable_parameters(),lr=.001,weight_decay=0)
                other.load_state_dict(payload['optimizer_state']);compare_tree(self,opt.state_dict(),other.state_dict())
                m.train_step(model,opt,scene,'cpu',1);m.train_step(restored,other,scene,'cpu',1)
                self.assertEqual(restored.weights_hash(),model.weights_hash())
                compare_tree(self,opt.state_dict(),other.state_dict())
                with self.assertRaises(FileExistsError):m.save_checkpoint(temp,restored,other,1,arm,protocol(),digest)
                bad=copy.deepcopy(payload);bad['model_sha256']='0'*64;path=Path(temp)/'bad.pt';torch.save(bad,path)
                with self.assertRaises(ValueError):m.load_checkpoint(path)


if __name__=='__main__':unittest.main()
