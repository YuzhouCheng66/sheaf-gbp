"""Bounded synthetic CPU contracts; no actual bank, remote, GPU or training job."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np
import torch
from torch import nn

spec=importlib.util.spec_from_file_location('multiscale310_test_runner',Path(__file__).with_name('train_mapping_multiscale_v310.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
CONFIG=Path(__file__).parents[1]/'configs/mapping_v299_multiscale_continuation_v310.json'


def protocol():
    p=json.loads(CONFIG.read_text(encoding='utf-8'));p['status']='REGISTERED_SYNTHETIC_TEST'
    p['model']['implementation_metadata_pending']=False;return p


def values(value=.5):return {k:value for k in m.KEYS}


def synthetic_report():
    profiles={}
    fields=('own_wall_f1','own_typed_accuracy','union_accuracy','peer_wall_f1','door_free_recall','pair_topology_accuracy','unknown_known_fpr')
    for h in m.base.SIZES:
        profiles[str(h)]={str(k):{f'rooms{r}/N{n}':dict(all=dict(metrics={key:.5 for key in fields},scenes=8,
            eligible={key:8 for key in fields},eligible_recipient_counts=[n]*8)) for r in m.base.ROOMS for n in m.base.AGENTS} for k in m.HOPS}
    return dict(version=310,split='VALIDATION',TEST_opened=False,actual_wire_audits_passed=True,profiles=profiles,
        records=[dict(identity=m.base.scene_key(s),spec=s) for s in m.base.scene_plan('VALIDATION')])


def assert_nested_equal(test,a,b):
    if isinstance(a,torch.Tensor):test.assertTrue(torch.equal(a,b))
    elif isinstance(a,np.ndarray):test.assertTrue(np.array_equal(a,b))
    elif isinstance(a,dict):
        test.assertEqual(set(a),set(b))
        for k in a:assert_nested_equal(test,a[k],b[k])
    elif isinstance(a,(list,tuple)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):assert_nested_equal(test,x,y)
    else:test.assertEqual(a,b)


class ToyReadout(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(128,2)
    def decode(self,features,*,resolution):
        z=self.linear(features);offset=torch.linspace(-.2,.3,resolution*resolution,device=z.device)
        return {name:z[...,i,None]+offset for i,name in enumerate(('known_logits','wall_logits'))}


class MultiscaleTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_registration_rejects_mutations_and_draft(self):
        p=protocol()
        with mock.patch.object(m.base,'read',return_value=p):self.assertEqual(m.registration('fixture'),p)
        edits=(lambda q:q.update(status='DRAFT'),lambda q:q['model'].update(implementation_metadata_pending=True),
            lambda q:q['parent'].update(parent_optimizer_inherited=True),lambda q:q['parent'].update(checkpoint_sha256='0'*64),
            lambda q:q['training'].update(lr_phases=[.001,.0001,.00001]),lambda q:q['training'].update(stage_weights=[.2]*5),
            lambda q:q['training'].update(recipient_chunk_size=2),lambda q:q['data'].update(TEST_opened=True),
            lambda q:q['data'].update(new_data_generated=True),lambda q:q['data'].update(TRAIN_scenes=1536),
            lambda q:q['plateau'].update(minimum_epochs_before_first_lr_reduction=3),
            lambda q:q['plateau']['recovery_trend_guard'].update(lookback_epochs=[3]),
            lambda q:q['execution'].update(session_seconds_cap=86400),lambda q:q['execution'].update(preflight_TRAIN_index=0),
            lambda q:q['communication'].update(profile=3),lambda q:q['validation'].update(protected128_R8_unknown_fpr_max_rise=.1))
        for edit in edits:
            bad=copy.deepcopy(p);edit(bad)
            with mock.patch.object(m.base,'read',return_value=bad),self.assertRaises(ValueError):m.registration('fixture')

    def test_epoch_unique_balanced_prefix_determinism_and_resume_cursor(self):
        specs=m.base.scene_plan('TRAIN');manifest={'cases':{m.base.scene_key(s):{'spec':s} for s in specs}}
        admitted=m.training_specs(manifest);first=m.epoch_plan(admitted,1);second=m.epoch_plan(admitted,2)
        self.assertEqual(first,m.epoch_plan(admitted,1));self.assertNotEqual(first,second)
        self.assertEqual(len(first),1776);self.assertEqual({m.base.scene_key(s) for s in first},{m.base.scene_key(s) for s in specs})
        for h in m.base.SIZES:self.assertEqual(sum(s['size']==h for s in first),592)
        self.assertEqual(len({(s['size'],s['rooms'],s['agents']) for s in first[:18]}),18)
        state=m.initial_progress();m.validate_progress(state,admitted)
        state.update(epoch=1,cursor=100,updates=100,phase='train',epoch_draw_sha256=m.base.json_hash(first),tracker=m.update_tracker(m.initial_tracker(),values(),0))
        m.validate_progress(state,admitted);self.assertEqual(first[100:],m.epoch_plan(admitted,state['epoch'])[state['cursor']:])
        state['cursor']=101
        with self.assertRaises(ValueError):m.validate_progress(state,admitted)
        wrong=copy.deepcopy(manifest);wrong['cases'].pop(next(iter(wrong['cases'])))
        with self.assertRaises(ValueError):m.training_specs(wrong)

    def test_plateau_only_complete15_metrics_exact_schedule_and_cumulative_improvement(self):
        state=m.initial_tracker()
        for epoch in range(13):
            state=m.update_tracker(state,values(),epoch)
            self.assertEqual(state['lr_phase'],0 if epoch<6 else 1 if epoch<9 else 2)
            self.assertEqual(state['plateau'],epoch==12)
        state=m.update_tracker(m.initial_tracker(),values(),0);key='128/K0_own_wall_f1'
        for epoch,delta in ((1,.0004),(2,.0008),(3,.0012)):
            current=values();current[key]+=delta;state=m.update_tracker(state,current,epoch)
        self.assertEqual(state['patience'],0);self.assertAlmostEqual(state['significant_reference'][key],.5012)
        self.assertEqual(state['raw_maxima'][key],.5012)
        for bad in ({},dict(values(),**{key:float('nan')})):
            with self.assertRaises(ValueError):m.update_tracker(m.initial_tracker(),bad,0)

    def test_metric_scope_and_reporting_pick_not_quality_or_promotion(self):
        report=synthetic_report();table=m.metric_table(report);self.assertTrue(set(m.KEYS)<=set(table))
        for kind in ('missing','eligibility','duplicate','TEST'):
            bad=copy.deepcopy(report);row=bad['profiles']['512']['8']['rooms20/N16']['all']
            if kind=='missing':row['metrics']['peer_wall_f1']=None
            elif kind=='eligibility':row['eligible']['peer_wall_f1']=7
            elif kind=='duplicate':bad['records'][-1]=copy.deepcopy(bad['records'][0])
            else:bad['TEST_opened']=True
            with self.assertRaises(ValueError):m.metric_table(bad)
        h=[dict(epoch=0,metrics=table,checkpoint='initial.pt'),dict(epoch=1,metrics=dict(table),checkpoint='epoch1.pt')]
        for size in m.base.SIZES:h[1]['metrics'][f'{size}/R8_peer_wall_f1']+=.1
        result=m.reporting_selection(h,protocol());self.assertEqual(result['reporting_pick']['epoch'],1)
        self.assertFalse(result['pick_allH_FPR_quality_passed']);self.assertTrue(result['reporting_pick_is_not_promotion'])
        h[1]['metrics']['128/K0_own_wall_f1']-=.02
        self.assertEqual(m.reporting_selection(h,protocol())['reporting_pick']['epoch'],0)

    def test_dip_recovery_blocks_stop_without_replacing_best_and_phase_window_is_local(self):
        state=m.update_tracker(m.initial_tracker(),values(.9),0)
        for epoch,value in enumerate((.1,.2,.3,.4,.5,.6),1):
            state=m.update_tracker(state,values(value),epoch)
            self.assertEqual(state['lr_phase'],0);self.assertFalse(state['plateau'])
            if epoch>1:self.assertEqual(state['patience'],0)
        self.assertEqual(state['significant_reference'],values(.9))
        self.assertTrue(any(t['lag']==1 for t in state['history'][-1]['recovery_trends']))
        # At epoch6 enter phase1 below the old best. Entry is a baseline, not a completed phase epoch.
        state=m.initial_tracker()
        for epoch,value in enumerate((.9,.1,.1,.1,.1,.1,.1)):state=m.update_tracker(state,values(value),epoch)
        self.assertEqual(state['lr_phase'],1);self.assertEqual(state['phase_entry_epoch'],6)
        # Deliberately make an older phase point tiny: it must not enter the new recovery window.
        state['history'][5]['metrics']=values(0.)
        for epoch in (7,8):
            state=m.update_tracker(state,values(.1),epoch);self.assertFalse(state['history'][-1]['recovery_trends']);self.assertEqual(state['lr_phase'],1)
        state=m.update_tracker(state,values(.1),9);self.assertEqual(state['lr_phase'],2)

    def test_full_pixel_recipient_chunk_gradient_and_old_loss_parity(self):
        from scripts.train_mapping_multiroom_v295 import stage_loss
        for h in (128,256,512):
            reference=ToyReadout();candidate=copy.deepcopy(reference)
            x=torch.linspace(-.3,.4,3*128).reshape(1,3,128).requires_grad_();y=x.detach().clone().requires_grad_()
            a={k:x+(k+.2)*x.sin() for k in m.HOPS};b={k:y+(k+.2)*y.sin() for k in m.HOPS}
            own=torch.zeros((1,3,h,h),dtype=torch.int8);own[...,::17,:]=1;own[:,1,:h//2]=-1;own[:,2]=-1
            targets={k:own for k in m.HOPS};scalar=m.monolithic_objective(reference,a,targets);scalar.backward()
            record=m.backward_staged(candidate,b,targets,recipient_chunk=2)
            self.assertAlmostEqual(float(scalar.detach()),record['total'],places=6);self.assertEqual(record['source_backward_calls'],1)
            torch.testing.assert_close(x.grad,y.grad,rtol=2e-5,atol=2e-7)
            for p,q in zip(reference.parameters(),candidate.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=2e-7)
            if h==128:
                heads=reference.decode(a[0],resolution=h);old,_=stage_loss(heads,own);new=m.losses.proper_terms(heads,own)['total']
                torch.testing.assert_close(old,new,rtol=0,atol=0)
        with self.assertRaises(ValueError):m.backward_staged(candidate,{0:y},{0:own})

    def test_actual_five_groups_checkpoint_Adam_RNG_and_next_update_resume(self):
        from scripts.test_mapping_multiscale_v299_v310 import MultiscaleV299Tests,private
        from scripts.mapping_multiscale_v299_v310 import MultiscaleV299
        MultiscaleV299Tests.setUpClass();model=MultiscaleV299.from_parent_payload(MultiscaleV299Tests.payload)
        own=private(128,2).numpy()[0];union=np.broadcast_to(own.max(0),own.shape).copy()
        scene=dict(private=own,forest={'edges':[(0,1,1.)]},targets={k:own if k==0 else union for k in m.HOPS})
        opt=torch.optim.AdamW(model.trainable_parameters(),lr=m.LRS[0],weight_decay=0.);fixed=m.immutable_state(model);initial=m.group_snapshot(model)
        log=m.train_step(model,opt,scene,'cpu',m.LRS[0]);self.assertTrue(all(x>0 for x in log['gradient_groups'].values()));m.assert_immutable(model,fixed)
        self.assertTrue(all(row['changed_elements']>0 for row in m.group_deltas(model,initial).values()))
        progress=m.initial_progress();progress.update(epoch=1,cursor=1,updates=1,phase='train',
            tracker=m.update_tracker(m.initial_tracker(),values(),0),epoch_draw_sha256='fixture')
        with tempfile.TemporaryDirectory() as temp:
            p=protocol();record=m.save_checkpoint(temp,model,opt,progress,p);expected=(random.random(),np.random.rand(),torch.rand(2))
            restored,restored_opt,saved,saved_record=m.load_checkpoint(temp,p)
            actual=(random.random(),np.random.rand(),torch.rand(2));assert_nested_equal(self,expected,actual)
            assert_nested_equal(self,opt.state_dict(),restored_opt.state_dict());self.assertEqual(saved,progress);self.assertEqual(record,saved_record)
            m.train_step(model,opt,scene,'cpu',m.LRS[0]);m.train_step(restored,restored_opt,scene,'cpu',m.LRS[0])
            self.assertEqual(model.weights_hash(),restored.weights_hash());assert_nested_equal(self,opt.state_dict(),restored_opt.state_dict())
            pointer=m.base.read(Path(temp)/'latest.json');pointer['sha256']='0'*64;m.base.write(Path(temp)/'latest.json',pointer)
            with self.assertRaises(ValueError):m.load_checkpoint(temp,p)

    def test_transaction_and_atomic_pointer_interruptions_preserve_last_good_checkpoint(self):
        specs=m.base.scene_plan('TRAIN');original=m.initial_progress();report=synthetic_report();report['seconds']=1.
        with mock.patch.object(m,'epoch_plan',side_effect=OSError('injected transition failure')):
            with self.assertRaises(OSError):m.evaluated_progress(original,report,specs,{'path':'epoch0.json','sha256':'fixture'})
        self.assertEqual(original,m.initial_progress())
        next_state=m.evaluated_progress(original,report,specs,{'path':'epoch0.json','sha256':'fixture'})
        m.validate_progress(next_state,specs);self.assertEqual(original,m.initial_progress())
        model=mock.Mock();model.to_payload.return_value={'state':'fixture'};model.weights_hash.return_value='fixture-model'
        opt=mock.Mock();opt.state_dict.return_value={'state':{}}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'latest.json.tmp').write_text('interrupted old fixed-name writer',encoding='utf-8')
            record=m.save_checkpoint(root,model,opt,next_state,protocol(),kind='epoch')
            self.assertEqual(next_state['history'][-1]['checkpoint'],record['path'])
            self.assertEqual(next_state['history'][-1]['model_sha256'],'fixture-model')
            saved=torch.load(root/record['path'],weights_only=False);self.assertEqual(saved['progress'],next_state)
            before=(root/'latest.json').read_bytes();snapshot=copy.deepcopy(next_state)
            with mock.patch.object(m,'write_json',side_effect=OSError('injected pointer write failure')):
                with self.assertRaises(OSError):m.save_checkpoint(root,model,opt,next_state,protocol(),kind='epoch')
            self.assertEqual((root/'latest.json').read_bytes(),before);self.assertEqual(next_state,snapshot)
            m.write_json(root/'latest.json',record);self.assertEqual(m.base.read(root/'latest.json'),record)
        for error in (TimeoutError(),InterruptedError(),m.ResourceLimitError()):self.assertEqual(m.failure_kind(error),'budget_limited_or_interrupted_not_plateau')
        for error in (ValueError(),RuntimeError(),OSError()):self.assertEqual(m.failure_kind(error),'failed_not_plateau')

    def test_native_preflight_proof_rejects_missing_gradient_or_wrong_witness(self):
        proof=dict(version=310,passed=True,optimizer_updates=0,TRAIN_only=True,model_unchanged=True,typed_roundtrip_exact=True,
            model_sha256=m.PARENT_STATE,protocol_sha256='fixture',source_manifest_sha256='fixture',artifacts={},
            rows=[dict(size=h,agents=16,rooms=20,index=3,wire_passed=True,full_native_pixels=True,optimizer_updates=0,
                peak_allocated_bytes=100,gradient_groups={g:1. for g in m.GROUPS},parent128_exact=h==128) for h in m.base.SIZES])
        with mock.patch.object(m.base,'read',return_value=proof),mock.patch.object(m.base,'sha',return_value='fixture'):m.require_preflight('fake')
        for kind in ('size','index','wire','zero','nonfinite','parent','update'):
            bad=copy.deepcopy(proof)
            if kind=='size':bad['rows'].pop()
            elif kind=='index':bad['rows'][0]['index']=0
            elif kind=='wire':bad['rows'][0]['wire_passed']=False
            elif kind=='zero':bad['rows'][0]['gradient_groups']['encoder']=0.
            elif kind=='nonfinite':bad['rows'][0]['gradient_groups']['encoder']=float('nan')
            elif kind=='parent':bad['rows'][0]['parent128_exact']=False
            else:bad['optimizer_updates']=1
            with mock.patch.object(m.base,'read',return_value=bad),mock.patch.object(m.base,'sha',return_value='fixture'),self.assertRaises(ValueError):m.require_preflight('fake')

    def test_resource_stop_and_existing_root_never_masquerade_plateau(self):
        p=protocol()
        with tempfile.TemporaryDirectory() as temp,mock.patch.object(m.shutil,'disk_usage',return_value=type('Disk',(),{'free':10**12})()):
            self.assertEqual(m.budget(10,p,temp,now=11),1)
            with self.assertRaises(TimeoutError):m.budget(10,p,temp,now=43210)
            with mock.patch.object(m,'STOP',True),self.assertRaises(InterruptedError):m.budget(10,p,temp,now=11)
            with mock.patch.object(m,'ROOT',str(Path(temp).resolve())),mock.patch.object(m,'registration',return_value=p),mock.patch.object(m,'ancestry') as parent:
                with self.assertRaises(FileExistsError):m.prepare(temp,'fake')
                parent.assert_not_called()

    def test_original128_array_and_actual_packet_parity_fail_closed(self):
        arrays=dict(own=np.zeros((2,128,128),np.int8),prediction=np.zeros((5,2,128,128),np.int8))
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);np.savez(folder/'a.npz',**arrays);(folder/'p.bin').write_bytes(b'actual')
            row=dict(arrays_file='a.npz',arrays_sha256=m.base.sha(folder/'a.npz'),saved_packet_stream='p.bin',packet_sha256=m.base.sha(folder/'p.bin'))
            m.parent_array_parity(arrays,[b'actual'],row,folder)
            bad=copy.deepcopy(arrays);bad['prediction'][0,0,0,0]=1
            with self.assertRaises(ValueError):m.parent_array_parity(bad,[b'actual'],row,folder)
            with self.assertRaises(ValueError):m.parent_array_parity(arrays,[b'tampered'],row,folder)


if __name__=='__main__':unittest.main()
