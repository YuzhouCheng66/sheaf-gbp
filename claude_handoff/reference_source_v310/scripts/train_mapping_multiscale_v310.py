"""Pinned v299 warm-start, native full-pixel training to registered plateau.

Fresh Adam at import, real Adam/RNG/cursor resume thereafter. Source-once
forest radio is unchanged. No TEST, implicit data creation, or capacity claim.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import socket
import sys
import time
import traceback
import numpy as np
import torch
from scripts import train_mapping_support_native_v306 as base
from scripts import mapping_support_training_v306 as losses
from scripts import mapping_multiroom_forest_v295 as forest

ROOT='/home/yuzhou/Desktop/sheaf-gbp/v299-multiscale-v310-5090'
PARENT='/home/yuzhou/Desktop/sheaf-gbp/fixed-chart-v299-r1-5090'
DATA='/home/yuzhou/Desktop/sheaf-gbp/support-native-pair-v306-5090'
CONFIG='mapping_v299_multiscale_continuation_v310.json'
SOURCE_SHA='b0b52faa464329c9d6251b71b4ac575d070622ebeb15e3a8bd850ebf90eb7154'
BANK_SHA='3e450d0675814f8fbfc8b8028df796fc19d5a42404abac460e6b1310ddf6ac8a'
PARENT_SHA='28cbc27ca76221c87810508bffd880a32f20c4a2a8a6cd614ee2ceaec165ccb1'
PARENT_STATE='3f52478660b73b2ba6b2032b9a79510be5a7d081ee9a22b7b3c27222e70008db'
PARENT_PINS={'report.json':'6a3f141e307614c76863b1a6e3d677ab6a9de7fe879ee3eae5e9c1574014b0fd',
    'outputs/joint_control/result.json':'8e9ba43f3b6cd26a99b38649b441a1a597c97f9598b5bcbb560aaf43fc7577b6',
    'external-manifest.json':'d1910be65c0366dffa4d50ccad4d1c5ac682aaee8ba339532950481f7480a853',
    'registered-protocol.json':'a8469b07d9096b08d7755871d91d3092b93e182fe42ba7488a51372e15d2a60c',
    'outputs/joint_control/validation-002000.json':'9e1d517005a0e1dec95613ca5136c9250ecd8bea943b95ce7143bbfa296f8cad'}
OVERLAY=('mapping_multiscale_v299_v310.py','test_mapping_multiscale_v299_v310.py',
    'train_mapping_multiscale_v310.py','test_train_mapping_multiscale_v310.py',
    'test_mapping_content_readout_v288.py','test_mapping_common_map_v284.py',
    'test_mapping_belief_readout_v282.py','test_mapping_recognition_hops_v277.py')
GROUPS=('encoder','mean_head','precision_head','lift','decoder')
HOPS=(0,1,2,4,8);WEIGHTS={0:.5,1:.125,2:.125,4:.125,8:.125};LRS=(5e-5,1.5e-5,4.5e-6)
MONITORS=('K0_own_wall_f1','R1_union_accuracy','R8_union_accuracy','R1_peer_wall_f1','R8_peer_wall_f1')
KEYS=tuple(f'{h}/{k}' for h in (128,256,512) for k in MONITORS)
STOP=False


class ResourceLimitError(OSError):
    """Registered resource stop, not an algorithm failure or plateau."""


def write_json(path,value):
    """Unique sibling temp: an interrupted write never poisons future resume."""
    path=Path(path);temp=path.with_name(path.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
    data=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'
    try:
        with temp.open('x',encoding='utf-8',newline='\n') as f:f.write(data);f.flush();os.fsync(f.fileno())
        temp.replace(path)
    finally:
        if temp.exists():temp.unlink()


def registration(path):
    p=base.read(path);pa=p['parent'];d=p['data'];m=p['model'];t=p['training'];s=p['plateau'];v=p['validation'];e=p['execution']
    base.require(p['version']==310 and p['root']==ROOT and not p['status'].startswith('DRAFT')
        and p['device']==dict(host='uk-linux',GPU_UUID=base.UUID,Windows_GPU=False,bigboy=False),'Final registered single5090 run required')
    base.require(pa['root']==PARENT and pa['checkpoint']=='outputs/joint_control/validated-002000.pt'
        and pa['checkpoint_sha256']==PARENT_SHA and pa['model_sha256']==PARENT_STATE
        and pa['weights_warmstart'] is True and pa['parent_optimizer_inherited'] is False,'Wrong inherited mainline checkpoint')
    for key,file in (('report_sha256','report.json'),('result_sha256','outputs/joint_control/result.json'),
            ('external_manifest_sha256','external-manifest.json'),('registered_protocol_sha256','registered-protocol.json'),
            ('validation_sha256','outputs/joint_control/validation-002000.json')):
        base.require(pa[key]==PARENT_PINS[file],'Parent pin changed '+key)
    base.require(d['root']==DATA and d['manifest_sha256']==BANK_SHA
        and d['registered_protocol_sha256']=='0a5e1469d9550d907f2a57520c56595eacd00b397c05ab9a9b40808562cd85fc'
        and d['native_sizes']==[128,256,512] and d['rooms']==[6,12,20] and d['agents']==[8,16]
        and d['TRAIN_scenes']==1776 and d['VALIDATION_scenes']==144 and d['draw_seed']==310101
        and d['epoch_updates']==1776 and d['TEST_opened'] is False and d['new_data_generated'] is False,'Immutable native data scope changed')
    base.require(m['new_parameters']==0 and m['mean_dim']==64 and m['diagonal_dim']==64
        and m['initial_128_exact_parent_parity_required'] is True and m['optimized_groups']==list(GROUPS)
        and m['implementation_metadata_pending'] is False
        and all(m[k] is False for k in ('private_decoder_skip','runtime_union_input','inference_teacher')),'Original model interface required')
    base.require(t['optimizer']=='AdamW' and t['fresh_optimizer_at_parent_import'] is True and t['weight_decay']==0
        and t['gradient_clip_norm']==5 and t['lr_phases']==list(LRS) and t['stages']==list(HOPS)
        and t['stage_weights']==list(WEIGHTS.values()) and t['recipient_chunk_size']==1
        and t['full_pixel_supervision'] is True and t['gradient_equivalence_rtol']==2e-5 and t['gradient_equivalence_atol']==2e-7
        and t['evaluation_every_epochs']==1 and t['save_latest_every_updates']==100
        and t['save_every_evaluated_epoch'] is True and t['resume_implemented_required'] is True,'Objective/execution recipe changed')
    base.require(s['native_sizes']==[128,256,512] and s['metrics_each_size']==list(MONITORS)
        and s['accuracy_min_delta']==.0005 and s['wall_f1_min_delta']==.001 and s['patience_evaluated_epochs']==3
        and s['minimum_epochs_before_first_lr_reduction']==6,'Plateau contract changed')
    guard=s['recovery_trend_guard']
    base.require(guard['lookback_epochs']==[1,2,3] and guard['phase_entry_baseline_allowed'] is True
        and guard['minimum_completed_phase_epochs_before_action']==3,'Recovery trend guard changed')
    base.require(v['all144_scenes_all_stages'] is True and v['actual_packet_serialization'] is True and v['raw_predictions_saved'] is True
        and v['protected128_max_drop']==dict(K0_own_wall_f1=.01,K0_own_typed_accuracy=.01,R8_union_accuracy=.01,
            R8_peer_wall_f1=.01,R8_door_free_recall=.02,R8_pair_topology_accuracy=.02)
        and v['protected128_R8_unknown_fpr_max_rise']==.005 and v['allH_R8_unknown_fpr_quality_max']==.05,'Selection/protection changed')
    base.require(e['source_parent']==DATA and e['source_manifest_sha256']==SOURCE_SHA and e['session_seconds_cap']==43200
        and e['minimum_free_disk_bytes']==30000000000 and e['GPU_only_authorized_5090'] is True
        and e['preflight_TRAIN_index']==3
        and e['never_kill_others'] is True and e['source_sealed'] is True,'Resource/source authority changed')
    base.require(p['communication']==dict(mean_dim=64,diagonal_dim=64,profile=4,actual_bytes_per_directed_packet=172,
        rounds=8,source_once_forest=True,setup_measured=False,whole_process_100x_claim=False)
        and p['communication_100x_claim'] is False and p['full_goal_complete'] is False,'Wire/claim scope changed')
    return p


def training_specs(manifest):
    specs=[row['spec'] for row in manifest['cases'].values() if row['spec']['split']=='TRAIN']
    expected=base.scene_plan('TRAIN');keys={base.scene_key(s) for s in specs}
    base.require(len(specs)==1776 and keys=={base.scene_key(s) for s in expected}
        and all(sum(s['size']==h for s in specs)==592 for h in base.SIZES),'Exact admitted TRAIN set required')
    return sorted(specs,key=base.scene_key)


def epoch_plan(specs,epoch):
    base.require(type(epoch) is int and epoch>=1 and len(specs)==1776 and len({base.scene_key(s) for s in specs})==1776
        and all(s['split']=='TRAIN' for s in specs),'Positive epoch and unique TRAIN set required')
    rng=np.random.default_rng(np.random.SeedSequence([310101,epoch]));streams={}
    for h in base.SIZES:
        for r in base.ROOMS:
            for n in base.AGENTS:
                values=sorted((s for s in specs if (s['size'],s['rooms'],s['agents'])==(h,r,n)),key=base.scene_key)
                streams[h,r,n]=[values[int(i)] for i in rng.permutation(len(values))]
    out=[]
    for offset in range(max(map(len,streams.values()))):
        active=[k for k,v in streams.items() if offset<len(v)]
        out.extend(streams[active[int(i)]][offset] for i in rng.permutation(len(active)))
    base.require(len(out)==1776 and {base.scene_key(s) for s in out}=={base.scene_key(s) for s in specs},'Epoch must contain each original scene once')
    return out


def backward_staged(model,features,targets,*,recipient_chunk=1):
    losses.validate_stages(features,targets)
    base.require(type(recipient_chunk) is int and recipient_chunk>=1,'Positive recipient chunk required')
    cotangents=[];records={};calls=0
    for k in HOPS:
        live=features[k];truth=targets[k];b,n,h,_=truth.shape;count=b*n
        leaf=live.detach().reshape(count,128).requires_grad_(True);flat=truth.reshape(count,h,h)
        accum={name:0. for name in ('total','known','wall')}
        for start in range(0,count,recipient_chunk):
            stop=min(count,start+recipient_chunk);f=leaf[start:stop].unsqueeze(0);t=flat[start:stop].unsqueeze(0)
            with torch.autocast(device_type=f.device.type,enabled=False):
                heads=model.decode(f,resolution=h);terms=losses.proper_terms(heads,t)
                scale=(stop-start)/count;weighted=WEIGHTS[k]*scale*terms['total']
            for key in accum:accum[key]+=scale*float(terms[key].detach())
            weighted.backward();calls+=1;del heads,terms,weighted
        base.require(leaf.grad is not None and bool(torch.isfinite(leaf.grad).all()),'Missing/nonfinite feature cotangent')
        cotangents.append(leaf.grad.detach().reshape(b,n,128));records[k]=dict(accum,
            observed_counts=(truth>=0).flatten(2).sum(-1).cpu().tolist(),wall_counts=(truth==1).flatten(2).sum(-1).cpu().tolist(),
            free_counts=(truth==0).flatten(2).sum(-1).cpu().tolist())
    torch.autograd.backward([features[k] for k in HOPS],cotangents)
    return dict(total=sum(WEIGHTS[k]*records[k]['total'] for k in HOPS),stages=records,weights=WEIGHTS,
        full_native_pixels=True,source_backward_calls=1,decoder_backward_calls=calls,optimizer_steps=0)


def monolithic_objective(model,features,targets):
    losses.validate_stages(features,targets)
    return sum(WEIGHTS[k]*losses.proper_terms(model.decode(features[k],resolution=targets[k].shape[-1]),targets[k])['total'] for k in HOPS)


def initial_tracker():
    return dict(last_epoch=-1,lr_phase=0,phase_entry_epoch=0,patience=0,significant_reference={},raw_maxima={},history=[],plateau=False)


def update_tracker(tracker,metrics,epoch):
    base.require(set(metrics)==set(KEYS) and all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in metrics.values()),'All15 finite same-scope metrics required')
    state=copy.deepcopy(tracker);base.require(type(epoch) is int and epoch==state['last_epoch']+1 and not state['plateau'],'Contiguous evaluated epochs required')
    improved=[]
    for key,value in metrics.items():
        threshold=.0005 if 'accuracy' in key else .001
        if key not in state['significant_reference'] or value-state['significant_reference'][key]>=threshold-1e-12:
            state['significant_reference'][key]=value;improved.append(key)
        state['raw_maxima'][key]=max(value,state['raw_maxima'].get(key,value))
    recovery=[];by_epoch={row['epoch']:row for row in state['history']}
    for lag in (1,2,3):
        previous=epoch-lag
        if previous<state['phase_entry_epoch'] or previous not in by_epoch:continue
        for key,value in metrics.items():
            gain=value-by_epoch[previous]['metrics'][key];threshold=.0005 if 'accuracy' in key else .001
            if gain>=threshold-1e-12:recovery.append(dict(metric=key,lag=lag,gain=gain))
    state['patience']=0 if improved or recovery else state['patience']+1
    event='significant_improvement' if improved else 'recovering_below_best' if recovery else 'no_significant_improvement'
    if epoch-state['phase_entry_epoch']>=3 and state['patience']>=3:
        if state['lr_phase']<2 and (state['lr_phase']>0 or epoch>=6):
            state['lr_phase']+=1;state['patience']=0;state['phase_entry_epoch']=epoch;event='registered_LR_reduction'
        elif state['lr_phase']==2:state['plateau']=True;event='registered_operational_plateau'
    state['last_epoch']=epoch;state['history'].append(dict(epoch=epoch,event=event,improved=improved,
        recovery_trends=recovery,metrics=dict(metrics),lr_phase=state['lr_phase'],phase_entry_epoch=state['phase_entry_epoch'],patience=state['patience']))
    return state


def metric_table(report):
    base.require(report['version']==310 and report['split']=='VALIDATION' and report['TEST_opened'] is False
        and report['actual_wire_audits_passed'] is True and len(report['records'])==144
        and {r['identity'] for r in report['records']}=={base.scene_key(s) for s in base.scene_plan('VALIDATION')},'Complete144 evaluation required')
    table={}
    names={0:('own_wall_f1','own_typed_accuracy'),1:('union_accuracy','peer_wall_f1'),
        8:('union_accuracy','peer_wall_f1','door_free_recall','pair_topology_accuracy','unknown_known_fpr')}
    for h in base.SIZES:
        for k,fields in names.items():
            for name in fields:
                cells=[]
                for rooms in base.ROOMS:
                    for n in base.AGENTS:
                        row=report['profiles'][str(h)][str(k)][f'rooms{rooms}/N{n}']['all'];value=row['metrics'].get(name)
                        base.require(type(value) in (int,float) and math.isfinite(value) and 0<=value<=1
                            and row['scenes']==8 and row['eligible'].get(name)==8
                            and row['eligible_recipient_counts']==[n]*8,'Incomplete metric '+name)
                        cells.append(value)
                table[f'{h}/{"K0" if k==0 else "R"+str(k)}_{name}']=float(np.mean(cells))
    return table


def reporting_selection(history,p):
    base.require(bool(history),'Initial complete metrics required');initial=history[0]['metrics'];eligible=[];metric_best={}
    for row in history:
        values=row['metrics'];protected=all(values['128/'+k]>=initial['128/'+k]-limit-1e-12 for k,limit in p['validation']['protected128_max_drop'].items())
        protected=protected and values['128/R8_unknown_known_fpr']<=initial['128/R8_unknown_known_fpr']+.005+1e-12
        if protected:eligible.append(row)
        for k,v in values.items():
            better=k not in metric_best or (v<metric_best[k]['value'] if 'fpr' in k else v>metric_best[k]['value'])
            if better:metric_best[k]=dict(value=v,epoch=row['epoch'],checkpoint=row['checkpoint'])
    score=lambda r:(np.mean([r['metrics'][f'{h}/R8_peer_wall_f1'] for h in base.SIZES]),
        np.mean([r['metrics'][f'{h}/R8_union_accuracy'] for h in base.SIZES]),-r['epoch'])
    pick=max(eligible,key=score) if eligible else None
    pareto=[r['epoch'] for r in history if not any(all(q['metrics'][k]>=r['metrics'][k] for k in KEYS)
        and any(q['metrics'][k]>r['metrics'][k] for k in KEYS) for q in history)]
    return dict(reporting_pick=None if pick is None else dict(epoch=pick['epoch'],checkpoint=pick['checkpoint']),
        protected_epochs=[r['epoch'] for r in eligible],metricwise_best=metric_best,pareto_epochs_15metrics=pareto,
        pick_allH_FPR_quality_passed=pick is not None and all(pick['metrics'][f'{h}/R8_unknown_known_fpr']<=.05 for h in base.SIZES),
        reporting_pick_is_not_promotion=True,full_goal_complete=False)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(root,model,opt,progress,p,*,kind='periodic'):
    folder=Path(root)/'checkpoints';folder.mkdir(exist_ok=True)
    path=folder/f"u{progress['updates']:09d}-e{progress['epoch']:04d}-{kind}-{time.time_ns()}.pt"
    snapshot=copy.deepcopy(progress)
    if kind=='epoch':snapshot['history'][-1].update(checkpoint=str(path.relative_to(root)),model_sha256=model.weights_hash())
    payload=dict(version=310,model=model.to_payload(),optimizer=opt.state_dict(),progress=snapshot,RNG=rng_state(),
        protocol_content_sha256=base.json_hash(p),parent_checkpoint_sha256=PARENT_SHA,
        parent_optimizer_inherited=False,v310_optimizer_resumable=True)
    temp=path.with_suffix('.tmp')
    with temp.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    temp.replace(path);record=dict(path=str(path.relative_to(root)),sha256=base.sha(path),model_sha256=model.weights_hash(),
        completed_updates=progress['updates'],epoch=progress['epoch'],phase=progress['phase'],kind=kind)
    write_json(path.with_suffix('.json'),record);write_json(Path(root)/'latest.json',record)
    progress.clear();progress.update(snapshot);return record


def load_checkpoint(root,p,device='cpu'):
    from scripts.mapping_multiscale_v299_v310 import MultiscaleV299
    root=Path(root).resolve();record=base.read(root/'latest.json');path=(root/record['path']).resolve()
    base.require(path.is_relative_to(root/'checkpoints') and base.sha(path)==record['sha256'],'Latest durable checkpoint invalid')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    base.require(payload['version']==310 and payload['protocol_content_sha256']==base.json_hash(p)
        and payload['parent_checkpoint_sha256']==PARENT_SHA and payload['v310_optimizer_resumable'] is True,'Incompatible actual resume state')
    model=MultiscaleV299.from_payload(payload['model']).to(device).eval()
    base.require(model.weights_hash()==record['model_sha256'] and payload['progress']['updates']==record['completed_updates'],'Checkpoint progress/state hash mismatch')
    opt=torch.optim.AdamW(model.trainable_parameters(),lr=LRS[payload['progress']['tracker']['lr_phase']],weight_decay=0.)
    opt.load_state_dict(payload['optimizer']);restore_rng(payload['RNG'])
    return model,opt,payload['progress'],record


def budget(start,p,root,*,now=None):
    if STOP:raise InterruptedError('Safe stop requested; resumable incomplete')
    elapsed=(time.monotonic() if now is None else now)-start
    if elapsed>=p['execution']['session_seconds_cap']:raise TimeoutError('Resource cap is NOT plateau')
    if shutil.disk_usage(root).free<p['execution']['minimum_free_disk_bytes']:raise ResourceLimitError('Disk safety floor; resumable incomplete')
    return elapsed


def _stop(signum,frame):
    global STOP
    STOP=True


def ancestry(p):
    pa=Path(p['parent']['root']);data=Path(p['data']['root'])
    for file,digest in PARENT_PINS.items():base.require(base.sha(pa/file)==digest,'Pinned parent changed '+file)
    base.require(base.sha(pa/p['parent']['checkpoint'])==PARENT_SHA,'Parent weights changed')
    result=base.read(pa/'outputs/joint_control/result.json');report=base.read(pa/'report.json')
    base.require(result==report['arms']['joint_control'] and result['completed_updates']==2000
        and result['endpoint']['sha256']==PARENT_SHA and result['endpoint']['model_sha256']==PARENT_STATE,'Parent endpoint binding failed')
    validation=base.read(pa/'outputs/joint_control/validation-002000.json')
    base.require(validation['weights_sha256']==PARENT_STATE and validation['step']==2000
        and len(validation['records'])==48 and validation['TEST_opened'] is False,'Wrong original128 validation')
    base.require(base.sha(data/'source-manifest.json')==SOURCE_SHA
        and base.sha(data/'registered-protocol.json')==p['data']['registered_protocol_sha256']
        and base.sha(data/'data-bank/manifest.json')==BANK_SHA,'Immutable native source/bank changed')
    seal=base.read(data/'source-manifest.json');base.check_files(data/'source',seal)
    specs=training_specs(base.read(data/'data-bank/manifest.json'))
    return seal,specs,validation


def prepare(root,config):
    root=Path(root).resolve();p=registration(config);base.require(str(root)==ROOT,'Registered root required')
    if root.exists():raise FileExistsError('Never overwrite existing run')
    seal,specs,parent_validation=ancestry(p);origin=Path(__file__).resolve().parent
    for name in OVERLAY:base.require((origin/name).is_file(),'Missing final overlay '+name)
    root.mkdir(parents=True);shutil.copytree(Path(DATA)/'source',root/'source')
    for name in OVERLAY:
        dest=root/'source/scripts'/name
        if dest.exists():raise FileExistsError('Never replace inherited source')
        shutil.copyfile(origin/name,dest)
    dest=root/'source/configs'/CONFIG
    if dest.exists():raise FileExistsError('Never replace inherited registration')
    shutil.copyfile(config,dest);write_json(root/'registered-protocol.json',p)
    files=dict(seal['files']);files.update({'scripts/'+n:base.sha(root/'source/scripts'/n) for n in OVERLAY});files['configs/'+CONFIG]=base.sha(dest)
    write_json(root/'source-manifest.json',dict(version=310,files=files,parent_manifest_sha256=SOURCE_SHA,protocol_sha256=base.sha(root/'registered-protocol.json')))
    write_json(root/'training-specs.json',dict(specs=specs,sha256=base.json_hash(specs),bank_manifest_sha256=BANK_SHA))
    shutil.copyfile(Path(PARENT)/'outputs/joint_control/validation-002000.json',root/'parent-validation.json')
    write_json(root/'status.json',dict(version=310,state='prepared_not_started',updates=0,resumable=False,TEST_opened=False))
    return dict(state='prepared_not_started',root=str(root))


def check(root):
    root=Path(root).resolve();p=registration(root/'registered-protocol.json');source=root/'source'
    base.require(str(root)==ROOT and Path(__file__).resolve()==source/'scripts'/Path(__file__).name,'Execute sealed registered driver only')
    seal=base.read(root/'source-manifest.json');base.require(seal['version']==310 and seal['parent_manifest_sha256']==SOURCE_SHA
        and seal['protocol_sha256']==base.sha(root/'registered-protocol.json'),'Seal binding changed')
    base.check_files(source,seal);_,specs,parent_validation=ancestry(p)
    stored=base.read(root/'training-specs.json');base.require(stored==dict(specs=specs,sha256=base.json_hash(specs),bank_manifest_sha256=BANK_SHA),'Cached epoch universe changed')
    base.require(base.sha(root/'parent-validation.json')==PARENT_PINS['outputs/joint_control/validation-002000.json'],'Parent validation mirror changed')
    from scripts.mapping_multiscale_v299_v310 import MultiscaleV299
    base.helpers()
    for name,module in tuple(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):base.require(Path(module.__file__).resolve().is_relative_to(source),'Unsealed import '+name)
    return p,specs,parent_validation


def make_model(p,device='cpu'):
    from scripts.mapping_multiscale_v299_v310 import MultiscaleV299
    path=Path(p['parent']['root'])/p['parent']['checkpoint'];base.require(base.sha(path)==PARENT_SHA,'Pinned parent checkpoint changed')
    model=MultiscaleV299.from_parent_payload(torch.load(path,map_location='cpu',weights_only=False)).to(device).eval()
    base.require(model.weights_hash()==PARENT_STATE and tuple(model.parameter_groups())==GROUPS,'Warm-start altered parent weights/groups')
    return model


def backward_scene(model,scene,device):
    model.zero_grad(set_to_none=True);own=torch.as_tensor(scene['private'][None],device=device)
    targets={k:torch.as_tensor(scene['targets'][k][None],device=device) for k in HOPS}
    result=forest.rollout(model,own,scene['forest']['edges'],decode_rounds=(),trace=False)
    record=backward_staged(model,{k:result['features'][k] for k in HOPS},targets)
    params=model.trainable_parameters();base.require(all(x.grad is not None and bool(torch.isfinite(x.grad).all()) for x in params),'Disconnected/nonfinite gradient')
    groups={name:sum(float(x.grad.detach().double().square().sum()) for x in values)**.5 for name,values in model.parameter_groups().items()}
    f=result['source_factors'];mu=f['mu'].detach();q=f['learned_q'].detach();active=f['active'].detach()[...,None].expand_as(mu)
    values=mu[active];precision=q[active]
    record.update(gradient_groups=groups,source_activity=dict(active_coordinates=int(active.sum()),
        mean_std=float(values.float().std(unbiased=False)) if values.numel() else None,
        mean_abs1_fraction=float((values.abs()==1).float().mean()) if values.numel() else None,
        precision_min=float(precision.min()) if precision.numel() else None,precision_max=float(precision.max()) if precision.numel() else None))
    return record


def train_step(model,opt,scene,device,lr):
    record=backward_scene(model,scene,device);norm=torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),5.,error_if_nonfinite=True)
    for group in opt.param_groups:group['lr']=lr
    opt.step();record.update(lr=lr,gradient_norm=float(norm));return record


def immutable_state(model):
    return dict(buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()},unused=model.frozen_unused_hash())


def group_snapshot(model):
    return {key:torch.cat([p.detach().cpu().reshape(-1) for p in values]) for key,values in model.parameter_groups().items()}


def group_deltas(model,initial):
    current=group_snapshot(model);base.require(set(current)==set(initial)==set(GROUPS),'Weight-delta group schema changed')
    out={}
    for key,value in current.items():
        reference=initial[key];base.require(value.shape==reference.shape and bool(torch.isfinite(value).all()),'Invalid parameter delta input')
        delta=value.double()-reference.double();denominator=float(reference.double().norm())
        out[key]=dict(changed_elements=int((value!=reference).sum()),numel=value.numel(),max_abs=float(delta.abs().max()),
            l2=float(delta.norm()),relative_l2=float(delta.norm())/denominator if denominator else None)
    return out


def assert_immutable(model,snapshot):
    values=dict(model.named_buffers());base.require(set(values)==set(snapshot['buffers'])
        and all(torch.equal(values[k].detach().cpu(),v) for k,v in snapshot['buffers'].items())
        and model.frozen_unused_hash()==snapshot['unused'],'Fixed normalization/unused weights changed')


def original_arrays(scene,result):
    f=result['source_factors'];arrays=dict(own=scene['private'],positions_m=scene['positions_m'],trajectory_m=scene['trajectory_m'],
        physical_adjacency=scene['adjacency'],selected_adjacency=scene['forest']['selected_adjacency'],door_mask=scene['door_mask'],
        targets=np.stack([scene['targets'][k] for k in HOPS]),reached=np.stack([scene['reached'][k] for k in HOPS]),
        source_q=f['q'][0].diagonal(dim1=-2,dim2=-1).cpu().numpy(),source_h=f['h'][0].cpu().numpy(),source_mu=f['mu'][0].cpu().numpy())
    for key in ('codes','means','variances','features'):arrays[key]=np.stack([x[0].cpu().numpy() for x in result[key]])
    return arrays


def parent_array_parity(arrays,packets,row,parent_folder):
    folder=Path(parent_folder).resolve();path=(folder/row['arrays_file']).resolve();packet=(folder/row['saved_packet_stream']).resolve()
    base.require(path.is_relative_to(folder) and packet.is_relative_to(folder) and base.sha(path)==row['arrays_sha256']
        and base.sha(packet)==row['packet_sha256'],'Original array/packet provenance failed')
    with np.load(path,allow_pickle=False) as old:
        base.require(set(old.files)<=set(arrays) and all(np.array_equal(old[key],arrays[key]) for key in old.files),'Initial128 original arrays differ')
    base.require(packet.read_bytes()==b''.join(packets),'Initial128 literal packet stream differs')


@torch.no_grad()
def evaluate(model,root,bank,epoch,updates,device,p,started,parent_validation):
    root=Path(root);target=root/'validation'/f'epoch-{epoch:04d}.json'
    if target.exists():
        old=base.read(target);base.require(old['weights_sha256']==model.weights_hash() and old['updates']==updates
            and old['epoch']==epoch and len(old['records'])==144,'Published validation/model mismatch')
        for row in old['records']:
            for filekey,hashkey in (('arrays_file','arrays_sha256'),('saved_packet_stream','packet_sha256')):
                base.require(base.sha(root/row[filekey])==row[hashkey],'Published validation artifact changed')
        return old
    folder=root/'validation'/f'epoch-{epoch:04d}-attempt-{time.time_ns()}';folder.mkdir(parents=True)
    _,_,_,metrics=base.helpers();before=model.weights_hash();records=[];tick=time.monotonic()
    oldrows={(r['spec']['rooms'],r['spec']['agents'],r['spec']['index']):r for r in parent_validation['records']}
    for spec in base.scene_plan('VALIDATION'):
        budget(started,p,root);scene=base.scene_from_bank(bank,spec);identity=base.scene_key(spec);edges=scene['forest']['edges']
        reference=oldrows[(spec['rooms'],spec['agents'],spec['index'])] if epoch==0 and spec['size']==128 else None
        contexts=({(x['sender'],x['receiver']):bytes.fromhex(x['nonce']) for x in reference['contexts']} if reference else
            {(i,j):hashlib.sha256(f'v310/{epoch}/{identity}/{i}/{j}'.encode()).digest()[:16] for i,j,_ in edges})
        own=torch.as_tensor(scene['private'][None],device=device)
        result=forest.rollout(model,own,edges,decode_rounds=(),serialize=True,edge_contexts=contexts,trace=False)
        wire=forest.audit_rollout(result,edges,contexts);base.require(wire['passed'] and wire['actual_data_bytes']==2*len(edges)*8*172,'Actual paid receive audit failed')
        arrays=original_arrays(scene,result);predictions=[];hops={};proper={};counts={}
        for k in HOPS:
            heads=model.decode(result['features'][k],resolution=spec['size'])
            terms=losses.proper_terms(heads,torch.as_tensor(scene['targets'][k][None],device=device))
            known=heads['known_logits'][0].cpu().numpy().reshape(scene['private'].shape);wall=heads['wall_logits'][0].cpu().numpy().reshape(scene['private'].shape)
            pred=np.where(known>=0,(wall>=0).astype(np.int8),-1).astype(np.int8);predictions.append(pred)
            hops[str(k)]=metrics.scene_metrics(scene,pred,k);proper[str(k)]={v:float(terms[v]) for v in ('known','wall','total')}
            counts[str(k)]={label:(scene['targets'][k]==value).sum((1,2)).tolist() for label,value in (('wall',1),('free',0),('unknown',-1))}
            if spec['index']==0:arrays[f'known_logits_K{k}']=known;arrays[f'wall_logits_K{k}']=wall
            del heads,terms
        arrays['prediction']=np.stack(predictions)
        if reference:parent_array_parity(arrays,result['packets'],reference,Path(PARENT)/'outputs/joint_control')
        path=folder/(identity+'.npz');base.save_npz(path,**arrays);packet=folder/(identity+'-packets.bin')
        with packet.open('xb') as f:f.write(b''.join(result['packets']))
        row=dict(spec=spec,identity=identity,manifest=scene['metadata']['manifest'],edges=edges,public_frame=scene['public_frame'],hops=hops,
            proper_losses=proper,observed_class_counts=counts,wire=wire,contexts=[dict(sender=i,receiver=j,nonce=v.hex()) for (i,j),v in contexts.items()],
            arrays_file=str(path.relative_to(root)),arrays_sha256=base.sha(path),saved_packet_stream=str(packet.relative_to(root)),packet_sha256=base.sha(packet),
            packet_lengths=list(map(len,result['packets'])),initial_parent128_array_exact=True if reference else None)
        write_json(folder/(identity+'.json'),row);records.append(row)
        if len(records)%12==0:write_json(folder/'progress.json',dict(scenes=len(records),total=144,epoch=epoch,updates=updates))
        del own,scene,result,arrays
    base.require(model.weights_hash()==before,'Validation changed model');budget(started,p,root)
    report=dict(version=310,epoch=epoch,updates=updates,split='VALIDATION',records=records,profiles=base.aggregate(records),weights_sha256=before,
        actual_wire_audits_passed=True,raw_predictions_not_overwritten=True,TEST_opened=False,seconds=time.monotonic()-tick,
        bank_manifest_sha256=BANK_SHA,protocol_sha256=base.sha(root/'registered-protocol.json'),
        initial_parent128_array_exact=all(r['initial_parent128_array_exact'] for r in records if r['spec']['size']==128) if epoch==0 else None,
        logits_saved_scope='All five stages,18 fixed index0 native scenes; all144 raw maps/factors/packets',
        communication_100x_claim=False,full_goal_complete=False)
    metric_table(report);write_json(target,report);return report


def validate_progress(progress,specs):
    base.require(type(progress['updates']) is int and progress['updates']>=0 and progress['phase'] in ('train','evaluate','complete'),'Invalid durable phase')
    epoch,cursor=progress['epoch'],progress['cursor'];base.require(type(epoch) is int and epoch>=0 and type(cursor) is int and 0<=cursor<=1776,'Invalid draw cursor')
    if epoch==0:base.require(progress['updates']==cursor==0 and progress['phase']=='evaluate','Invalid initial validation state')
    else:
        plan=epoch_plan(specs,epoch);base.require(progress['epoch_draw_sha256']==base.json_hash(plan)
            and progress['updates']==(epoch-1)*1776+cursor,'Resume draw prefix/cursor mismatch')
    base.require(progress['tracker']['last_epoch']==(epoch if progress['phase']=='complete' else epoch-1),
        'Validation/plateau phase mismatch')


def claim_session(root,resume):
    root=Path(root);path=root/'process.lock'
    if path.exists():
        prior=base.read(path);base.require(resume and prior['host']==socket.gethostname(),'Foreign/live session lock requires attention')
        try:os.kill(prior['pid'],0)
        except ProcessLookupError:path.rename(root/f'closed-stale-lock-{time.time_ns()}.json')
        else:raise RuntimeError('Existing PID still present; never kill it')
    with path.open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),host=socket.gethostname()),f)
    folder=root/'sessions'/f'session-{time.time_ns()}';folder.mkdir(parents=True);return path,folder


def status(root,state,progress,**more):
    write_json(Path(root)/'status.json',dict(version=310,state=state,pid=os.getpid(),updates=progress['updates'],
        epoch=progress['epoch'],cursor=progress['cursor'],phase=progress['phase'],lr_phase=progress['tracker']['lr_phase'],
        operational_plateau=progress['tracker']['plateau'],resumable=not progress['tracker']['plateau'],TEST_opened=False,**more))


def preflight(root,device='cuda:0',gpu_uuid=base.UUID):
    root=Path(root).resolve();p,_,_=check(root)
    base.require(not (root/'started.json').exists(),'Run already started');folder=root/'preflight';folder.mkdir(exist_ok=False)
    tick=time.monotonic();rows=[]
    try:
        budget(tick,p,root);environment=base.device_setup(base.read(Path(DATA)/'registered-protocol.json'),device,gpu_uuid)
        bank=base.admitted_bank(DATA);model=make_model(p,device);before=model.weights_hash();fixed=immutable_state(model)
        from scripts.mapping_content_readout_v288 import ContentReadout64
        parent=ContentReadout64.from_payload(torch.load(Path(PARENT)/p['parent']['checkpoint'],map_location='cpu',weights_only=False)).to(device).eval()
        _,_,_,metrics=base.helpers()
        for h in base.SIZES:
            start=time.monotonic();spec=dict(split='TRAIN',size=h,rooms=20,agents=16,index=p['execution']['preflight_TRAIN_index']);scene=base.scene_from_bank(bank,spec)
            data_seconds=time.monotonic()-start;own=torch.as_tensor(scene['private'][None],device=device);edges=scene['forest']['edges']
            contexts={(i,j):hashlib.sha256(f'v310/preflight/{h}/{i}/{j}'.encode()).digest()[:16] for i,j,_ in edges}
            parity=None
            with torch.no_grad():
                actual=forest.rollout(model,own,edges,decode_rounds=(),serialize=True,edge_contexts=contexts,trace=False)
                audit=forest.audit_rollout(actual,edges,contexts);base.require(audit['passed'],'Native receiver audit failed')
                if h==128:
                    old=forest.rollout(parent,own,edges,decode_rounds=HOPS,serialize=True,edge_contexts=contexts,trace=False)
                    base.require(old['packets']==actual['packets'],'Original128 packet parity failed')
                    for k in ('q','h','mu','learned_q'):base.require(torch.equal(old['source_factors'][k],actual['source_factors'][k]),'Original128 source factor differs '+k)
                    for k in ('features','means','variances','codes'):base.require(all(torch.equal(a,b) for a,b in zip(old[k],actual[k])),'Original128 beliefs differ')
                    for k in HOPS:
                        heads=model.decode(actual['features'][k],resolution=128)
                        base.require(all(torch.equal(heads[name],old['decoded'][k][name]) for name in heads),'Original128 decoder differs')
                    parity=True;del old,heads
            del actual
            backward_scene(model,scene,device);model.zero_grad(set_to_none=True);torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats();start=time.monotonic();log=backward_scene(model,scene,device);torch.cuda.synchronize();backward_seconds=time.monotonic()-start
            base.require(set(log['gradient_groups'])==set(GROUPS) and all(math.isfinite(x) and x>0 for x in log['gradient_groups'].values()),'All five intended gradients must be live')
            model.zero_grad(set_to_none=True);start=time.monotonic()
            with torch.no_grad():
                actual=forest.rollout(model,own,edges,decode_rounds=(),serialize=True,edge_contexts=contexts,trace=False)
                audit=forest.audit_rollout(actual,edges,contexts);base.require(audit['passed'],'Timed literal receive failed')
                arrays=original_arrays(scene,actual);predictions=[];scores={}
                for k in HOPS:
                    heads=model.decode(actual['features'][k],resolution=h)
                    terms=losses.proper_terms(heads,torch.as_tensor(scene['targets'][k][None],device=device))
                    known=heads['known_logits'][0].cpu().numpy().reshape(scene['private'].shape);wall=heads['wall_logits'][0].cpu().numpy().reshape(scene['private'].shape)
                    pred=np.where(known>=0,(wall>=0).astype(np.int8),-1).astype(np.int8);predictions.append(pred)
                    scores[k]=dict(total=float(terms['total']),metrics=metrics.scene_metrics(scene,pred,k));del heads,terms
                arrays['prediction']=np.stack(predictions);base.save_npz(folder/f'TRAIN-H{h}.npz',**arrays)
                with (folder/f'TRAIN-H{h}-packets.bin').open('xb') as f:f.write(b''.join(actual['packets']))
                write_json(folder/f'TRAIN-H{h}.json',dict(scores=scores,wire=audit))
            torch.cuda.synchronize();evaluation_seconds=time.monotonic()-start
            rows.append(dict(size=h,agents=16,rooms=20,index=spec['index'],full_native_pixels=True,optimizer_updates=0,
                parent128_exact=parity,wire_passed=True,backward_seconds=backward_seconds,evaluation_seconds=evaluation_seconds,
                data_seconds=data_seconds,peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),gradient_groups=log['gradient_groups'],loss=log))
            write_json(folder/'progress.json',dict(rows=rows,optimizer_updates=0));del scene,own,actual,arrays
        base.require(model.weights_hash()==before==parent.weights_hash(),'Preflight changed inherited tensors');assert_immutable(model,fixed)
        from scripts.mapping_multiscale_v299_v310 import MultiscaleV299
        restored=MultiscaleV299.from_payload(model.to_payload());base.require(restored.weights_hash()==before,'Typed adapter roundtrip failed')
        check(root)
        proof=dict(version=310,passed=True,optimizer_updates=0,TRAIN_only=True,model_unchanged=True,model_sha256=before,
            typed_roundtrip_exact=True,rows=rows,environment=environment,seconds=time.monotonic()-tick,
            conservative_epoch_seconds=1.2*sum(592*(r['backward_seconds']+r['data_seconds'])+48*(r['evaluation_seconds']+r['data_seconds']) for r in rows)+30.,
            estimate_qualification='N16 costs charged to every N8 scene; optimizer/logging overhead estimated, not measured. No finite runtime implies plateau.',
            protocol_sha256=base.sha(root/'registered-protocol.json'),source_manifest_sha256=base.sha(root/'source-manifest.json'),
            artifacts={f.name:base.sha(f) for f in folder.iterdir() if f.is_file()})
        write_json(root/'preflight.json',proof);return proof
    except BaseException:
        write_json(folder/'failure.json',dict(version=310,optimizer_updates=0,rows=rows,error=traceback.format_exc()));raise


def require_preflight(root):
    root=Path(root);r=base.read(root/'preflight.json')
    base.require(r['version']==310 and r['passed'] is True and r['optimizer_updates']==0 and r['TRAIN_only'] is True
        and r['model_unchanged'] is True and r['typed_roundtrip_exact'] is True and r['model_sha256']==PARENT_STATE
        and r['protocol_sha256']==base.sha(root/'registered-protocol.json') and r['source_manifest_sha256']==base.sha(root/'source-manifest.json'),
        'Valid pinned zero-update admission required')
    base.require(len(r['rows'])==3 and {x['size'] for x in r['rows']}==set(base.SIZES),'All native resolutions required')
    for row in r['rows']:
        base.require(row['agents']==16 and row['rooms']==20 and row['index']==3 and row['wire_passed'] is True
            and row['full_native_pixels'] is True and row['optimizer_updates']==0 and row['peak_allocated_bytes']>0
            and set(row['gradient_groups'])==set(GROUPS) and all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in row['gradient_groups'].values())
            and (row['size']!=128 or row['parent128_exact'] is True),'Invalid native admission row')
    for name,digest in r['artifacts'].items():base.require(Path(name).name==name and base.sha(root/'preflight'/name)==digest,'Preflight witness changed')
    return r


def initial_progress():
    return dict(updates=0,epoch=0,cursor=0,epoch_draw_sha256=None,phase='evaluate',tracker=initial_tracker(),
        history=[],last_completed_validation=None,training_seconds=0.,validation_seconds=0.)


def evaluated_progress(progress,report,specs,validation):
    """Pure phase transaction; failed evaluation bookkeeping leaves prior state intact."""
    values=metric_table(report);next_progress=copy.deepcopy(progress)
    next_progress['tracker']=update_tracker(progress['tracker'],{k:values[k] for k in KEYS},progress['epoch'])
    next_progress['validation_seconds']+=report['seconds'];next_progress['last_completed_validation']=validation
    next_progress['history'].append(dict(epoch=progress['epoch'],updates=progress['updates'],metrics=values,
        validation=validation,checkpoint=None))
    if next_progress['tracker']['plateau']:next_progress['phase']='complete'
    else:
        next_progress['epoch']+=1;next_progress['cursor']=0;next_progress['phase']='train'
        next_progress['epoch_draw_sha256']=base.json_hash(epoch_plan(specs,next_progress['epoch']))
    validate_progress(next_progress,specs);return next_progress


def failure_kind(error):
    return 'budget_limited_or_interrupted_not_plateau' if isinstance(error,(TimeoutError,InterruptedError,KeyboardInterrupt,ResourceLimitError)) else 'failed_not_plateau'


def run(root,*,resume=False,device='cuda:0',gpu_uuid=base.UUID):
    root=Path(root).resolve();p,specs,parent_validation=check(root);require_preflight(root)
    base.require((root/'started.json').exists()==resume,'Use run exactly once, resume only an existing run')
    if resume:base.require(base.read(root/'status.json')['state'] in ('running','training','budget_limited_or_interrupted_not_plateau'),
        'Only resource/interrupted sessions resume; code failures require explicit review, plateau cannot autoextend')
    lock,session=claim_session(root,resume);start=time.monotonic();progress=initial_progress();model=opt=None;unsafe_update=False
    handlers={s:signal.signal(s,_stop) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        budget(start,p,root);environment=base.device_setup(base.read(Path(DATA)/'registered-protocol.json'),device,gpu_uuid)
        bank=base.admitted_bank(DATA);initial_groups=group_snapshot(make_model(p,'cpu'))
        if resume:model,opt,progress,_=load_checkpoint(root,p,device)
        else:
            with (root/'started.json').open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),version=310,parent_optimizer_inherited=False),f)
            random.seed(310101);np.random.seed(310101);torch.manual_seed(310101)
            model=make_model(p,device);opt=torch.optim.AdamW(model.trainable_parameters(),lr=LRS[0],weight_decay=0.)
            save_checkpoint(root,model,opt,progress,p,kind='initial')
        validate_progress(progress,specs);fixed=immutable_state(model)
        write_json(session/'start.json',dict(resume=resume,environment=environment,progress=progress,
            protocol_sha256=base.sha(root/'registered-protocol.json'),source_manifest_sha256=base.sha(root/'source-manifest.json')))
        with (session/'updates.jsonl').open('x',encoding='utf-8') as logfile:
            while not progress['tracker']['plateau']:
                budget(start,p,root);status(root,'running',progress,session=str(session.relative_to(root)))
                if progress['phase']=='evaluate':
                    report=evaluate(model,root,bank,progress['epoch'],progress['updates'],device,p,start,parent_validation)
                    path=root/'validation'/f"epoch-{progress['epoch']:04d}.json"
                    validation=dict(path=str(path.relative_to(root)),sha256=base.sha(path),epoch=progress['epoch'])
                    next_progress=evaluated_progress(progress,report,specs,validation)
                    next_progress['history'][-1]['initial_to_current_parameter_groups']=group_deltas(model,initial_groups);unsafe_update=True
                    for group in opt.param_groups:group['lr']=LRS[next_progress['tracker']['lr_phase']]
                    save_checkpoint(root,model,opt,next_progress,p,kind='epoch');progress=next_progress;unsafe_update=False
                    write_json(root/'selection.json',reporting_selection(progress['history'],p))
                    write_json(root/'learning-curves.json',progress['history']);continue
                plan=epoch_plan(specs,progress['epoch']);validate_progress(progress,specs)
                while progress['cursor']<len(plan):
                    budget(start,p,root);spec=plan[progress['cursor']];scene=base.scene_from_bank(bank,spec)
                    torch.cuda.synchronize();tick=time.monotonic();unsafe_update=True
                    log=train_step(model,opt,scene,device,LRS[progress['tracker']['lr_phase']]);torch.cuda.synchronize()
                    progress['training_seconds']+=time.monotonic()-tick;progress['updates']+=1;progress['cursor']+=1
                    assert_immutable(model,fixed);base.require(all(bool(torch.isfinite(x).all()) for x in model.parameters()),'Nonfinite updated weights')
                    log.update(update=progress['updates'],epoch=progress['epoch'],cursor=progress['cursor'],spec=spec,
                        epoch_draw_sha256=progress['epoch_draw_sha256']);logfile.write(json.dumps(log,sort_keys=True,allow_nan=False)+'\n');logfile.flush()
                    unsafe_update=False
                    del scene
                    if progress['updates']%100==0:
                        save_checkpoint(root,model,opt,progress,p);status(root,'training',progress,session_seconds=time.monotonic()-start)
                progress['phase']='evaluate';save_checkpoint(root,model,opt,progress,p,kind='evaluation-pending')
            check(root);assert_immutable(model,fixed)
            result=dict(version=310,state='complete_registered_plateau',operational_plateau=True,updates=progress['updates'],
                evaluated_epochs=progress['tracker']['last_epoch'],tracker=progress['tracker'],history=progress['history'],selection=reporting_selection(progress['history'],p),
                latest=base.read(root/'latest.json'),training_seconds=progress['training_seconds'],validation_seconds=progress['validation_seconds'],
                current_session_seconds=time.monotonic()-start,protocol_sha256=base.sha(root/'registered-protocol.json'),
                source_manifest_sha256=base.sha(root/'source-manifest.json'),bank_manifest_sha256=BANK_SHA,
                parent_checkpoint_sha256=PARENT_SHA,parent_optimizer_inherited=False,actual_Adam_resume_supported=True,
                TEST_opened=False,global_optimum_claim=False,capacity_impossibility_claim=False,communication_100x_claim=False,full_goal_complete=False,
                acceptance_status='pending_independent_raw_packet_audits_and_first_principles_review')
            write_json(root/'report.json',result);status(root,result['state'],progress);return result
    except BaseException as error:
        state=failure_kind(error)
        failure=dict(version=310,state=state,error=traceback.format_exc(),resumable=state!='failed_not_plateau',
            operational_plateau=False,progress=progress,session_seconds=time.monotonic()-start,
            failed_update_may_be_partial=unsafe_update,resume_from_last_durable_checkpoint_only=True)
        if model is not None and opt is not None and not unsafe_update and state!='failed_not_plateau':
            try:failure['recovery']=save_checkpoint(root,model,opt,progress,p,kind='recovery')
            except BaseException:failure['recovery_error']=traceback.format_exc()
        write_json(session/'failure.json',failure);write_json(root/'status.json',failure);raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        if lock.exists() and base.read(lock).get('pid')==os.getpid():lock.unlink()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','check','preflight','run','resume'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--config',type=Path)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--gpu-uuid',default=base.UUID);a=parser.parse_args()
    if a.mode=='prepare':
        if a.config is None:parser.error('prepare requires --config')
        answer=prepare(a.root,a.config)
    elif a.mode=='check':check(a.root);answer=dict(state='checked')
    elif a.mode=='preflight':answer=preflight(a.root,a.device,a.gpu_uuid)
    else:answer=run(a.root,resume=a.mode=='resume',device=a.device,gpu_uuid=a.gpu_uuid)
    print({k:answer[k] for k in ('state','root','passed','updates','evaluated_epochs') if k in answer},flush=True)


if __name__=='__main__':main()
