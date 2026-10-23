"""One sealed, fixed native128/256/512 pair; no automatic launch or resume.

Preparation of original sensors is an explicit CPU-only command. Training uses
the admitted staged full-pixel backward, unchanged forest transport, and no
private readout input. Recovery checkpoints are evidence, not resume support.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ARMS=('support_local','cross_address')
HOPS=(0,1,2,4,8)
STEPS=(0,1080,2160)
SIZES=(128,256,512)
ROOMS=(6,12,20)
AGENTS=(8,16)
UUID='GPU-238bdbd5-9f06-da6d-09f6-830fd831e817'
ROOT='/home/yuzhou/Desktop/sheaf-gbp/support-native-pair-v306-5090'
ADMISSION='/home/yuzhou/Desktop/sheaf-gbp/support-native-v306-r1-admission-5090'
ADMISSION_SHA='e2bee45dfd511c38bd90723d70134d6d400774d5c8cf06fc44147149c752a3c6'
ADMISSION_SEAL='6d62528248c8b88709c7645925cf40249a7f82efba74b23dbde320753c3f56e9'
CONFIG='mapping_support_native_pair_v306.json'
OVERLAY=('train_mapping_support_native_v306.py','test_train_mapping_support_native_v306.py')
STOP_REQUESTED=False


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('x',encoding='utf-8') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    temporary.replace(path)


def save_npz(path,**arrays):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('xb') as f:np.savez_compressed(f,**arrays)
    temporary.replace(path)


def registration(path):
    p=read(path);d=p['data'];t=p['training'];m=p['model'];v=p['validation'];g=p['gate']
    require(p['version']==306 and p['output_root']==ROOT and p['admission_root']==ADMISSION
        and p['admission_report_sha256']==ADMISSION_SHA and p['admission_source_manifest_sha256']==ADMISSION_SEAL
        and p['arms_in_sequential_order']==list(ARMS),'Wrong admitted fixed pair')
    require(p['device']==dict(host='uk-linux',UUID=UUID,Windows_GPU=False,bigboy=False),'Only authorized5090')
    require(d['native_sizes']==list(SIZES) and d['rooms']==list(ROOMS) and d['agents']==list(AGENTS)
        and d['TRAIN_indices']==[0,255] and d['VALIDATION_indices']==[0,7] and d['TEST_opened'] is False
        and d['draw_seed']==306101 and d['draws_per_arm']==2160 and d['TRAIN_pool_tuples']==4608
        and d['actual_unique_TRAIN_scenes']==1776 and d['actual_unique_TRAIN_worlds']==296
        and d['VALIDATION_scenes']==144 and d['VALIDATION_worlds']==24,'Fixed original native scene plan required')
    require(t['optimizer']=='AdamW' and t['lr_start']==1e-3 and t['lr_end']==1e-4 and t['weight_decay']==0
        and t['gradient_clip_norm']==5 and t['updates_per_arm']==2160 and t['stages']==list(HOPS)
        and t['stage_weights']==[.2]*5 and t['fixed_evaluations']==list(STEPS)
        and t['whole_pair_seconds_cap_after_preparation']==7200 and t['CPU_bank_preparation_seconds_cap']==3600
        and t['resume_implemented'] is False and t['automatic_extension'] is False and t['early_K0_stop'] is False,
        'Fixed optimizer/loss/budget required')
    require(m['mean_dim']==64 and m['diagonal_dim']==64 and m['seed']==306101 and m['fresh_both_arms'] is True
        and m['stored_parameters_each']==146730 and m['effective_parameters']==dict(support_local=129450,cross_address=146730)
        and m['decoder_private_skip'] is False and v['all_scenes_all_stages'] is True,'Fixed typed model required')
    require(g['K0_both_arms']==dict(macro_own_wall_f1_min=.80,worst_H_rooms_own_wall_f1_min=.70,
        macro_own_wall_precision_min=.80,macro_own_wall_recall_min=.75,macro_own_known_recall_min=.95)
        and g['primary']==dict(R8_macro_peer_wall_f1_gain_min=.05,macro_own_K0_to_R8_f1_retention_gain_min=.05)
        and g['protection']==dict(macro_K0_own_f1_drop_max=.02,macro_R8_own_f1_drop_max=.0,
            macro_R8_unknown_fpr_rise_max=.005,macro_R8_unknown_fpr_absolute_max=.05,
            macro_R8_topology_drop_max=.02,macro_R8_wall_precision_drop_max=.02,
            macro_R8_door_free_recall_drop_max=.02,worst_cell_R8_own_or_peer_f1_or_topology_drop_max=.05,
            worst_cell_R8_unknown_fpr_rise_max=.02)
        and g['unknown_metric_or_incomplete_is_pass'] is False,'Registered gates changed')
    require(p['communication']['profile']==4 and p['communication']['actual_bytes_per_directed_packet']==172
        and p['communication']['rounds']==8 and p['communication']['whole_process_100x_claim'] is False
        and p['full_goal_complete'] is False,'Unchanged transport and scope required')
    return p


def draw_plan():
    rng=np.random.default_rng(306101);result=[]
    for _ in range(120):
        block=[]
        for rooms in ROOMS:
            index=int(rng.integers(256))
            block.extend(dict(split='TRAIN',index=index,rooms=rooms,agents=n,size=h) for h in SIZES for n in AGENTS)
        result.extend(block[int(i)] for i in rng.permutation(18))
    return result


def scene_plan(split):
    require(split in ('TRAIN','VALIDATION'),'TEST is closed')
    if split=='TRAIN':return list({scene_key(s):s for s in draw_plan()}.values())
    return [dict(split=split,index=i,rooms=r,agents=n,size=h) for h in SIZES for r in ROOMS for n in AGENTS for i in range(8)]


def scene_key(spec):
    return f"{spec['split']}-H{spec['size']:03d}-r{spec['rooms']:02d}-n{spec['agents']:02d}-i{spec['index']:03d}"


def scheduled_lr(step):
    require(type(step) is int and 0<=step<2160,'Update index0..2159 required')
    return 1e-4+.5*(1e-3-1e-4)*(1+math.cos(math.pi*step/2159))


def budget_check(started,cap,*,now=None):
    elapsed=(time.monotonic() if now is None else now)-started
    if STOP_REQUESTED:raise InterruptedError('SIGTERM/INT requested: stop at complete operation boundary')
    if not math.isfinite(elapsed) or elapsed<0 or elapsed>=cap:raise TimeoutError('Registered resource cap reached')
    return elapsed


def helpers():
    from scripts import mapping_native_bank_v306 as bank
    from scripts import mapping_support_forest_v306 as forest
    from scripts import mapping_support_training_v306 as loss
    from scripts import train_mapping_multiroom_v295 as metrics
    return bank,forest,loss,metrics


def check_files(folder,manifest):
    folder=Path(folder).resolve()
    for name,digest in manifest['files'].items():
        path=(folder/name).resolve()
        require(path.is_relative_to(folder) and sha(path)==digest,'Sealed source changed: '+name)


def prepare(root,config):
    root=Path(root).resolve();p=registration(config);admission=Path(p['admission_root'])
    require(str(root)==p['output_root'],'Explicit registered fresh output required')
    if root.exists():raise FileExistsError('No duplicate preparation or overwrite')
    require(sha(admission/'report.json')==ADMISSION_SHA and sha(admission/'source-manifest.json')==ADMISSION_SEAL,'Successful admission pins changed')
    report=read(admission/'report.json');old=read(admission/'source-manifest.json')
    require(report['state']=='complete_cost_admission' and report['passed'] is True and report['optimizer_updates']==0
        and report['TEST_opened'] is False and report['source_manifest_sha256']==ADMISSION_SEAL,'Native admission did not pass')
    check_files(admission/'source',old)
    for split,digest in p['old_manifest_pins'].items():
        require(sha(Path(p['old_bank'])/(split+'-manifest.json'))==digest,'Original native128 bank changed')
    origin=Path(__file__).resolve().parent
    for name in OVERLAY:require((origin/name).is_file(),'Missing reviewed overlay '+name)
    root.mkdir(parents=True);shutil.copytree(admission/'source',root/'source')
    for name in OVERLAY:
        destination=root/'source/scripts'/name
        if destination.exists():raise FileExistsError('Never overwrite inherited source')
        shutil.copyfile(origin/name,destination)
    config_dest=root/'source/configs'/CONFIG;config_dest.parent.mkdir(exist_ok=True)
    if config_dest.exists():raise FileExistsError('Inherited registration collision')
    shutil.copyfile(config,config_dest)
    shutil.copyfile(admission/'report.json',root/'admission-report.json')
    shutil.copyfile(admission/'source-manifest.json',root/'admission-source-manifest.json')
    write(root/'registered-protocol.json',p)
    files=dict(old['files']);files.update({'scripts/'+n:sha(root/'source/scripts'/n) for n in OVERLAY})
    files['configs/'+CONFIG]=sha(config_dest)
    write(root/'source-manifest.json',dict(version=306,files=files,inherited_admission_source_sha256=ADMISSION_SEAL,
        protocol_sha256=sha(root/'registered-protocol.json')))
    draws=draw_plan();write(root/'draw-plan.json',dict(seed=306101,draws=draws,draw_sha256=json_hash(draws),updates=2160))
    write(root/'status.json',dict(state='prepared_no_data_no_training',version=306,updates=0))
    return dict(state='prepared_no_data_no_training',root=str(root))


def check(root):
    root=Path(root).resolve();p=registration(root/'registered-protocol.json');source=root/'source'
    require(str(root)==p['output_root'] and Path(__file__).resolve()==source/'scripts'/Path(__file__).name,'Execute sealed new driver only')
    manifest=read(root/'source-manifest.json')
    require(manifest['protocol_sha256']==sha(root/'registered-protocol.json') and manifest['inherited_admission_source_sha256']==ADMISSION_SEAL,'Protocol binding changed')
    check_files(source,manifest)
    require(sha(root/'admission-report.json')==ADMISSION_SHA and sha(root/'admission-source-manifest.json')==ADMISSION_SEAL,'Admitted cost evidence changed')
    plan=read(root/'draw-plan.json');expected=draw_plan()
    require(plan['draws']==expected and plan['draw_sha256']==json_hash(expected) and plan['updates']==2160 and plan['seed']==306101,'Shared realized draws changed')
    helpers()
    for name,module in tuple(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):
            require(Path(module.__file__).resolve().is_relative_to(source),'Unsealed script import: '+name)
    return p


def prepare_data(root):
    root=Path(root).resolve();p=check(root);bank_module,_,_,_=helpers()
    if (root/'data-started.json').exists() or (root/'started.json').exists():raise FileExistsError('No repeated/implicit bank preparation')
    write(root/'data-started.json',dict(pid=os.getpid(),training_updates=0));tick=time.monotonic();completed=0
    try:
        torch.set_num_threads(1)
        bank=bank_module.NativeBank.create(root/'data-bank',p['old_bank'],p['old_manifest_pins'])
        plan=scene_plan('TRAIN')+scene_plan('VALIDATION')
        require(len(plan)==1920 and len(scene_plan('TRAIN'))==1776,'Fixed unique bank scope')
        for spec in plan:
            budget_check(tick,3600)
            bank.register128(spec) if spec['size']==128 else bank.prepare(spec)
            completed+=1
            if completed%25==0:write(root/'status.json',dict(state='preparing_original_sensors',completed=completed,total=len(plan),seconds=time.monotonic()-tick,training_updates=0))
        budget_check(tick,3600)
        ready=dict(state='complete_data_preparation',completed=completed,TRAIN_scenes=1776,VALIDATION_scenes=144,
            seconds=time.monotonic()-tick,manifest_sha256=sha(root/'data-bank/manifest.json'),
            draw_sha256=json_hash(draw_plan()),protocol_sha256=sha(root/'registered-protocol.json'),training_updates=0)
        write(root/'data-ready.json',ready);write(root/'status.json',ready);return ready
    except BaseException:
        write(root/'data-failure.json',dict(error=traceback.format_exc(),completed=completed,seconds=time.monotonic()-tick,training_updates=0))
        write(root/'status.json',dict(state='failed_data_preparation',completed=completed,training_updates=0));raise


def admitted_bank(root):
    root=Path(root);ready=read(root/'data-ready.json');bank_module,_,_,_=helpers()
    require(ready['state']=='complete_data_preparation' and ready['completed']==1920 and ready['TRAIN_scenes']==1776
        and ready['VALIDATION_scenes']==144 and ready['seconds']<=3600 and ready['training_updates']==0
        and ready['protocol_sha256']==sha(root/'registered-protocol.json') and ready['draw_sha256']==json_hash(draw_plan()),'Incomplete/unbound bank preparation')
    return bank_module.NativeBank(root/'data-bank',expected_manifest_sha256=ready['manifest_sha256'])


def make_model(arm,device='cpu'):
    from scripts.mapping_support_local_v306 import SupportLocal64
    from scripts.mapping_support_pool_v306 import enable_native_pool
    require(arm in ARMS,'Unknown paired arm')
    model=SupportLocal64(arm=arm,seed=306101).to(device).eval();enable_native_pool(model)
    return model


def load_checkpoint(path,device='cpu'):
    from scripts.mapping_support_local_v306 import SupportLocal64
    from scripts.mapping_support_pool_v306 import enable_native_pool
    payload=torch.load(path,map_location='cpu',weights_only=False)
    require(payload['version']==306 and payload['arm'] in ARMS and 0<=payload['step']<=2160,'Wrong checkpoint')
    model=SupportLocal64.from_payload(payload).to(device).eval();enable_native_pool(model)
    require(model.arm==payload['arm'],'Typed checkpoint arm mismatch')
    return model,payload


def save_checkpoint(folder,model,opt,step,arm,p,draw_digest,name=None):
    require(type(step) is int and 0<=step<=2160 and model.arm==arm and arm in ARMS,'Actual completed step/arm required')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);path=folder/(name or f'checkpoint-{step:06d}.pt')
    if path.exists():raise FileExistsError('Do not replace a checkpoint')
    payload=model.to_payload();payload.update(version=306,arm=arm,step=step,completed_updates=step,
        optimizer_state=opt.state_dict(),actual_draw_sha256=draw_digest,protocol_sha256=json_hash(p),
        protocol_sha256_kind='canonical-json-content-not-file',
        execution_pool='FixedNativePool4',execution_transport=dict(profile=4,packet_bytes=172,rounds=8),
        recovery_is_resume=False,cpu_rng_state=torch.get_rng_state(),
        cuda_rng_state=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [])
    tmp=path.with_name(path.name+'.tmp');torch.save(payload,tmp);tmp.replace(path)
    return dict(path=str(path),sha256=sha(path),model_sha256=model.weights_hash(),step=step)


def scene_from_bank(bank,spec):
    module,_,_,_=helpers();cached=bank.get(spec);inp=cached['inputs'];targets={};reached={}
    for k in HOPS:targets[k],reached[k]=module.forest_targets(inp,k)
    return dict(spec=spec,private=inp['private'],positions_m=inp['positions_m'],trajectory_m=inp['trajectory_m'],
        adjacency=inp['adjacency'],public_frame=inp['public_frame'],door_mask=cached['evaluation']['door_mask'],
        metadata=cached['metadata'],forest=cached['forest'],targets=targets,reached=reached)


def train_step(model,opt,scene,device,step):
    _,core,loss,_=helpers();model.zero_grad(set_to_none=True)
    own=torch.as_tensor(scene['private'][None],device=device)
    targets={k:torch.as_tensor(scene['targets'][k][None],device=device) for k in HOPS}
    result=core.rollout(model,own,scene['forest']['edges'],features_only=True,trace=False)
    record=loss.backward_staged(model,{k:result['features'][k] for k in HOPS},targets)
    parameters=model.trainable_parameters()
    require(math.isfinite(record['total']) and all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in parameters),'Nonfinite/disconnected full gradient')
    groups={name:math.sqrt(sum(float(v.grad.detach().double().square().sum()) for v in values)) for name,values in model.parameter_groups().items()}
    norm=torch.nn.utils.clip_grad_norm_(parameters,5.,error_if_nonfinite=True)
    for group in opt.param_groups:group['lr']=scheduled_lr(step)
    opt.step()
    return dict(total=record['total'],stages=record['stages'],gradient_norm=float(norm),gradient_groups=groups,lr=scheduled_lr(step))


def aggregate(records):
    _,_,_,metrics=helpers()
    expected={scene_key(s) for s in scene_plan('VALIDATION')}
    require(len(records)==144 and {r['identity'] for r in records}==expected
        and all(r['identity']==scene_key(r['spec']) for r in records),'Complete unique native144 scene panel required')
    return {str(h):metrics.aggregate([r for r in records if r['spec']['size']==h]) for h in SIZES}


def paired_gate(local_endpoint,cross_endpoint,p=None):
    """Candidate is local, reference is cross-address; equal nine-cell macro."""
    checks=[];errors=[];records=(local_endpoint,cross_endpoint)
    expected={scene_key(s) for s in scene_plan('VALIDATION')}
    def add(name,passed,**values):checks.append(dict(name=name,passed=bool(passed),**values))
    def cells(record,k,name):
        result=[]
        for h in SIZES:
            for rooms in ROOMS:
                values=[]
                for n in AGENTS:
                    row=record['profiles'][str(h)][str(k)][f'rooms{rooms}/N{n}']['all']
                    value=row['metrics'].get(name)
                    require(type(value) in (int,float) and math.isfinite(value) and 0<=value<=1,'Missing/nonfinite '+name)
                    require(row['scenes']==8 and row['eligible'].get(name)==8
                        and row['eligible_recipient_counts']==[n]*8
                        and all(type(c) is int for c in row['eligible_recipient_counts']),
                        'Incomplete metric denominators '+name)
                    values.append(value)
                result.append(float(np.mean(values)))
        return np.asarray(result)
    try:
        for arm,record in zip(ARMS,records):
            rows=record['records']
            require(record['arm']==arm and record['step']==2160 and record['split']=='VALIDATION'
                and record['TEST_opened'] is False and record['actual_wire_audits_passed'] is True and len(rows)==144
                and {r['identity'] for r in rows}==expected and all(r['identity']==scene_key(r['spec']) for r in rows),
                'Incomplete actual endpoint '+arm)
            own=cells(record,0,'own_wall_f1')
            add(arm+'/K0/macroF1',own.mean()>=.80,value=float(own.mean()))
            add(arm+'/K0/worstF1',own.min()>=.70,value=float(own.min()))
            for key,minimum in (('own_wall_precision',.80),('own_wall_recall',.75),('own_known_recall',.95)):
                value=float(cells(record,0,key).mean());add(arm+'/K0/'+key,value>=minimum,value=value)
        l0=cells(local_endpoint,0,'own_wall_f1');c0=cells(cross_endpoint,0,'own_wall_f1')
        l8=cells(local_endpoint,8,'own_wall_f1');c8=cells(cross_endpoint,8,'own_wall_f1')
        peer=cells(local_endpoint,8,'peer_wall_f1')-cells(cross_endpoint,8,'peer_wall_f1')
        retention=(l8-l0)-(c8-c0)
        add('gain/R8/peerF1',peer.mean()>=.05-1e-12,delta=float(peer.mean()))
        add('gain/ownRetention',retention.mean()>=.05-1e-12,delta=float(retention.mean()))
        add('protect/K0/ownF1',(l0-c0).mean()>=-.02-1e-12,delta=float((l0-c0).mean()))
        add('protect/R8/ownF1',(l8-c8).mean()>=-1e-12,delta=float((l8-c8).mean()))
        for key in ('pair_topology_accuracy','wall_precision','door_free_recall'):
            delta=cells(local_endpoint,8,key)-cells(cross_endpoint,8,key)
            add('protect/R8/'+key,delta.mean()>=-.02-1e-12,delta=float(delta.mean()))
        fpr=cells(local_endpoint,8,'unknown_known_fpr');delta=fpr-cells(cross_endpoint,8,'unknown_known_fpr')
        add('protect/R8/FPRrise',delta.mean()<=.005+1e-12,delta=float(delta.mean()))
        add('protect/R8/FPRabsolute',fpr.mean()<=.05+1e-12,value=float(fpr.mean()))
        for key in ('own_wall_f1','peer_wall_f1','pair_topology_accuracy'):
            delta=cells(local_endpoint,8,key)-cells(cross_endpoint,8,key)
            for index,value in enumerate(delta):add(f'cell{index}/'+key,value>=-.05-1e-12,delta=float(value))
        delta=fpr-cells(cross_endpoint,8,'unknown_known_fpr')
        for index,value in enumerate(delta):add(f'cell{index}/FPR',value<=.02+1e-12,delta=float(value))
    except (KeyError,TypeError,ValueError) as exc:errors.append(str(exc))
    return dict(valid=not errors,passed=not errors and all(c['passed'] for c in checks),checks=checks,errors=errors,
        candidate='support_local',control='cross_address',full_goal_complete=False,whole_process_100x_claim=False)


@torch.no_grad()
def evaluate(model,root,bank,arm,step,device,*,started,cap):
    require(arm in ARMS and step in STEPS and model.arm==arm,'Fixed validation milestone required')
    root=Path(root);folder=root/'outputs'/arm;folder.mkdir(parents=True,exist_ok=True)
    examples=folder/f'validation-examples-{step:06d}';examples.mkdir(exist_ok=False)
    _,core,loss,metrics=helpers();before=model.weights_hash();records=[];tick=time.monotonic()
    for spec in scene_plan('VALIDATION'):
        budget_check(started,cap);scene=scene_from_bank(bank,spec);identity=scene_key(spec)
        own=torch.as_tensor(scene['private'][None],device=device);edges=scene['forest']['edges']
        contexts={(i,j):hashlib.sha256(f'v306/{identity}/{step}/{i}/{j}'.encode()).digest()[:16] for i,j,_ in edges}
        result=core.rollout(model,own,edges,features_only=True,serialize=True,edge_contexts=contexts,trace=False)
        wire=core.audit_rollout(result,edges,contexts)
        require(wire['passed'] is True and wire['actual_data_bytes']==len(edges)*2*8*172,'Actual packet/replay audit failed')
        packet_path=examples/(identity+'-packets.bin')
        with packet_path.open('xb') as f:f.write(b''.join(result['packets']))
        factors=result['source_factors']
        arrays=dict(own=scene['private'],positions_m=scene['positions_m'],trajectory_m=scene['trajectory_m'],
            physical_adjacency=scene['adjacency'],selected_adjacency=scene['forest']['selected_adjacency'],door_mask=scene['door_mask'],
            targets=np.stack([scene['targets'][k] for k in HOPS]),reached=np.stack([scene['reached'][k] for k in HOPS]),
            source_q=factors['q'][0].cpu().numpy(),source_h=factors['h'][0].cpu().numpy(),
            source_mu=factors['mu'][0].cpu().numpy(),source_data_q=factors['data_q'][0].cpu().numpy(),
            means=np.stack([v[0].cpu().numpy() for v in result['means']]),
            variances=np.stack([v[0].cpu().numpy() for v in result['variances']]),
            codes=np.stack([v[0].cpu().numpy() for v in result['codes']]),
            features=np.stack([v[0].cpu().numpy() for v in result['features']]))
        row=dict(spec=spec,identity=identity,manifest=scene['metadata']['manifest'],edges=edges,public_frame=scene['public_frame'],
            hops={},proper_losses={},wire=wire,contexts=[dict(sender=i,receiver=j,nonce=v.hex()) for (i,j),v in contexts.items()],
            saved_packet_stream=str(packet_path.relative_to(folder)),packet_sha256=sha(packet_path),packet_lengths=list(map(len,result['packets'])),
            reached_source_counts={str(k):scene['reached'][k].sum(1).tolist() for k in HOPS})
        predictions=[]
        for k in HOPS:
            heads=model.decode(result['features'][k],resolution=spec['size'])
            terms=loss.proper_terms(heads,torch.as_tensor(scene['targets'][k][None],device=device))
            row['proper_losses'][str(k)]={name:float(terms[name]) for name in ('total','known','wall')}
            known=heads['known_logits'][0].cpu().numpy().reshape(scene['private'].shape)
            wall=heads['wall_logits'][0].cpu().numpy().reshape(scene['private'].shape)
            raw=np.where(known>=0,(wall>=0).astype(np.int8),-1).astype(np.int8);predictions.append(raw)
            row['hops'][str(k)]=metrics.scene_metrics(scene,raw,k)
            if step==2160 and spec['index']==0 and spec['agents']==16 and k in (0,8):
                arrays[f'known_logits_K{k}']=known;arrays[f'wall_logits_K{k}']=wall
            del heads,terms
        arrays['prediction']=np.stack(predictions)
        if step==0 and arm=='cross_address':
            initial=root/'outputs/support_local/validation-examples-000000'/f'{identity}.npz'
            with np.load(initial,allow_pickle=False) as old:
                require(set(old.files)==set(arrays) and all(np.array_equal(old[key],value) for key,value in arrays.items()),'Initial function differs: '+identity)
            other=initial.with_name(identity+'-packets.bin');require(other.read_bytes()==packet_path.read_bytes(),'Initial actual packet mismatch')
            row['same_initial_function_exact']=True
        path=examples/(identity+'.npz');save_npz(path,**arrays)
        row.update(arrays_file=str(path.relative_to(folder)),arrays_sha256=sha(path))
        write(examples/(identity+'.json'),row)
        records.append(row)
        if len(records)%12==0:write(root/'status.json',dict(state='evaluating',arm=arm,completed_updates=step,scenes=len(records),total=144,pair_seconds=time.monotonic()-started))
        del result,arrays,scene,own
    budget_check(started,cap);require(model.weights_hash()==before,'Validation changed model')
    report=dict(version=306,arm=arm,step=step,split='VALIDATION',records=records,profiles=aggregate(records),
        seconds=time.monotonic()-tick,weights_sha256=before,actual_wire_audits_passed=True,raw_predictions_not_overwritten=True,
        TEST_opened=False,native_sizes=list(SIZES),same_initial_function_exact=step==0 and arm=='cross_address',
        protocol_sha256=sha(root/'registered-protocol.json'),bank_manifest_sha256=sha(root/'data-bank/manifest.json'),
        setup_measured=False,whole_process_100x_claim=False,loop_deduplication_claim=False,
        interior_metric_qualification='Inherited6-raster-pixel border; not common metric-width across H')
    write(folder/f'validation-{step:06d}.json',report);return report


def device_setup(p,device,gpu_uuid):
    require(device=='cuda:0' and gpu_uuid==UUID==p['device']['UUID'] and os.environ.get('CUDA_VISIBLE_DEVICES')==UUID,'Explicit authorized single5090 required')
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True)
    rows=[r for r in inventory.splitlines() if r.split(',')[0].strip()==UUID]
    require(len(rows)==1 and '5090' in rows[0],'Authorized5090 missing')
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    require(not any(row.split(',')[0].strip()==UUID for row in processes.splitlines() if row.strip()),'GPU not idle before initialization; do not kill tasks')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    require(os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8','Deterministic cuBLAS workspace required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'One visible GPU required')
    torch.cuda.set_per_process_memory_fraction(.85,0);torch.manual_seed(306101)
    return dict(gpu=rows[0].strip(),UUID=UUID,device=device,torch=torch.__version__,AMP=False,TF32=False,deterministic=True)


def _stop(signum,frame):
    global STOP_REQUESTED
    STOP_REQUESTED=True


def run(root,device='cuda:0',gpu_uuid=UUID):
    root=Path(root).resolve();p=check(root);bank=admitted_bank(root)
    with (root/'started.json').open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),resume=False),f)
    tick=time.monotonic();cap=7200;arms={};model=None;opt=None;arm=None;completed=0;attempted=None;actual_draws=[]
    old_handlers={s:signal.signal(s,_stop) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        environment=device_setup(p,device,gpu_uuid);draws=draw_plan();initial_hash=None
        for arm in ARMS:
            completed=0;attempted=None;actual_draws=[];arm_start=time.monotonic();training_seconds=0.;io_seconds=0.
            budget_check(tick,cap);model=make_model(arm,device);initial=model.weights_hash()
            if initial_hash is None:initial_hash=initial
            require(initial==initial_hash,'Initial tensors differ between arms')
            folder=root/'outputs'/arm;folder.mkdir(parents=True,exist_ok=False)
            opt=torch.optim.AdamW(model.trainable_parameters(),lr=1e-3,weight_decay=0.)
            buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
            torch.cuda.reset_peak_memory_stats();curves=[];logs=[];checkpoints=[]
            checkpoints.append(save_checkpoint(folder,model,opt,0,arm,p,json_hash(actual_draws)))
            curves.append(evaluate(model,root,bank,arm,0,device,started=tick,cap=cap))
            write(folder/'curves.json',curves)
            for step,spec in enumerate(draws):
                budget_check(tick,cap);attempted=step+1;start=time.monotonic();scene=scene_from_bank(bank,spec)
                io_seconds+=time.monotonic()-start;torch.cuda.synchronize();start=time.monotonic()
                log=train_step(model,opt,scene,device,step);torch.cuda.synchronize()
                training_seconds+=time.monotonic()-start;completed=step+1;actual_draws.append(spec)
                require(all(bool(torch.isfinite(v).all()) for v in model.parameters()),'Nonfinite updated parameter')
                require(all(torch.equal(v.detach().cpu(),buffers[k]) for k,v in model.named_buffers()),'Fixed public buffers changed')
                del scene
                if completed%50==0 or completed in STEPS:
                    logs.append(dict(step=completed,scene=spec,**log,seconds=time.monotonic()-arm_start))
                    write(folder/'train-log.json',logs)
                    write(root/'status.json',dict(state='training',arm=arm,completed_updates=completed,attempted_step=attempted,pair_seconds=time.monotonic()-tick))
                budget_check(tick,cap)
                if completed in STEPS:
                    checkpoints.append(save_checkpoint(folder,model,opt,completed,arm,p,json_hash(actual_draws)))
                    curves.append(evaluate(model,root,bank,arm,completed,device,started=tick,cap=cap));write(folder/'curves.json',curves)
            require(completed==2160 and actual_draws==draws,'Actual finite draw stream mismatch')
            result=dict(arm=arm,completed_updates=completed,initial_weights_sha256=initial,endpoint_weights_sha256=model.weights_hash(),
                actual_draw_sha256=json_hash(actual_draws),training_seconds=training_seconds,data_read_target_seconds=io_seconds,
                validation_seconds=sum(c['seconds'] for c in curves),wall_seconds=time.monotonic()-arm_start,
                peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),construction=model.construction_metadata(),
                execution_pool='FixedNativePool4',checkpoints=checkpoints,endpoint=curves[-1],
                actual_initial_to_endpoint={str(h):{'initial':curves[0]['profiles'][str(h)],'endpoint':curves[-1]['profiles'][str(h)]} for h in SIZES},
                buffers_unchanged=True,optimizer='AdamW',resume_implemented=False)
            write(folder/'result.json',result);arms[arm]=result
            del model,opt;model=None;opt=None;gc.collect();torch.cuda.empty_cache()
        gate=paired_gate(arms['support_local']['endpoint'],arms['cross_address']['endpoint'],p)
        budget_check(tick,cap)
        report=dict(version=306,state='complete_native_support_pair',arms=arms,gates=gate,all_registered_guards_passed=gate['passed'],
            elapsed_seconds=time.monotonic()-tick,CPU_bank_preparation=read(root/'data-ready.json'),environment=environment,
            source_manifest_sha256=sha(root/'source-manifest.json'),protocol_sha256=sha(root/'registered-protocol.json'),
            actual_draw_sha256=json_hash(draw_plan()),updates_per_arm=2160,same_initial_function_exact=True,
            TEST_opened=False,no_automatic_successor=True,full_goal_complete=False,whole_process_100x_claim=False)
        write(root/'report.json',report);write(root/'status.json',dict(state=report['state'],gate_passed=gate['passed'],completed_updates_per_arm=2160,seconds=report['elapsed_seconds']))
        return report
    except BaseException:
        failure=dict(state='failed_or_resource_stopped_native_pair',arm=arm,completed_updates=completed,attempted_step=attempted,
            seconds=time.monotonic()-tick,error=traceback.format_exc(),actual_draw_sha256=json_hash(actual_draws),completed_arms=list(arms),resume_implemented=False)
        if model is not None and opt is not None:
            try:failure['recovery']=save_checkpoint(root/'outputs'/arm,model,opt,completed,arm,p,json_hash(actual_draws),'recovery.pt')
            except BaseException:failure['recovery_error']=traceback.format_exc()
        write(root/'failure.json',failure);write(root/'status.json',failure);raise
    finally:
        for sig,handler in old_handlers.items():signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','data','run','check'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--config',type=Path)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--gpu-uuid',default=UUID)
    args=parser.parse_args()
    if args.mode=='prepare':
        if args.config is None:parser.error('prepare needs --config')
        result=prepare(args.root,args.config)
    elif args.mode=='data':
        signal.signal(signal.SIGTERM,_stop);signal.signal(signal.SIGINT,_stop);result=prepare_data(args.root)
    elif args.mode=='check':result=dict(state='checked',version=check(args.root)['version'])
    else:result=run(args.root,args.device,args.gpu_uuid)
    print({k:result[k] for k in ('state','root','elapsed_seconds','all_registered_guards_passed') if k in result},flush=True)


if __name__=='__main__':main()
