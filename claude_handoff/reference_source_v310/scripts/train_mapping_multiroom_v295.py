"""Sealed native128 real-sensor adaptation; no automatic launch or successor.

The two experiment arms retain the SAME typed v288 joint_belief model. Only
supervised stages differ. A centralized batch is simulation storage, never an
agent's inference permissions. Forest schedules are public/offline: measured
application frames are not a whole-process compression claim. Recovery files
are evidence, NOT an implemented optimizer-resume CLI.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

import numpy as np
import torch

from scripts import mapping_multiroom_data_v293 as data
from scripts import mapping_multiroom_forest_v295 as core
from scripts.mapping_content_readout_v288 import ContentReadout64
from scripts.mapping_hop_targets_v277 import hop_targets, reachable_sources
from scripts.mapping_observed_data_v246 import reconstruction_terms
from scripts.mapping_recognition_metrics_v277 import node_metrics

ARMS=('source_only','causal_joint')
HOPS=(0,1,2,4,8)
STEPS=(0,1000,2000,4000)
PARENT_SHA='3502ca39c00a58018f25d3f8093c99097a9de6cdd9f5bbe95537d982ce9cb99b'
PARENT_STATE='29762380a156af8d83a93d5c4799898b2f6205f8e8bf2dacb72b15db3c7bd277'
GPU_UUID='GPU-238bdbd5-9f06-da6d-09f6-830fd831e817'
COHORTS=('all','h_ge2','h_gt2','route_ge2','route_gt2','no_new_evidence')
CONFIG='mapping_multiroom_domain_adaptation_v295.json'
STOP_REQUESTED=False


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path):
    with open(path,encoding='utf-8-sig') as f:return json.load(f)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with open(temporary,'w',encoding='utf-8') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def save_npz(path,**arrays):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    with open(temporary,'wb') as f:np.savez_compressed(f,**arrays)
    os.replace(temporary,path)


def registration(path):
    p=read(path);d=p['data'];t=p['training'];e=p['execution']
    if (p.get('version')!=295 or p.get('TEST_opened') is not False
            or p['parent']['file_sha256']!=PARENT_SHA or p['parent']['model_state_sha256']!=PARENT_STATE
            or d['native_size']!=128 or d['room_counts']!=[6,12,20] or d['agent_counts']!=[8,16]
            or d['TRAIN_indices_inclusive']!=[0,255] or d['VALIDATION_indices_inclusive']!=[0,7]
            or d['TRAIN_scene_tuples']!=1536 or d['VALIDATION_scene_tuples']!=48
            or t['arms_in_sequential_order']!=list(ARMS) or t['seed']!=20261017
            or t['updates_per_arm']!=4000 or t['fixed_evaluation_updates']!=list(STEPS)
            or t['learning_rate_start']!=5e-5 or t['learning_rate_end']!=5e-6
            or t['gradient_clip_norm']!=5. or t['max_seconds_per_arm']!=6000
            or p['common_protocol']['stages']!=list(HOPS) or not t['no_auto_extension']
            or e['gpu_uuid']!=GPU_UUID or e['host']!='uk-linux'
            or e['windows_GPU_allowed'] is not False or e['bigboy_allowed'] is not False):
        raise ValueError('Registered v295 native128 paired protocol required; no TEST or silent variant')
    return p


def execution_transport(root):
    root=Path(root)
    return dict(profile=4,precision_range=[1e-8,128.],packet_bytes=172,rounds=8,
        forest_core_sha256=sha(root/'source/scripts/mapping_multiroom_forest_v295.py'),
        protocol_sha256=sha(root/'registered-protocol.json'),
        construction_transport_metadata_is_historical=True,setup_measured=False)


def scene_plan(split):
    if split not in ('TRAIN','VALIDATION'):raise ValueError('TEST is closed')
    return [dict(split=split,index=i,rooms=r,agents=n,size=128)
            for r in (6,12,20) for n in (8,16) for i in range(256 if split=='TRAIN' else 8)]


def draw_plan():
    return np.random.default_rng(20261017).integers(1536,size=4000,dtype=np.int64)


def scene_id(spec):
    if spec not in scene_plan(spec.get('split')):raise ValueError('Unregistered scene identity')
    return f"{spec['split']}-r{spec['rooms']:02d}-n{spec['agents']:02d}-i{spec['index']:03d}"


def scheduled_lr(step):
    if type(step) is not int or not 0<=step<4000:raise ValueError('Update index0..3999 required')
    return 5e-6+.5*(5e-5-5e-6)*(1+math.cos(math.pi*step/3999))


def forest_labels(private,edges):
    directed=tuple((a,b) for i,j,_ in edges for a,b in ((i,j),(j,i)))
    targets={k:hop_targets(private,directed,k) for k in HOPS}
    reached={k:core.reachability(edges,len(private),k) for k in HOPS}
    return targets,reached


class SceneBank:
    """Original sensor rasters on disk; no target or learned-factor cache."""
    def __init__(self,folder):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.entries={};self.generation_seconds=0.

    def get(self,spec):
        identity=scene_id(spec);path=self.folder/(identity+'.npz');meta_path=self.folder/(identity+'.json')
        if path.exists()!=meta_path.exists():raise RuntimeError('Incomplete original scene cache; do not silently overwrite')
        if not path.exists():
            start=time.monotonic();fixture=data.make_fixture(**spec);inp=fixture['inputs']
            arrays=dict(private=inp['private'],positions_m=inp['positions_m'],adjacency=inp['adjacency'],
                        trajectory_m=inp['trajectory_m'],door_mask=fixture['evaluation']['door_mask'])
            save_npz(path,**arrays)
            meta=dict(identity=identity,spec=spec,manifest=fixture['manifest'],public_frame=inp['public_frame'],
                      cache_sha256=sha(path),generation_seconds=time.monotonic()-start,
                      arrays_sha256={k:data.array_hash(v) for k,v in arrays.items()},targets_cached=False)
            write(meta_path,meta);self.generation_seconds+=meta['generation_seconds']
        meta=read(meta_path)
        if meta['spec']!=spec or meta['cache_sha256']!=sha(path):raise ValueError('Original scene cache identity/hash changed')
        with np.load(path,allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
        if {k:data.array_hash(v) for k,v in arrays.items()}!=meta['arrays_sha256']:raise ValueError('Sensor cache arrays differ')
        self.entries[identity]=dict(spec=spec,cache_sha256=meta['cache_sha256'],manifest=meta['manifest'])
        selected=core.select_forest(arrays['positions_m'],meta['public_frame']['communication_radius_m'],arrays['adjacency'])
        targets,reached=forest_labels(arrays['private'],selected['edges'])
        return dict(**arrays,metadata=meta,forest=selected,targets=targets,reached=reached)

    def build_train(self,status):
        for i,spec in enumerate(scene_plan('TRAIN')):
            if STOP_REQUESTED:raise InterruptedError('Stop requested during data preparation; no optimizer update')
            self.get(spec)
            if i%50==0:write(status,dict(state='building_original_TRAIN_bank',scenes=i+1,total=1536,training_updates=0))
        write(self.folder/'TRAIN-manifest.json',dict(scenes=self.entries,generation_seconds=self.generation_seconds,
            physical_worlds=768,original_only=True,targets_cached=False,learned_factors_cached=False,TEST_opened=False))


def tensors(scene,device):
    own=torch.as_tensor(scene['private'][None],device=device)
    targets={k:torch.as_tensor(v[None],device=device) for k,v in scene['targets'].items()}
    return own,targets


def stage_loss(output,target):
    if target.ndim!=4 or target.shape[0]!=1:raise ValueError('One whole scene with equal recipient weight required')
    b,n,h,w=target.shape
    if any(output[k].shape!=(b,n,h*w) for k in ('known_logits','wall_logits')):raise ValueError('Exact per-recipient flat heads required')
    flat={k:output[k].reshape(b*n,h*w) for k in ('known_logits','wall_logits')}
    if any(not bool(torch.isfinite(v).all()) for v in flat.values()):raise FloatingPointError('Nonfinite raw map logits')
    terms=reconstruction_terms(flat,target.reshape(b*n,h,w))
    return terms['known']+terms['wall'],terms


def objective(result,targets,arm):
    if arm not in ARMS:raise ValueError('Unknown learning arm')
    weights={0:1.} if arm=='source_only' else {0:.5,1:.125,2:.125,4:.125,8:.125}
    loss=None;parts={}
    for k,weight in weights.items():
        value,terms=stage_loss(result['decoded'][k],targets[k])
        parts[f'K{k}_known']=terms['known'];parts[f'K{k}_wall']=terms['wall']
        loss=value*weight if loss is None else loss+value*weight
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite proper objective')
    return loss,parts


def train_rollout(model,own,edges,arm):
    if arm not in ARMS:raise ValueError('Unknown learning arm')
    return core.rollout(model,own,edges,rounds=0 if arm=='source_only' else 8,
                        decode_rounds=(0,) if arm=='source_only' else HOPS,trace=False)


def load_model(checkpoint,device='cpu',*,initial=True):
    if initial and sha(checkpoint)!=PARENT_SHA:raise ValueError('Wrong current parent checkpoint file')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=ContentReadout64.from_payload(payload).to(device).eval()
    if model.arm!='joint_belief' or (initial and model.weights_hash()!=PARENT_STATE):raise ValueError('Wrong typed parent function')
    model.trainable_parameters()
    return model


def state_buffers(model):return {k:v.detach().cpu().clone() for k,v in model.named_buffers()}


def assert_buffers(model,snapshot):
    current=dict(model.named_buffers())
    if set(current)!=set(snapshot) or any(not torch.equal(current[k].detach().cpu(),v) for k,v in snapshot.items()):
        raise AssertionError('Fixed public/normalization buffer changed')


def gradient_groups(model,loss=None,*,require_nonzero=True):
    groups=model.parameter_groups();parameters=[v for group in groups.values() for v in group]
    gradients=(torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True) if loss is not None
               else [p.grad for p in parameters])
    result={};offset=0
    for name,group in groups.items():
        values=gradients[offset:offset+len(group)];offset+=len(group)
        if any(v is None or not bool(torch.isfinite(v).all()) for v in values):raise AssertionError('Disconnected/nonfinite '+name)
        result[name]=math.sqrt(sum(float(v.detach().double().square().sum()) for v in values))
        if require_nonzero and result[name]<=0:raise AssertionError('Zero active gradient '+name)
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise AssertionError('Frozen unused decoder received gradient')
    return result


def _sync(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)


@torch.no_grad()
def actual_rollout(model,own,edges):
    contexts={(i,j):os.urandom(16) for i,j,_ in edges}
    actual=core.rollout(model,own,edges,rounds=8,decode_rounds=HOPS,serialize=True,edge_contexts=contexts)
    audit=core.audit_rollout(actual,edges,contexts)
    expected=2*len(edges)*8*172
    if audit['actual_data_bytes']!=expected or audit['actual_packet_count']!=2*len(edges)*8:raise AssertionError('Paid schedule mismatch')
    return actual,audit,contexts


def mean_rows(rows):
    if not rows:return dict(metrics={},eligible={},recipients=0)
    keys=sorted(set.union(*(set(r) for r in rows)));metrics={};eligible={}
    for key in keys:
        values=[r[key] for r in rows if r.get(key) is not None and not isinstance(r[key],(bool,str,list,dict))]
        if any(not math.isfinite(float(v)) for v in values):raise ValueError('Nonfinite reported metric')
        metrics[key]=float(np.mean(values)) if values else None;eligible[key]=len(values)
    return dict(metrics=metrics,eligible=eligible,recipients=len(rows))


def scene_metrics(scene,prediction,k):
    rows=node_metrics(prediction,scene['private'],scene['targets'][k],{'door_mask':scene['door_mask']})
    physical=tuple(map(tuple,np.argwhere(scene['adjacency']).tolist()))
    distances={'h_ge2':reachable_sources(len(rows),physical,1),'h_gt2':reachable_sources(len(rows),physical,2),
               'route_ge2':core.reachability(scene['forest']['edges'],len(rows),1),
               'route_gt2':core.reachability(scene['forest']['edges'],len(rows),2)}
    for i,row in enumerate(rows):
        arrived=scene['reached'][k][i];target=scene['targets'][k][i]
        row['reached_source_count']=int(arrived.sum());row['has_new_evidence']=int(row['peer_cells']>0)
        local=scene['private'][i]>=0
        row['own_known_recall']=float((prediction[i][local]>=0).mean()) if local.any() else None
        for label,near in distances.items():
            within=scene['private'][arrived&near[i]].max(0)
            novel=(target>=0)&(within<0)
            row[label+'_novel_cells']=int(novel.sum());row[label+'_sources']=int((arrived&~near[i]).sum())
        correct=total=0
        for a,b in (((slice(None),slice(None,-1)),(slice(None),slice(1,None))),
                    ((slice(None,-1),slice(None)),(slice(1,None),slice(None)))):
            eligible=(target[a]>=0)&(target[b]>=0)
            truth=(target[a]==0)&(target[b]==0);guess=(prediction[i][a]==0)&(prediction[i][b]==0)
            correct+=int(((truth==guess)&eligible).sum());total+=int(eligible.sum())
        row['local_free_adjacency_accuracy']=correct/total if total else None
        row['local_free_adjacency_pairs']=total
    return dict(nodes=rows,all=mean_rows(rows),
                **{key:mean_rows([r for r in rows if r[key+'_novel_cells']>0]) for key in distances},
                no_new_evidence=mean_rows([r for r in rows if not r['has_new_evidence']]))


def aggregate(records):
    result={}
    for k in HOPS:
        result[str(k)]={}
        for rooms in (6,12,20):
            for n in (None,8,16):
                rows=[r for r in records if r['spec']['rooms']==rooms and (n is None or r['spec']['agents']==n)]
                key=f'rooms{rooms}/'+('allN' if n is None else f'N{n}')
                result[str(k)][key]={cohort:dict(**mean_rows([r['hops'][str(k)][cohort]['metrics'] for r in rows]),
                    scenes=len(rows),eligible_recipient_counts=[r['hops'][str(k)][cohort]['recipients'] for r in rows])
                    for cohort in COHORTS}
    return result


@torch.no_grad()
def evaluate(model,root,bank,arm,step,device):
    if arm not in ARMS or step not in STEPS:raise ValueError('Fixed registered evaluation endpoint required')
    folder=Path(root)/'outputs'/arm;folder.mkdir(parents=True,exist_ok=True)
    examples=folder/f'validation-examples-{step:06d}';examples.mkdir(exist_ok=False)
    before=model.weights_hash();records=[];tick=time.monotonic()
    for spec in scene_plan('VALIDATION'):
        scene=bank.get(spec);own,_=tensors(scene,device);result,wire,contexts=actual_rollout(model,own,scene['forest']['edges'])
        identity=scene_id(spec);path=examples/(identity+'.npz');packets=examples/(identity+'-packets.bin')
        with open(packets,'wb') as f:f.write(b''.join(result['packets']))
        arrays=dict(own=scene['private'],positions_m=scene['positions_m'],physical_adjacency=scene['adjacency'],
            selected_adjacency=scene['forest']['selected_adjacency'],door_mask=scene['door_mask'],
            targets=np.stack([scene['targets'][k] for k in HOPS]),reached=np.stack([scene['reached'][k] for k in HOPS]),
            codes=np.stack([v[0].cpu().numpy() for v in result['codes']]),
            means=np.stack([v[0].cpu().numpy() for v in result['means']]),
            variances=np.stack([v[0].cpu().numpy() for v in result['variances']]),
            source_q=result['source_factors']['q'][0].diagonal(dim1=-2,dim2=-1).cpu().numpy(),
            source_h=result['source_factors']['h'][0].cpu().numpy())
        row=dict(spec=spec,identity=identity,manifest=scene['metadata']['manifest'],edges=scene['forest']['edges'],
            target_sha256={str(k):data.array_hash(v) for k,v in scene['targets'].items()},hops={},wire=wire,
            contexts=[dict(sender=i,receiver=j,nonce=nonce.hex()) for (i,j),nonce in contexts.items()],
            saved_packet_stream=str(packets.relative_to(folder)),packet_sha256=sha(packets),packet_lengths=list(map(len,result['packets'])))
        predictions=[];losses={}
        for k in HOPS:
            out=result['decoded'][k];known=out['known_logits'][0].cpu().numpy().reshape(scene['private'].shape)
            wall=out['wall_logits'][0].cpu().numpy().reshape(scene['private'].shape)
            raw=np.where(known>=0,(wall>=0).astype(np.int8),-1).astype(np.int8);predictions.append(raw)
            row['hops'][str(k)]=scene_metrics(scene,raw,k)
            value,terms=stage_loss(out,torch.as_tensor(scene['targets'][k][None],device=device))
            losses[str(k)]=dict(total=float(value),known=float(terms['known']),wall=float(terms['wall']))
            if spec['index']==0:arrays[f'known_logits_K{k}']=known;arrays[f'wall_logits_K{k}']=wall
        arrays['prediction']=np.stack(predictions);save_npz(path,**arrays)
        row.update(arrays_file=str(path.relative_to(folder)),arrays_sha256=sha(path),proper_losses=losses)
        records.append(row)
        if STOP_REQUESTED:raise InterruptedError('Requested stop after current complete evaluation scene')
    if model.weights_hash()!=before:raise AssertionError('Evaluation modified checkpoint')
    report=dict(version=295,arm=arm,model_arm=model.arm,step=step,split='VALIDATION',records=records,
        profiles=aggregate(records),weights_sha256=before,seconds=time.monotonic()-tick,
        actual_wire_audits_passed=True,raw_predictions_not_overwritten=True,native_size=128,
        execution_transport=execution_transport(root),setup_measured=False,full_transcript_measured=False,communication_100x_claim=False,TEST_opened=False)
    write(folder/f'validation-{step:06d}.json',report);return report


def paired_gate(control,candidate):
    expected={scene_id(s) for s in scene_plan('VALIDATION')};checks=[];errors=[]
    for arm,record in zip(ARMS,(control,candidate)):
        rows=record.get('records',[])
        if (record.get('arm')!=arm or record.get('step')!=4000 or record.get('split')!='VALIDATION'
                or record.get('TEST_opened') is not False or record.get('actual_wire_audits_passed') is not True
                or len(rows)!=48 or {r.get('identity') for r in rows}!=expected):errors.append('Incomplete/wrong '+arm+' endpoint')
    if errors:return dict(passed=False,valid=False,errors=errors)
    def metric(record,k,profile,name):
        value=record['profiles'][str(k)][profile]['all']['metrics'].get(name)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError('Missing/ineligible/nonfinite '+profile+'/'+name)
        return value
    def add(name,passed,**values):checks.append(dict(name=name,passed=bool(passed),**values))
    try:
        for rooms in (6,12,20):
            profile=f'rooms{rooms}/allN'
            for record in (control,candidate):
                for key,limit,lower in (('own_typed_accuracy',.90,True),('own_wall_f1',.75,True),('unknown_known_fpr',.05,False)):
                    value=metric(record,0,profile,key);add(f"K0/{record['arm']}/{profile}/{key}",value>=limit if lower else value<=limit,value=value)
            for k in (2,8):
                for key,delta in (('union_accuracy',.03),('peer_wall_f1',.10)):
                    a=metric(control,k,profile,key);b=metric(candidate,k,profile,key)
                    add(f'gain/K{k}/{profile}/{key}',b-a>=delta-1e-12,control=a,candidate=b,delta=b-a)
                coverage=candidate['profiles'][str(k)][profile]['h_ge2' if k==2 else 'h_gt2']['eligible_recipient_counts']
                add(f'coverage/K{k}/{profile}',sum(coverage)>0,recipients=sum(coverage))
            for n in (8,16):
                profile=f'rooms{rooms}/N{n}'
                for k in HOPS:
                    for key,drop in (('own_typed_accuracy',.01),('known_recall',.01),('own_known_recall',.01),('own_wall_precision',.01),
                                     ('own_wall_recall',.01),('own_wall_f1',.01),('door_free_recall',.02),('pair_topology_accuracy',.02)):
                        a=metric(control,k,profile,key);b=metric(candidate,k,profile,key)
                        add(f'protect/K{k}/{profile}/{key}',b>=a-drop-1e-12,control=a,candidate=b)
                        if k and key.startswith('own_'):
                            source=metric(candidate,0,profile,key)
                            add(f'sourcefacts/K{k}/{profile}/{key}',b>=source-.01-1e-12,source=source,candidate=b)
                    a=metric(control,k,profile,'unknown_known_fpr');b=metric(candidate,k,profile,'unknown_known_fpr')
                    add(f'FPR/K{k}/{profile}',b<=.03+1e-12 and b<=a+.005+1e-12,control=a,candidate=b)
    except (KeyError,TypeError,ValueError) as exc:errors.append(str(exc))
    return dict(passed=not errors and all(c['passed'] for c in checks),valid=not errors,errors=errors,checks=checks,
                pilot_only=True,full_goal_complete=False,communication_100x_claim=False)


def prepare(root,config,checkpoint):
    root=Path(root).resolve();p=registration(config);source=root/'source'
    if (root/'registered-protocol.json').exists():raise FileExistsError('Prepared experiment cannot be overwritten')
    if Path(__file__).resolve()!=source/'scripts'/Path(__file__).name:raise RuntimeError('Stage immutable minimal source bundle first, then prepare from it')
    if sha(checkpoint)!=PARENT_SHA:raise ValueError('Wrong parent checkpoint')
    if sha(source/'scripts/mapping_multiroom_data_v293.py')!=p['data']['generator_sha256']:raise ValueError('Original generator changed')
    p=dict(p,checkpoint=str(Path(checkpoint).resolve()),checkpoint_sha256=sha(checkpoint),prepared_root=str(root),
           implementation=dict(source_only_decoded_stages=[0],joint_decoded_stages=list(HOPS),
              equal_compute_claim=False,resume_implemented=False,validation_scene_order='rooms,N,index',
              direct_source_contains_no_labels=True,source_manifest_excludes_runtime_artifacts=True))
    files={str(f.relative_to(source)).replace('\\','/'):sha(f) for f in sorted(source.rglob('*')) if f.is_file() and f.suffix in ('.py','.json')}
    write(root/'registered-protocol.json',p)
    write(root/'source-manifest.json',dict(files=files,version=295,protocol_sha256=sha(root/'registered-protocol.json')))
    draws=draw_plan();save_npz(root/'draw-plan.npz',scene_indices=draws)
    write(root/'draw-plan.json',dict(seed=20261017,updates=4000,scenes=scene_plan('TRAIN'),
        actual_draw_sha256=data.array_hash(draws),npz_sha256=sha(root/'draw-plan.npz'),sampling='Independent uniform scene tuple per update; same realized sequence for both arms'))
    write(root/'status.json',dict(state='prepared_not_launched',version=295,training_updates=0,TEST_opened=False))
    return p


def check(root):
    root=Path(root).resolve();p=registration(root/'registered-protocol.json');source=root/'source'
    if Path(__file__).resolve()!=source/'scripts'/Path(__file__).name:raise RuntimeError('Execute the sealed driver only')
    manifest=read(root/'source-manifest.json')
    if manifest['protocol_sha256']!=sha(root/'registered-protocol.json'):raise ValueError('Registered protocol changed')
    for relative,digest in manifest['files'].items():
        target=(source/relative).resolve()
        if not target.is_relative_to(source) or sha(target)!=digest:raise ValueError('Sealed source changed: '+relative)
    if sha(p['checkpoint'])!=PARENT_SHA:raise ValueError('Pinned parent file changed')
    plan=read(root/'draw-plan.json')
    if sha(root/'draw-plan.npz')!=plan['npz_sha256']:raise ValueError('Draw archive changed')
    with np.load(root/'draw-plan.npz') as saved:draws=saved['scene_indices']
    if not np.array_equal(draws,draw_plan()) or data.array_hash(draws)!=plan['actual_draw_sha256']:raise ValueError('Realized draw plan differs')
    return p


def device_setup(root,device,gpu_uuid):
    p=check(root)
    if torch.device(device).type!='cuda' or gpu_uuid!=p['execution']['gpu_uuid'] or os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu_uuid:
        raise RuntimeError('Explicit registered GPU UUID and CUDA_VISIBLE_DEVICES required; CPU CLI never trains')
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True)
    matching=[r.strip() for r in rows.splitlines() if r.split(',')[0].strip()==gpu_uuid]
    if len(matching)!=1 or '5090' not in matching[0]:raise RuntimeError('Registered RTX5090 unavailable')
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    foreign=[r for r in processes.splitlines() if r.split(',')[0].strip()==gpu_uuid and int(r.split(',')[1])!=os.getpid()]
    if foreign:raise RuntimeError('Other compute task occupies registered GPU; do not kill it')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if torch.cuda.device_count()!=1:raise RuntimeError('One explicitly visible GPU required')
    torch.cuda.set_per_process_memory_fraction(.85,device);torch.manual_seed(20261017)
    return p,dict(device=device,gpu_uuid=gpu_uuid,gpu=matching[0],memory_fraction=.85,torch=torch.__version__,
                  cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'])


def preflight(root,device,gpu_uuid):
    root=Path(root);p,context=device_setup(root,device,gpu_uuid)
    if (root/'preflight-started.json').exists() or (root/'started.json').exists():raise FileExistsError('Do not repeat native admission')
    write(root/'preflight-started.json',dict(pid=os.getpid(),parameter_updates=0))
    bank=SceneBank(root/'data-bank');start=time.monotonic()
    scene=bank.get(dict(split='TRAIN',index=0,rooms=20,agents=16,size=128));generation=time.monotonic()-start
    own,targets=tensors(scene,device);arms={};shared=None
    for arm in ARMS:
        model=load_model(p['checkpoint'],device);before=model.weights_hash();unused=model.frozen_unused_hash();buffers=state_buffers(model)
        if shared is not None and shared!=before:raise AssertionError('Two arms do not share exact initialization')
        shared=before;torch.cuda.reset_peak_memory_stats();timings=[]
        for trial in range(3):
            model.zero_grad(set_to_none=True);_sync(device);tick=time.monotonic()
            result=train_rollout(model,own,scene['forest']['edges'],arm);loss,parts=objective(result,targets,arm)
            if trial==0 and arm=='causal_joint':fused=gradient_groups(model,stage_loss(result['decoded'][8],targets[8])[0])
            loss.backward();_sync(device)
            if trial:timings.append(time.monotonic()-tick)
        gradients=gradient_groups(model)
        del result,loss,parts;model.zero_grad(set_to_none=True);_sync(device);tick=time.monotonic()
        actual,wire,contexts=actual_rollout(model,own,scene['forest']['edges'])
        # Complete one-scene evaluation cost includes CPU raw topology and literal packet replay.
        archived=dict(own=scene['private']);predictions=[]
        for k in HOPS:
            out=actual['decoded'][k];known=out['known_logits'][0].cpu().numpy().reshape(scene['private'].shape)
            wall=out['wall_logits'][0].cpu().numpy().reshape(scene['private'].shape)
            prediction=np.where(known>=0,(wall>=0).astype(np.int8),-1).astype(np.int8)
            predictions.append(prediction);scene_metrics(scene,prediction,k)
            archived[f'known_logits_K{k}']=known;archived[f'wall_logits_K{k}']=wall
        archived['prediction']=np.stack(predictions)
        save_npz(root/f'preflight-{arm}-TRAIN-scene.npz',**archived)
        _sync(device);evaluation_seconds=time.monotonic()-tick
        with torch.no_grad():
            fast=core.rollout(model,own,scene['forest']['edges'],rounds=8,decode_rounds=HOPS,trace=False)
            for key in ('codes','means','variances','features'):
                if any(not torch.equal(a,b) for a,b in zip(fast[key],actual[key])):raise AssertionError('Serialized forward differs: '+key)
            for k in HOPS:
                if any(not torch.equal(fast['decoded'][k][key],actual['decoded'][k][key]) for key in ('known_logits','wall_logits')):raise AssertionError('Serialized map differs')
            floating=core.rollout(model,own,scene['forest']['edges'],rounds=8,decode_rounds=(),quantized=False,trace=False)
            for k in range(9):
                reference=core.central_reference(floating['source_factors']['q'],floating['source_factors']['h'],scene['forest']['edges'],k)
                torch.testing.assert_close(floating['means'][k],reference['mean'],atol=1e-11,rtol=1e-11)
                torch.testing.assert_close(floating['variances'][k],reference['variance'],atol=1e-11,rtol=1e-11)
            if arm=='source_only':
                from scripts import mapping_common_map_v284 as previous
                # Native old protocol only supports N2/3; this disclosed two-source
                # slice is a codec diagnostic, not a new training/evaluation arm.
                pair_edges=[(0,1,1.)]
                old=previous.rollout(model,own[:,:2],pair_edges,mode='equality',rounds=2,precision_profile=3,trace=False)
                new=core.rollout(model,own[:,:2],pair_edges,rounds=2,decode_rounds=(0,1,2),trace=False)
                if not torch.equal(old['source_factors']['q'],new['source_factors']['q']) or not torch.equal(old['source_factors']['h'],new['source_factors']['h']):raise AssertionError('Profile diagnostic changed source')
                if not torch.equal(old['codes'][0],new['codes'][0]) or any(not torch.equal(old['decoded'][0][key],new['decoded'][0][key]) for key in ('known_logits','wall_logits')):raise AssertionError('Profile diagnostic changed K0 function')
                codec_shift=dict(scope='Two original sources from same TRAIN scene; old core is N2/3 only',
                    old_profile=3,new_profile=4,source_exact=True,K0_codes_and_heads_exact=True,
                    code_changed_fraction=float((old['codes'][2]!=new['codes'][2]).float().mean()),
                    mean_abs_difference=float((old['means'][2]-new['means'][2]).abs().max()),
                    not_quality_admission=True,no_extra_training_arm=True)
                del old,new
        assert_buffers(model,buffers)
        if model.weights_hash()!=before or model.frozen_unused_hash()!=unused:raise AssertionError('Zero-update admission changed model')
        update_seconds=max(timings);estimate=4000*update_seconds+4*48*evaluation_seconds
        arms[arm]=dict(initial_state_sha256=before,frozen_unused_sha256=unused,gradient_groups=gradients,
            fused_only_gradient_groups=fused if arm=='causal_joint' else None,seconds_per_update=update_seconds,
            one_scene_evaluation_seconds=evaluation_seconds,estimated_training_plus_validation_seconds=estimate,
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),wire=wire,passed=estimate<6000,
            float_reached_subset_parity=True,serialized_tensor_exact=True,weights_unchanged=True,
            trainable_parameters=sum(v.numel() for v in model.trainable_parameters()),total_parameters=sum(v.numel() for v in model.parameters()))
        del model,actual,fast,floating;gc.collect();torch.cuda.empty_cache()
    report=dict(version=295,passed=all(v['passed'] for v in arms.values()),arms=arms,context=context,
        scene=scene['metadata']['spec'],generation_seconds=generation,estimated_1536_scene_generation_seconds=1536*generation,
        data_generation_excluded_from_training_cap=True,training_estimate_includes_all_registered_evaluations=True,
        estimate_not_guarantee=True,timing_omissions=['AdamW.step and gradient clipping','per-update cache I/O and target assembly','checkpoint writes and JSON bookkeeping'],
        parameter_updates=0,optimizer_updates=0,weights_unchanged=True,VALIDATION_opened=False,TEST_opened=False,
        execution_transport=execution_transport(root),old_to_new_codec_zero_update=codec_shift,
        protocol_sha256=sha(root/'registered-protocol.json'),source_manifest_sha256=sha(root/'source-manifest.json'))
    write(root/'preflight.json',report);write(root/'status.json',dict(state='preflight_passed' if report['passed'] else 'resource_not_admitted',training_updates=0))
    return report


def require_preflight(root):
    root=Path(root);r=read(root/'preflight.json')
    if (r.get('passed') is not True or r.get('parameter_updates')!=0 or r.get('weights_unchanged') is not True
            or r.get('VALIDATION_opened') is not False or r.get('TEST_opened') is not False
            or r.get('protocol_sha256')!=sha(root/'registered-protocol.json')
            or r.get('source_manifest_sha256')!=sha(root/'source-manifest.json') or set(r.get('arms',{}))!=set(ARMS)
            or any(not a['passed'] or a['initial_state_sha256']!=PARENT_STATE for a in r['arms'].values())):
        raise RuntimeError('Passing bound-to-source zero-update native preflight required')
    return r


def save_checkpoint(folder,model,opt,step,arm,p,draw_digest,name=None):
    path=Path(folder)/(name or f'validated-{step:06d}.pt');temporary=path.with_suffix('.tmp')
    payload=dict(construction=model.construction_metadata(),state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
        model_sha256=model.weights_hash(),optimizer=opt.state_dict(),step=step,training_arm=arm,arm=model.arm,
        protocol=p,parent_sha256=PARENT_SHA,actual_draw_sha256=draw_digest,frozen_unused_sha256=model.frozen_unused_hash(),
        execution_transport=execution_transport(Path(folder).parents[1]),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),resume_implemented=False,TEST_opened=False)
    torch.save(payload,temporary);os.replace(temporary,path)
    return dict(path=str(path),sha256=sha(path),model_sha256=payload['model_sha256'],step=step)


def _stop(signum,frame):
    global STOP_REQUESTED
    STOP_REQUESTED=True


def run(root,device,gpu_uuid):
    root=Path(root);p,context=device_setup(root,device,gpu_uuid);require_preflight(root)
    if (root/'started.json').exists():raise FileExistsError('This run already started; no implicit resume or duplicate launch')
    write(root/'started.json',dict(pid=os.getpid(),context=context,parameter_updates=0))
    signal.signal(signal.SIGTERM,_stop);signal.signal(signal.SIGINT,_stop)
    started=time.monotonic();bank=SceneBank(root/'data-bank')
    plans=scene_plan('TRAIN');draws=draw_plan();arms={};model=opt=None;completed=0;arm=None
    try:
        bank.build_train(root/'status.json')
        # Original sensor generation is measured separately, outside either arm cap.
        for spec in scene_plan('VALIDATION'):
            if STOP_REQUESTED:raise InterruptedError('Stop requested before training')
            bank.get(spec)
        write(root/'data-bank/VALIDATION-manifest.json',dict(scenes={k:v for k,v in bank.entries.items() if v['spec']['split']=='VALIDATION'},TEST_opened=False))
        for arm in ARMS:
            completed=0;torch.manual_seed(20261017);model=load_model(p['checkpoint'],device)
            buffers=state_buffers(model);unused=model.frozen_unused_hash();initial=model.weights_hash()
            folder=root/'outputs'/arm;folder.mkdir(parents=True,exist_ok=False)
            opt=torch.optim.AdamW(model.trainable_parameters(),lr=5e-5,weight_decay=0.)
            curves=[];actual_draws=[];tick=time.monotonic();train_seconds=0.;loss_log=[]
            torch.cuda.reset_peak_memory_stats()
            initial_report=evaluate(model,root,bank,arm,0,device);curves.append(dict(step=0,profiles=initial_report['profiles']))
            save_checkpoint(folder,model,opt,0,arm,p,data.array_hash(np.array([],dtype=np.int64)))
            for step,scene_number in enumerate(draws):
                if STOP_REQUESTED:raise InterruptedError('User/system signal requested safe stop')
                if time.monotonic()-tick>=6000:raise TimeoutError('Registered resource cap; incomplete, not convergence')
                scene=bank.get(plans[int(scene_number)]);own,targets=tensors(scene,device)
                for group in opt.param_groups:group['lr']=scheduled_lr(step)
                model.eval();model.zero_grad(set_to_none=True);_sync(device);update_tick=time.monotonic()
                result=train_rollout(model,own,scene['forest']['edges'],arm);loss,parts=objective(result,targets,arm)
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),5.,error_if_nonfinite=True)
                opt.step();_sync(device);train_seconds+=time.monotonic()-update_tick
                completed=step+1;actual_draws.append(int(scene_number))
                if completed%50==0:
                    entry=dict(step=completed,scene=scene_id(plans[int(scene_number)]),loss=float(loss.detach()),
                        parts={k:float(v.detach()) for k,v in parts.items()},lr=scheduled_lr(step),gradient_norm=float(norm),
                        train_seconds=train_seconds,arm_wall_seconds=time.monotonic()-tick)
                    loss_log.append(entry);write(folder/'training-curve.json',loss_log)
                    write(root/'status.json',dict(state='training',arm=arm,completed_updates=completed,**{k:v for k,v in entry.items() if k!='step'}))
                del result,loss,parts,own,targets
                if completed in STEPS[1:]:
                    assert_buffers(model,buffers)
                    if model.frozen_unused_hash()!=unused:raise AssertionError('Unused source decoder/lift changed')
                    validation=evaluate(model,root,bank,arm,completed,device);curves.append(dict(step=completed,profiles=validation['profiles']))
                    checkpoint=save_checkpoint(folder,model,opt,completed,arm,p,data.array_hash(np.asarray(actual_draws,dtype=np.int64)))
                    write(folder/'curves.json',curves)
            digest=data.array_hash(np.asarray(actual_draws,dtype=np.int64))
            if digest!=data.array_hash(draws):raise AssertionError('Actual TRAIN sequence diverged')
            arms[arm]=dict(completed_updates=completed,initial_state_sha256=initial,actual_draw_sha256=digest,
                training_seconds=train_seconds,wall_seconds=time.monotonic()-tick,peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                endpoint=checkpoint,curves=curves,frozen_unused_sha256=unused,
                trainable_parameters=sum(v.numel() for v in model.trainable_parameters()),total_parameters=sum(v.numel() for v in model.parameters()))
            write(folder/'result.json',arms[arm]);del model,opt;model=opt=None;gc.collect();torch.cuda.empty_cache()
        control=read(root/'outputs/source_only/validation-004000.json');candidate=read(root/'outputs/causal_joint/validation-004000.json')
        report=dict(version=295,state='complete_multiroom_domain_pair',arms=arms,paired_gate=paired_gate(control,candidate),
            elapsed_seconds=time.monotonic()-started,original_bank_generation_seconds=bank.generation_seconds,
            data_manifest_sha256=sha(root/'data-bank/TRAIN-manifest.json'),context=context,
            protocol_sha256=sha(root/'registered-protocol.json'),source_manifest_sha256=sha(root/'source-manifest.json'),
            inherited_parent_file_sha256=PARENT_SHA,same_initial_function=True,same_actual_draws=True,
            no_equal_FLOP_claim=True,native_neural_sizes=[128],native256_512_supported=False,
            execution_transport=execution_transport(root),whole_transcript_compression_claim=False,full_goal_complete=False,TEST_opened=False)
        check(root);write(root/'report.json',report);write(root/'status.json',dict(state=report['state'],completed_updates_per_arm=4000,full_goal_complete=False))
        return report
    except BaseException as exc:
        recovery=None
        if model is not None and opt is not None:
            recovery=save_checkpoint(root/'outputs'/arm,model,opt,completed,arm,p,
                data.array_hash(np.asarray(actual_draws,dtype=np.int64)),name=f'recovery-{completed:06d}.pt')
        write(root/'status.json',dict(state='resource_incomplete' if isinstance(exc,TimeoutError) else 'interrupted' if isinstance(exc,InterruptedError) else 'failed',
            arm=arm,completed_updates=completed,recovery=recovery,error=traceback.format_exc(),resume_implemented=False,full_goal_complete=False))
        raise


def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','preflight','run'):mode.add_argument('--'+name,action='store_true')
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--config',type=Path)
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--device',default='cuda:0');parser.add_argument('--gpu-uuid')
    args=parser.parse_args()
    if args.prepare:
        if args.config is None or args.checkpoint is None:parser.error('prepare needs explicit config and pinned checkpoint')
        result=prepare(args.out,args.config,args.checkpoint)
    elif args.preflight:
        try:result=preflight(args.out,args.device,args.gpu_uuid)
        except BaseException:
            write(args.out/'status.json',dict(state='failed_preflight',parameter_updates=0,error=traceback.format_exc()));raise
    else:result=run(args.out,args.device,args.gpu_uuid)
    print(json.dumps({k:v for k,v in result.items() if k in ('version','state','passed','paired_gate','elapsed_seconds')},allow_nan=False),flush=True)


if __name__=='__main__':main()
