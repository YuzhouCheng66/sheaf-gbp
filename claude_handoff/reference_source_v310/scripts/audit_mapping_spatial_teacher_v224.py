"""Frozen legacy1024 SOURCE audit on the current64D experiment's worlds.

Before compressing or distilling a high-dimensional representation, check that
its actual decoder is stronger on THIS distribution. Native legacy shape cap
is128, never silently resize512 and call it native. No training or GBP here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from scripts import train_mapping_implicit64_v192 as base
from scripts.mapping_pointwise_capacity_v128 import PointwiseCapacityCodec,PointwiseCapacityConfig

TEACHER=Path('/home/yuzhou/Desktop/sheaf-gbp/mapping-resume-c1024-visible-20261016')
CHECKPOINT=TEACHER/'runs/c1024/validated-165000.pt'
CHECKPOINT_SHA='882429e5ec23a14aa829a7b02d3f45ff3503f1a001967cb148778c3bf36b3c52'
CURRENT=Path('/home/yuzhou/Desktop/sheaf-gbp/mapping-observed-distance-v223-20261016')
FILE='scripts/audit_mapping_spatial_teacher_v224.py'


def load_teacher(path,device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['arm']!='c1024':raise ValueError('Require actual1024 control')
    model=PointwiseCapacityCodec(PointwiseCapacityConfig(planes=4,seed=int(saved['config']['seed'])))
    model.load_state_dict(saved['model_state'],strict=True)
    model=model.to(device).eval()
    for value in model.parameters():value.requires_grad_(False)
    return model,dict(arm=saved['arm'],counters=saved.get('counters'),config=saved['config'],parameters=model.counts())


def state_hash(model):
    digest=hashlib.sha256()
    for key,value in model.state_dict().items():digest.update(key.encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def make_positions(private):
    # Public role is not the target map: use an actually observed center-nearest
    # cell, or fixedcenterifempty. Agentmarker includedinoriginalencoder only.
    size=private.shape[-1];yy,xx=np.meshgrid(np.arange(size),np.arange(size),indexing='ij')
    distance=(yy-size/2)**2+(xx-size/2)**2;positions=[]
    for source in private:
        eligible=np.where(source>=0,distance,np.inf)
        index=int(np.argmin(eligible)) if np.any(source>=0) else (size//2)*size+size//2
        positions.append((index//size,index%size))
    return np.asarray(positions,dtype=np.int64)


def prepare(root):
    if root.exists():raise FileExistsError('No overwrite')
    if base.read(CURRENT/'status.json')['state']!='complete_observed_geometry_pair':raise RuntimeError('Currentstageincomplete')
    manifest=base.read(TEACHER/'source-manifest-v191.json')
    for rel,digest in manifest['files'].items():
        if base.sha(TEACHER/'source'/rel)!=digest:raise RuntimeError('Legacysealedsourcechanged:'+rel)
    if base.sha(CHECKPOINT)!=CHECKPOINT_SHA:raise RuntimeError('Protectedcheckpointchanged')
    root.mkdir();shutil.copytree(TEACHER/'source',root/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in (FILE,'scripts/train_mapping_implicit64_v192.py','scripts/mapping_implicit_model_v192.py','scripts/mapping_implicit_data_v192.py'):
        dest=root/'source'/name
        if dest.exists():raise FileExistsError('Do notreplacelegacydependency:'+name)
        shutil.copy2(Path(__file__).resolve().parents[1]/name,dest)
    shutil.copy2(CURRENT/'preserved-prior.json',root/'preserved-prior.json')
    p=dict(version=224,gpu_uuid=base.GPU,checkpoint=str(CHECKPOINT),checkpoint_sha256=CHECKPOINT_SHA,
           sizes=[128],complexities=['base','medium'],splits=['TRAIN','VALIDATION'],indices=list(range(12)),
           source_modes=['union_oracle','A','B'],quantization_modes=['continuous','8bit'],weight_updates=0,
           interpretation='Sourceautoencoding only,notGBPfusedor64Dresult. Both1024meansAND1024precisions observed-onlyspatialfields inputdecoder,2048scalars; notafairbandwidthcompetitor. Hugeinheritedtrainingbudgetdisclosed.',
           purpose='Testwhetherprotectedhighdimensionalcodecisasuitablegeometryteacher onCURRENTsyntheticworldsbefore trainingcompressedbottleneck. Ifnotbetter, do notassumestrongfromolddistributionmetrics.',
           thresholds='Reportnative3classternaryargmaxANDcurrentknown.5/conditionalwall.5hierarchicalgate; neverselectindividually.',
           position_rule='Chooseknowncellnearestmapcenterfromsource; thisisencoder-onlypositionmarker,nohiddenoccupancyorpeerinformation. Samepositionacrossquantized/continuous. Allpositionsrecorded.',
           scaling='Legacypublic_geometrycapsH,W<=128 anddecoderends128. No512nativeclaimorunregisteredresizing.',
           current_control_report=str(CURRENT/'report.json'),current_control_sha256=base.sha(CURRENT/'report.json'),
           test_opened=False,full_goal_complete=False,automatic_followup_training=False)
    base.write(root/'registered-protocol.json',p)
    files={v.relative_to(root/'source').as_posix():base.sha(v) for v in (root/'source').rglob('*') if v.is_file()}
    base.write(root/'source-manifest.json',dict(files=files,protocol_sha256=base.sha(root/'registered-protocol.json'),
                                             preserved_sha256=base.sha(root/'preserved-prior.json')))
    print(json.dumps(dict(prepared=str(root),legacy_checkpoint_unchanged=True,native_size=128)),flush=True)


@torch.no_grad()
def run(root):
    p=base.check(root)
    if (root/'started.json').exists():raise FileExistsError('No rerun')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=base.GPU:raise RuntimeError('WrongGPU')
    base.guard();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if base.sha(p['checkpoint'])!=p['checkpoint_sha256']:raise RuntimeError('Checkpointchanged')
    base.write(root/'started.json',dict(pid=os.getpid(),time=time.time()));rows=[];started=time.monotonic()
    try:
        model,metadata=load_teacher(p['checkpoint'],'cuda');frozen=state_hash(model);base.write(root/'teacher.json',metadata)
        for split in p['splits']:
            for complexity in p['complexities']:
                for index in p['indices']:
                    base.guard();sample=base.data.sample_world(split,index,128,complexity)
                    private=np.stack((sample['union'],*sample['private']));positions=make_positions(private)
                    q,h=model.encode(torch.as_tensor(private[None],device='cuda'),torch.as_tensor(positions[None],device='cuda'))
                    qc,mc,wq,wh=model.quantizer.encode(q,h)
                    rq,rh=model.quantizer.decode(qc,mc)
                    torch.testing.assert_close(wq,rq,rtol=0,atol=0);torch.testing.assert_close(wh,rh,rtol=0,atol=0)
                    for mode,qv,hv in (('continuous',q,h),('8bit',rq,rh)):
                        logp=model.decode(qv,hv,128,128)[0].float()
                        native=logp.argmax(-1).cpu().numpy().astype(np.int8)-1
                        known=logp[...,1:].logsumexp(-1).exp()
                        conditional_wall=(logp[...,2]-logp[...,1:].logsumexp(-1)).exp()
                        gated=torch.where(known>=.5,(conditional_wall>=.5).to(torch.int8),-1).cpu().numpy()
                        for i,source in enumerate(p['source_modes']):
                            target=private[i];ms={}
                            for rule,pred in (('native_argmax',native[i]),('hierarchical',gated[i])):
                                m=base.measures(pred,target);m.update(base.door_measures(pred,dict(sample,union=target)));ms[rule]=m
                            rows.append(dict(split=split,complexity=complexity,index=index,size=128,source=source,
                                             quantization=mode,position=positions[i].tolist(),metrics=ms))
                        if split=='VALIDATION' and index in (0,2) and mode=='8bit':
                            np.savez_compressed(root/f'{complexity}-{index}-source.npz',target=private,native=native,hierarchical=gated,
                                                means=(hv/qv.clamp_min(1e-12)).cpu().numpy(),precision=qv.cpu().numpy())
                    base.write(root/'status.json',dict(state='frozen_legacy_source_audit',completed_worlds=len(rows)//6,total_worlds=48,
                                                      elapsed_s=time.monotonic()-started,test_opened=False))
        if state_hash(model)!=frozen or base.sha(p['checkpoint'])!=p['checkpoint_sha256']:raise AssertionError('Protectedmodelchanged')
        profiles={}
        for split in p['splits']:
            for complexity in p['complexities']:
                for mode in p['quantization_modes']:
                    for source_group in ('union_oracle','own'):
                        selected=[r for r in rows if r['split']==split and r['complexity']==complexity and r['quantization']==mode
                                  and (r['source']=='union_oracle' if source_group=='union_oracle' else r['source']!='union_oracle')]
                        for rule in ('native_argmax','hierarchical'):
                            profiles[f'{split}/{complexity}/{source_group}/{mode}/{rule}']=base.average([r['metrics'][rule] for r in selected])
        base.write(root/'report.json',dict(records=rows,profiles=profiles,teacher=metadata,weights_unchanged=True,
                                          elapsed_s=time.monotonic()-started,test_opened=False,full_goal_complete=False))
        base.write(root/'status.json',dict(state='complete_legacy_source_transfer_audit',report='report.json',test_opened=False))
        print(json.dumps(dict(state='complete',profiles=profiles)),flush=True)
    except BaseException:
        base.write(root/'status.json',dict(state='failed_legacy_source_audit',error=traceback.format_exc(),completed_records=len(rows),
                                         checkpoint_sha256_after=base.sha(p['checkpoint'])));raise


def launch(root):
    base.check(root);base.guard()
    if (root/'launch.json').exists():raise FileExistsError('Alreadylaunched')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=base.GPU,PYTHONPATH='.:exploration_grid',PYTHONDONTWRITEBYTECODE='1',
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (root/'audit.log').open('x') as stream:
        child=subprocess.Popen([sys.executable,'-u',str(root/'source'/FILE),'--root',str(root)],cwd=root/'source',env=env,
                               stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    base.write(root/'launch.json',dict(pid=child.pid,time=time.time()));print(json.dumps(dict(pid=child.pid,root=str(root))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--launch',action='store_true');args=parser.parse_args();root=Path(args.root).resolve()
    if args.prepare:prepare(root)
    elif args.launch:launch(root)
    else:run(root)
