"""Frozen spatial teacher -> PCA64 -> trainable nonlinear64 residual.

This finite diagnostic trains only a compressed feature autoencoder. It does
not claim that teacher features are ground truth, that the teacher is perfect,
or that a source codec implements GBP. Native 128 only in this stage.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import unittest
import numpy as np
import torch
from scripts import train_mapping_implicit64_v192 as base
from scripts.audit_mapping_spatial_teacher_v224 import load_teacher,state_hash,make_positions
from scripts.mapping_spatial_bottleneck_v225 import fit_pca,SpatialBottleneck,pack_features,unpack_features
from scripts.mapping_implicit_model_v192 import latent_quantize,latent_codes,codes_to_latent

PARENT=Path('/home/yuzhou/Desktop/sheaf-gbp/mapping-legacy-transfer-audit-v224-20261016')
CURRENT=Path('/home/yuzhou/Desktop/sheaf-gbp/mapping-observed-distance-v223-20261016')
FILE='scripts/train_mapping_spatial_bottleneck_v225.py'
FILES=[FILE,'scripts/mapping_spatial_bottleneck_v225.py','scripts/test_mapping_spatial_bottleneck_v225.py']


def prepare(root):
    if root.exists():raise FileExistsError('No overwrite')
    if base.read(PARENT/'status.json')['state']!='complete_legacy_source_transfer_audit':raise RuntimeError('Audit incomplete')
    manifest=base.read(PARENT/'source-manifest.json')
    for rel,digest in manifest['files'].items():
        if base.sha(PARENT/'source'/rel)!=digest:raise RuntimeError('Changed dependency:'+rel)
    old=base.read(PARENT/'registered-protocol.json')
    if base.sha(old['checkpoint'])!=old['checkpoint_sha256']:raise RuntimeError('Teacher changed')
    root.mkdir();shutil.copytree(PARENT/'source',root/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for rel in FILES:shutil.copy2(Path(__file__).resolve().parents[1]/rel,root/'source'/rel)
    shutil.copy2(PARENT/'preserved-prior.json',root/'preserved-prior.json')
    prior=base.read(CURRENT/'registered-protocol.json')
    p=dict(version=225,gpu_uuid=base.GPU,checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],
        seed=225101,latent_dim=64,updates=4000,batch_size=128,learning_rate=.0002,end_learning_rate=.00002,
        validation_steps=[0,1000,2000,4000],train_worlds_per_complexity=prior['train_worlds_per_complexity'],
        train_size=128,complexities=prior['complexities'],validation_worlds_per_complexity=12,
        input='Only frozen legacy source mean1024 + log1p precision1024; raw source -> protected teacher encoder -> bottleneck. No private features or covariance bypass decoder.',
        objective='TRAIN standardized source-feature MSE with 8-bit STE. All 6144 union/A/B views weighted equally. No ground-truth pixel or distance supervision in this phase. This tests compressibility of teacher representation, not optimized map fidelity.',
        initialization='TRAIN-only PCA64. 2048->256->64 encoder residual and64->256->2048 decoder residual zero-output initialized around PCA. Standardization floor.05 and per-code bound1.05*TRAIN maxscore fixed shared model data. Same frozen legacy spatial decoder throughout.',
        controls='Frozen PCA64 at step0, continuous and 8bit at every validation; old uncompressed1024 source audit pinned. No claim of equal inherited training budget or bandwidth to the old2048-scalar codec.',
        selection='Single checkpoint by mean128 base/medium own/union accuracy-wallrecall-wallprecision score minus unknown falseknown. Never choose per-case or per-map size.',
        scope='128 native only; no resizing512, noGBP, no100x whole-timelineclaim. Unknown learned source-support reconstruction, not provenance certification.',
        train_bank_sha256=base.read(CURRENT/'data-manifest.json')['sha256'],
        frozen_teacher_audit=str(PARENT/'report.json'),frozen_teacher_report_sha256=base.sha(PARENT/'report.json'),
        max_seconds=1800,automatic_followup_training=False,test_opened=False,full_goal_complete=False)
    base.write(root/'registered-protocol.json',p)
    files={v.relative_to(root/'source').as_posix():base.sha(v) for v in (root/'source').rglob('*') if v.is_file()}
    base.write(root/'source-manifest.json',dict(files=files,protocol_sha256=base.sha(root/'registered-protocol.json'),
                                             preserved_sha256=base.sha(root/'preserved-prior.json')))
    print(json.dumps(dict(prepared=str(root),updates=4000)),flush=True)


@torch.no_grad()
def source_features(teacher,private):
    positions=make_positions(private)
    q,h=teacher.encode(torch.as_tensor(private[None],device='cuda'),torch.as_tensor(positions[None],device='cuda'))
    return pack_features(q,h)[0]


@torch.no_grad()
def make_features(teacher,p,root):
    bank,digest=base.make_bank(p)
    if digest!=p['train_bank_sha256']:raise AssertionError('TRAIN worlds changed')
    features=[];ids=[];started=time.monotonic()
    for i,item in enumerate(bank):
        world_id=item['world_id']
        if not world_id.lower().startswith('train'):raise AssertionError('TRAIN-only')
        views=np.stack((item['union'],*item['private']))
        features.append(source_features(teacher,views));ids.extend([world_id]*3)
        if i%200==0:base.write(root/'status.json',dict(state='precompute_train_features',worlds=i,total=len(bank)))
    data=torch.cat(features);del features,bank;gc.collect()
    feature_digest=hashlib.sha256(data.cpu().numpy().tobytes()).hexdigest()
    base.write(root/'data-manifest.json',dict(sha256=digest,feature_sha256=feature_digest,worlds=len(ids)//3,
                                          views=len(ids),shape=list(data.shape),elapsed_s=time.monotonic()-started,test_opened=False))
    return data,ids


@torch.no_grad()
def evaluate(model,teacher,p,folder=None):
    model.eval();rows=[];profiles={};mse=[]
    for complexity in p['complexities']:
        for index in range(p['validation_worlds_per_complexity']):
            sample=base.data.sample_world('VALIDATION',index,128,complexity)
            private=np.stack((sample['union'],*sample['private']));original=source_features(teacher,private)
            z=model.encode(original);codes=model.to_codes(original)
            hard=codes_to_latent(codes)
            torch.testing.assert_close(hard,latent_quantize(z),rtol=0,atol=0)
            for mode,value in (('continuous',z),('8bit',hard)):
                recovered=model.decode(value)
                q,h=unpack_features(recovered)
                logp=teacher.decode(q[None],h[None],128,128)[0]
                known=logp[...,1:].logsumexp(-1).exp();wall=(logp[...,2]-logp[...,1:].logsumexp(-1)).exp()
                pred=torch.where(known>=.5,(wall>=.5).to(torch.int8),-1).cpu().numpy()
                feature_mse=float((model.normalize(recovered)-model.normalize(original)).square().mean())
                mse.append(dict(complexity=complexity,index=index,quantization=mode,mse=feature_mse))
                for source,target,prediction in zip(('union_oracle','A','B'),private,pred):
                    metrics=base.measures(prediction,target);metrics.update(base.door_measures(prediction,dict(sample,union=target)))
                    rows.append(dict(complexity=complexity,index=index,source=source,quantization=mode,metrics=metrics))
                if folder is not None and index in (0,2) and mode=='8bit':
                    folder.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(folder/f'{complexity}-{index}.npz',target=private,prediction=pred,codes=codes.cpu().numpy())
        for mode in ('continuous','8bit'):
            for group in ('union_oracle','own'):
                selected=[r['metrics'] for r in rows if r['complexity']==complexity and r['quantization']==mode and
                          (r['source']=='union_oracle' if group=='union_oracle' else r['source']!='union_oracle')]
                profiles[f'128/{complexity}/{group}/{mode}']=base.average(selected)
    selection={key:value for key,value in profiles.items() if key.endswith('/8bit')}
    model.train();return dict(profiles=profiles,quality=base.score(selection),records=rows,feature_mse=mse,test_opened=False)


def save(path,model,opt,rng,step,p,digest):
    value=dict(model_type='spatial_bottleneck_v225',model=model.state_dict(),optimizer=opt.state_dict(),
        additional_updates=step,protocol=p,draw_digest=digest,numpy_rng=rng.bit_generator.state,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),test_opened=False)
    temp=path.with_suffix('.tmp');torch.save(value,temp);os.replace(temp,path)


def run(root):
    p=base.check(root);base.guard()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=base.GPU:raise RuntimeError('WrongGPU')
    if (root/'started.json').exists():raise FileExistsError('No implicit restart')
    suite=unittest.defaultTestLoader.loadTestsFromName('scripts.test_mapping_spatial_bottleneck_v225')
    if not unittest.TextTestRunner().run(suite).wasSuccessful():raise RuntimeError('Tests failed')
    torch.set_num_threads(4);torch.manual_seed(p['seed']);torch.cuda.manual_seed_all(p['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base.write(root/'started.json',dict(pid=os.getpid(),time=time.time()));start=time.monotonic();step=0;model=opt=None
    try:
        teacher,metadata=load_teacher(p['checkpoint'],'cuda');teacher_hash=state_hash(teacher)
        features,ids=make_features(teacher,p,root)
        stats,pca=fit_pca(features,ids,p['latent_dim']);model=SpatialBottleneck({k:v.cpu() for k,v in stats.items()}).to('cuda')
        base.write(root/'model.json',dict(parameters=sum(v.numel() for v in model.parameters()),shared_statistic_values=sum(v.numel() for v in model.buffers()),
            payload_values=64,payload_bytes=64,private_skip=False,teacher=metadata,pca=pca))
        opt=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=0.)
        rng=np.random.default_rng(p['seed']+1);digest=hashlib.sha256();curves=[];best=None
        def validate(at):
            nonlocal best
            result=evaluate(model,teacher,p,root/f'examples-{at:06d}')
            checkpoint=root/f'validated-{at:06d}.pt';save(checkpoint,model,opt,rng,at,p,digest.hexdigest())
            row=dict(path=str(checkpoint),sha256=base.sha(checkpoint),additional_updates=at,quality=result['quality'])
            if best is None or row['quality']>best['quality']:best=row
            base.write(root/'selection.json',dict(best=best));base.write(root/f'validation-{at:06d}.json',result)
            curves.append(dict(step=at,quality=result['quality'],profiles=result['profiles'],feature_mse=result['feature_mse']))
            print(json.dumps(dict(validation_step=at,quality=result['quality'],profiles=result['profiles'])),flush=True)
        validate(0)
        for step in range(1,p['updates']+1):
            if time.monotonic()-start>p['max_seconds']:raise TimeoutError('Resource limit, not convergence')
            index=rng.integers(len(features),size=p['batch_size']);digest.update(index.astype('<i8').tobytes())
            batch=features[torch.as_tensor(index,device='cuda')]
            lr=p['end_learning_rate']+.5*(p['learning_rate']-p['end_learning_rate'])*(1+math.cos(math.pi*(step-1)/p['updates']))
            for group in opt.param_groups:group['lr']=lr
            opt.zero_grad(set_to_none=True)
            z=latent_quantize(model.encode(batch));recovered=model.decode_normalized(z)
            loss=(recovered-model.normalize(batch)).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not torch.isfinite(norm):raise FloatingPointError('Gradient')
            opt.step()
            if step%100==0 or step==1:
                row=dict(state='training_spatial_bottleneck',step=step,loss=float(loss.detach()),lr=lr,elapsed_s=time.monotonic()-start)
                base.write(root/'status.json',row)
                with (root/'train.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            if step in p['validation_steps']:base.guard();validate(step)
        if state_hash(teacher)!=teacher_hash or base.sha(p['checkpoint'])!=p['checkpoint_sha256']:raise AssertionError('Protected teacher changed')
        base.write(root/'report.json',dict(best=best,curves=curves,draw_digest=digest.hexdigest(),completed_updates=step,
            frozen_teacher_unchanged=True,elapsed_s=time.monotonic()-start,full_goal_complete=False,test_opened=False,
            interpretation='Source feature compression only, not ground-truth map-trained, not neural GBP or100xwhole-timeline success.'))
        base.write(root/'status.json',dict(state='complete_spatial_bottleneck',report='report.json',test_opened=False))
    except BaseException:
        if model is not None and opt is not None:save(root/'recovery.pt',model,opt,rng,step,p,digest.hexdigest())
        base.write(root/'status.json',dict(state='failed_spatial_bottleneck',step=step,error=traceback.format_exc(),test_opened=False));raise


def launch(root):
    base.check(root);base.guard()
    if (root/'launch.json').exists():raise FileExistsError('Already launched')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=base.GPU,PYTHONPATH='.:exploration_grid',PYTHONDONTWRITEBYTECODE='1',
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (root/'train.log').open('x') as stream:
        child=subprocess.Popen([sys.executable,'-u',str(root/'source'/FILE),'--root',str(root)],cwd=root/'source',env=env,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    base.write(root/'launch.json',dict(pid=child.pid,time=time.time()));print(json.dumps(dict(pid=child.pid,root=str(root))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--prepare',action='store_true');parser.add_argument('--launch',action='store_true')
    args=parser.parse_args();root=Path(args.root).resolve()
    if args.prepare:prepare(root)
    elif args.launch:launch(root)
    else:run(root)
