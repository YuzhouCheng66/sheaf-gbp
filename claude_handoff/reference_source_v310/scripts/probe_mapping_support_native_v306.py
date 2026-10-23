"""Sealed TRAIN-only native cost admission; zero optimizer updates.

This measures the actual stagewise full-pixel gradient algorithm, not N1
forward smoke or a resized512 input. It never starts a training successor.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
import torch

OLD = Path('/home/yuzhou/Desktop/sheaf-gbp/multiroom-domain-v295-5090')
OLD_SEAL = '6e7a5d8e9abcfb53547fd3f2a63e6a510b8f6b2e341e97aa05913c9619f2d51b'
UUID = 'GPU-238bdbd5-9f06-da6d-09f6-830fd831e817'
MODEL_SHA = 'c2fe2f7eaca4603349c73ad41816ef73bc746b4d9fb908f954e1d8f46223552c'
FILES = ('mapping_support_local_v306.py', 'test_mapping_support_local_v306.py',
    'mapping_native_bank_v306.py', 'test_mapping_native_bank_v306.py',
    'mapping_support_forest_v306.py', 'test_mapping_support_forest_v306.py',
    'mapping_support_training_v306.py', 'test_mapping_support_training_v306.py',
    'mapping_support_pool_v306.py', 'test_mapping_support_pool_v306.py',
    'probe_mapping_support_native_v306.py', 'test_probe_mapping_support_native_v306.py')
HOPS = (0, 1, 2, 4, 8)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('x', encoding='utf-8') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
    temporary.replace(path)


def source_check():
    if sha(OLD/'source-manifest.json') != OLD_SEAL:
        raise ValueError('Original source seal changed')
    entries = read(OLD/'source-manifest.json')['files']
    for name, digest in entries.items():
        if sha(OLD/'source'/name) != digest:
            raise ValueError('Original source changed: '+name)
    return entries


def prepare(output, protocol):
    output = Path(output).resolve()
    if output.exists() or OLD == output or OLD in output.parents:
        raise FileExistsError('Fresh independent admission directory required')
    p = read(protocol)
    if p.get('version') != 306 or p.get('optimizer_updates') != 0 or p.get('TEST_opened') is not False:
        raise ValueError('Registered zero-update admission required')
    inherited = source_check()
    origin = Path(__file__).resolve().parent
    if sha(origin/'mapping_support_local_v306.py') != MODEL_SHA:
        raise ValueError('Final independently reviewed model required')
    for name in FILES:
        if not (origin/name).is_file():
            raise FileNotFoundError(name)
    output.mkdir(parents=True)
    shutil.copytree(OLD/'source', output/'source')
    for name in FILES:
        if (output/'source/scripts'/name).exists():
            raise FileExistsError('No overwrite of inherited source')
        shutil.copyfile(origin/name, output/'source/scripts'/name)
    write(output/'registered-admission.json', p)
    files = {name:sha(output/'source'/name) for name in inherited}
    files.update({'scripts/'+name:sha(output/'source/scripts'/name) for name in FILES})
    write(output/'source-manifest.json', dict(files=files, inherited_source_sha256=OLD_SEAL,
        protocol_sha256=sha(output/'registered-admission.json')))
    write(output/'status.json', dict(state='prepared_zero_update_admission', optimizer_updates=0))
    return dict(state='prepared_zero_update_admission', output=str(output), files=len(files))


def check(output):
    output = Path(output).resolve()
    if Path(__file__).resolve() != output/'source/scripts'/Path(__file__).name:
        raise ValueError('Execute the sealed external probe only')
    seal = read(output/'source-manifest.json')
    if seal['protocol_sha256'] != sha(output/'registered-admission.json'):
        raise ValueError('Admission changed')
    for name, digest in seal['files'].items():
        if sha(output/'source'/name) != digest:
            raise ValueError('Sealed source changed: '+name)
    p = read(output/'registered-admission.json')
    if (p['version'] != 306 or p['optimizer_updates'] != 0 or p['TEST_opened'] is not False
            or p['device_uuid'] != UUID or p['arms'] != ['support_local','cross_address']
            or p['sizes'] != [128,256,512] or p['agents'] != [8,16]
            or p['rooms'] != 20 or p['TRAIN_index'] != 0 or p['decode_rounds'] != list(HOPS)
            or p['warmups_per_case'] != 1 or p['measured_passes_per_case'] != 2
            or p['proposed_updates_per_arm'] != 2160 or p['auto_train_after_probe'] is not False):
        raise ValueError('Fixed registered native cost scope required')
    return p


def estimate(records, updates, evaluation_allowance_seconds):
    """Each arm averages the equal six H/N cohorts, not the fastest sample."""
    if type(updates) is not int or updates <= 0:
        raise ValueError('Positive fixed updates required')
    times = {}
    for arm in ('support_local','cross_address'):
        rows = [r for r in records if r['arm'] == arm]
        if len(rows) != 6 or {(r['size'],r['agents']) for r in rows} != {(h,n) for h in (128,256,512) for n in (8,16)}:
            raise ValueError('All native H/N cohorts required')
        means = [float(np.mean(r['measured_seconds'])) for r in rows]
        if not np.isfinite(means).all() or min(means) <= 0:
            raise ValueError('Positive finite complete backward times required')
        times[arm] = float(np.mean(means))*updates
    return dict(gradient_training_seconds=times,
        estimated_pair_seconds=sum(times.values())*1.5+evaluation_allowance_seconds,
        overhead_multiplier=1.5, evaluation_and_IO_allowance_seconds=evaluation_allowance_seconds,
        estimate_not_measured_training_duration=True)


def pooling_parity(device):
    """Only the legacy reference may use nondeterministic AdaptiveAvgPool CUDA.

    No model update. The actual execution replacement keeps strict determinism.
    """
    from scripts.mapping_support_pool_v306 import FixedNativePool4
    rows=[]
    for side in (4,8,16):
        reference=torch.linspace(-.4,.7,2*64*side*side,device=device).reshape(2,64,side,side).requires_grad_()
        replacement=reference.detach().clone().requires_grad_(True)
        upstream=torch.linspace(-.3,.8,2*64*16,device=device).reshape(2,64,4,4)
        try:
            torch.use_deterministic_algorithms(False)
            expected=torch.nn.functional.adaptive_avg_pool2d(reference,(4,4))
            expected.backward(upstream)
        finally:
            torch.use_deterministic_algorithms(True)
        actual=FixedNativePool4()(replacement);actual.backward(upstream)
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-7)
        torch.testing.assert_close(replacement.grad,reference.grad,rtol=2e-5,atol=2e-7)
        rows.append(dict(input_side=side,max_output_error=float((actual-expected).abs().max()),
            max_gradient_error=float((replacement.grad-reference.grad).abs().max())))
    return dict(passed=True,rows=rows,rtol=2e-5,atol=2e-7,
        original_reference_backward_only_determinism_disabled=True,actual_execution_deterministic=True)


def run(output):
    output = Path(output).resolve(); p = check(output)
    started = output/'started.json'
    with started.open('x', encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(), optimizer_updates=0),f)
    records = []; tick = time.monotonic()
    try:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        if os.environ['CUBLAS_WORKSPACE_CONFIG'] != ':4096:8':
            raise RuntimeError('Registered deterministic cuBLAS workspace required')
        raw = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total',
            '--format=csv,noheader'],text=True).strip()
        processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory',
            '--format=csv,noheader'],text=True).strip()
        if UUID not in raw or processes:
            raise RuntimeError('Authorized idle5090 required before GPU initialization')
        torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('One visible authorized GPU required')
        from scripts.mapping_support_local_v306 import SupportLocal64
        from scripts.mapping_support_pool_v306 import enable_native_pool
        from scripts.mapping_support_forest_v306 import rollout
        from scripts.mapping_support_training_v306 import backward_staged
        from scripts.mapping_native_bank_v306 import data, forest_targets, select_forest
        device = torch.device('cuda:0'); initial_hashes = {}
        pool_proof=pooling_parity(device)
        write(output/'pooling-parity.json',pool_proof)
        for arm in p['arms']:
            model=SupportLocal64(arm=arm).to(device)
            enable_native_pool(model); initial_hashes[arm]=model.weights_hash()
            for h in p['sizes']:
                for n in p['agents']:
                    spec=dict(split='TRAIN',index=0,rooms=20,agents=n,size=h)
                    t=time.monotonic(); fixture=data.make_fixture(**spec); generation=time.monotonic()-t
                    inp=fixture['inputs']; forest=select_forest(inp['positions_m'],inp['adjacency'])
                    t=time.monotonic(); labels={k:forest_targets(inp,k)[0] for k in HOPS}; label_seconds=time.monotonic()-t
                    own=torch.as_tensor(inp['private'][None],device=device)
                    targets={k:torch.as_tensor(v[None],device=device) for k,v in labels.items()}
                    seconds=[]; peak=[]; last=None; groups=None
                    for iteration in range(3):
                        model.zero_grad(set_to_none=True); torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats(); begin=time.monotonic()
                        result=rollout(model,own,forest['edges'],rounds=8,decode_rounds=HOPS,
                            features_only=True,trace=False)
                        last=backward_staged(model,{k:result['features'][k] for k in HOPS},targets)
                        torch.cuda.synchronize(); elapsed=time.monotonic()-begin
                        allocated=int(torch.cuda.max_memory_allocated())
                        groups={name:float(sum(v.grad.detach().double().square().sum().item()
                            for v in pars if v.grad is not None)**.5) for name,pars in model.parameter_groups().items()}
                        if any(v.grad is None or not bool(torch.isfinite(v.grad).all()) for v in model.parameters()):
                            raise AssertionError('Disconnected/nonfinite actual full gradient')
                        if any(value<=0 or not np.isfinite(value) for value in groups.values()):
                            raise AssertionError('Missing live optimized group')
                        if iteration:seconds.append(elapsed);peak.append(allocated)
                        del result
                    records.append(dict(arm=arm,size=h,agents=n,spec=spec,measured_seconds=seconds,
                        peak_allocated_bytes=max(peak),gradient_norms=groups,last_loss=last,
                        sensor_generation_seconds=generation,target_generation_seconds=label_seconds,
                        source_sha256=fixture['manifest']['source_sha256'],
                        own_known_fraction=float((inp['private']>=0).mean())))
                    write(output/'progress.json',dict(state='measuring_zero_update_native_cost',records=records,
                        optimizer_updates=0,elapsed_seconds=time.monotonic()-tick))
                    del own,targets,labels,fixture,inp;gc.collect();torch.cuda.empty_cache()
            if model.weights_hash()!=initial_hashes[arm]:
                raise AssertionError('Zero-update admission changed parameters')
            del model;gc.collect();torch.cuda.empty_cache()
        if len(set(initial_hashes.values())) != 1:
            raise AssertionError('Arms do not share initial tensors')
        predicted=estimate(records,2160,p['evaluation_and_IO_allowance_seconds'])
        passed=(max(r['peak_allocated_bytes'] for r in records)<=p['peak_allocated_bytes_limit']
            and predicted['estimated_pair_seconds']<=p['estimate_seconds_limit'])
        report=dict(version=306,state='complete_cost_admission',passed=passed,records=records,
            initial_hashes=initial_hashes,estimates=predicted,elapsed_seconds=time.monotonic()-tick,
            optimizer_updates=0,TRAIN_only=True,TEST_opened=False,model_or_accuracy_admitted=False,
            actual_packets_audited_here=False,automatic_training_started=False,
            pooling_parity=pool_proof,execution_only_regular_pool_repair=True,
            gpu_inventory=raw,source_manifest_sha256=sha(output/'source-manifest.json'),
            protocol_sha256=sha(output/'registered-admission.json'))
        write(output/'report.json',report);write(output/'status.json',dict(state=report['state'],passed=passed,optimizer_updates=0))
        return report
    except BaseException:
        write(output/'failure.json',dict(state='failed_cost_admission',records=records,
            error=traceback.format_exc(),optimizer_updates=0))
        write(output/'status.json',dict(state='failed_cost_admission',optimizer_updates=0))
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','run'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--protocol',type=Path)
    args=parser.parse_args()
    if args.mode=='prepare':
        if args.protocol is None:parser.error('prepare needs --protocol')
        result=prepare(args.output,args.protocol)
    else:result=run(args.output)
    print({k:result[k] for k in ('state','passed','elapsed_seconds','output') if k in result},flush=True)


if __name__=='__main__':main()
