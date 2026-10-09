#!/usr/bin/env python3
"""Single seed42 shuffled-BGS run, retaining LCC as the primary evaluation."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.bgs_guidance_A_control.evaluation import (
    PROTOCOL, METRICS, sha256, reuse_references, evaluate_shuffle, write_outputs, write_csv,
    summarize, validate_coverage)
from experiments.bgs_guidance_A.run_experiment import write_json, validation_history


def verify_protected():
    manifest=json.loads((Path(__file__).parent/'protected_files_before.json').read_text())
    changed=[p for p,h in manifest.items() if sha256(ROOT/p)!=h]
    if changed: raise RuntimeError('Existing baseline/BGS/audit files changed: '+str(changed))
    return len(manifest)


def verify_initialization(snapshot, expected):
    path=snapshot/'initialization.json'
    if path.exists():
        initial=json.loads(path.read_text())
        if initial['student_sha256']!=expected or initial['teacher_sha256']!=expected:
            raise RuntimeError('Training initialization differs from the seed42 reference')
        return True
    return False


def main():
    import os
    import torch
    from model.vnet import VNet
    from experiments.bgs_guidance_A.checkpointed_train import (
        tensor_state_digest, load_recovery_state)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output_dir',type=Path,default=ROOT/'Results/experiment_A_control_shuffle_seed42')
    args=parser.parse_args()
    output=args.output_dir.resolve()
    if output.exists(): raise FileExistsError('Refusing to overwrite '+str(output))
    cases=(ROOT.parent/'Dataset/LA/test.list').read_text().splitlines()
    if len(cases)!=20 or len(set(cases))!=20: raise ValueError('Exactly 20 distinct LA cases are required')
    reused,provenance=reuse_references(cases)
    protected_count=verify_protected()
    torch.manual_seed(42)
    reference=VNet(n_channels=1,n_classes=2)
    expected=tensor_state_digest(reference.state_dict()); del reference
    output.mkdir(parents=True)
    write_json(output/'protocol.json',dict(PROTOCOL,case_ids=cases,expected_initialization_sha256=expected))
    write_json(output/'reuse_provenance.json',provenance)
    write_outputs(output,reused,cases)
    snapshot=output/'result_LA_10l/fold_0'
    entrypoint=ROOT/'experiments/bgs_guidance_A/checkpointed_train.py'
    command=[sys.executable,str(entrypoint),'--data_path',str(ROOT.parent/'Dataset/LA'),
             '--dataset','LA','--labeled_num','10','--patch_size','112','112','80',
             '--batch_size','4','--labeled_bs','2','--seed','42','--base_lr','0.01',
             '--max_iterations','30000','--test_interval','200','--n_fold','1',
             '--consistency_rampup','200','--consistency','0.1','--ema_decay','0.9',
             '--device',args.device,'--output_dir',str(output),
             '--guidance_mode','shuffled_bgs','--bgs_alpha','0.1']
    metadata=dict(stage='starting',started_at=datetime.now().isoformat(),command=command,
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),device=args.device,
                  protocol=PROTOCOL,reused=provenance,protected_files=protected_count,
                  expected_initialization_sha256=expected)
    write_json(output/'metadata.json',metadata)
    code_dir=output/'control_source'; code_dir.mkdir()
    for source in [ROOT/'train.py',ROOT/'trainer.py',ROOT/'utils/boundary_guidance.py',entrypoint,
                   Path(__file__),Path(__file__).parent/'evaluation.py']:
        (code_dir/source.name).write_bytes(source.read_bytes())
    write_json(output/'run_status.json',dict(stage='starting',mode='shuffled_bgs',max_iterations=30000))
    print('A-Control starting: '+str(output),flush=True)
    try:
        start=time.monotonic()
        with (output/'stdout.log').open('w') as console:
            process=subprocess.Popen(command,cwd=ROOT,stdout=console,stderr=subprocess.STDOUT)
            while True:
                try: code=process.wait(timeout=30)
                except subprocess.TimeoutExpired: code=None
                progress=dict(stage='training',mode='shuffled_bgs',pid=process.pid,max_iterations=30000,
                              elapsed_seconds=time.monotonic()-start)
                path=snapshot/'log.txt'
                if path.exists():
                    content=path.read_text()
                    losses=re.findall(r'iteration (\d+) : loss :',content)
                    progress['last_completed_iteration']=int(losses[-1]) if losses else None
                    history=validation_history(path)
                    if history:
                        progress.update(latest_validation=history[-1],best_validation=max(history,key=lambda r:r['dice']))
                    progress['same_initialization_verified']=verify_initialization(snapshot,expected)
                write_json(output/'run_status.json',progress)
                if code is not None:
                    if code: raise RuntimeError('Training exited with status '+str(code)+'; see stdout.log')
                    break
        history=validation_history(snapshot/'log.txt')
        if progress.get('last_completed_iteration')!=29999 or history[-1]['iteration']!=30000:
            raise RuntimeError('The full 30000-iteration protocol did not complete')
        write_csv(output/'validation_history.csv',history)
        best_state=load_recovery_state(snapshot/'recovery_best.pt')
        iteration=best_state['completed_iterations']
        best_row=next(r for r in history if r['iteration']==iteration)
        best=dict(best_row,dice=float(best_state['best_performance']))
        checkpoint=snapshot/('Model_iter_%d.pth'%iteration)
        kwargs={'weights_only':True} if 'weights_only' in __import__('inspect').signature(torch.load).parameters else {}
        ema=torch.load(checkpoint,map_location='cpu',**kwargs)
        if any(not torch.equal(ema[k],best_state['teacher'][k]) for k in ema):
            raise RuntimeError('Full best checkpoint does not match the original best EMA checkpoint')
        if best_state['bgs_shuffle_rng'] is None:
            raise RuntimeError('Full checkpoint is missing the dedicated permutation RNG')
        del best_state,ema
        write_json(output/'run_status.json',dict(stage='evaluating_raw_and_lcc',best_validation=best))
        shuffled=evaluate_shuffle(checkpoint,iteration,args.device,ROOT.parent/'Dataset/LA',cases)
        validate_coverage(reused+shuffled,cases,('none','bgs','shuffled_bgs'))
        summary=summarize(reused+shuffled)
        actual=next(r for r in summary if r['mode']=='shuffled_bgs' and r['prediction']=='lcc')
        if abs(actual['dice']-best['dice'])>1e-5:
            raise RuntimeError('LCC final evaluation does not reproduce primary validation')
        checks=re.findall(r'BGS shuffle iteration (\d+) : distribution_equal (True|False) mean_delta ([\deE+.-]+) norm_delta ([\deE+.-]+) skip (True|False)',(snapshot/'log.txt').read_text())
        if len(checks)!=30000 or any(r[1]!='True' for r in checks):
            raise RuntimeError('Missing or failed weight-permutation invariant checks')
        write_json(output/'control_checks.json',dict(logged_iterations=len(checks),distribution_exact=True,
                   max_fp64_mean_delta=max(float(r[2]) for r in checks),max_fp64_norm_delta=max(float(r[3]) for r in checks),
                   skipped_iterations=sum(r[4]=='True' for r in checks),same_initialization=True,
                   complete_best_checkpoint_matches_ema=True,shuffle_rng_saved=True,
                   all_20_cases_retained=True,primary='LCC Dice'))
        inventory={name:sha256(snapshot/name) for name in
                   ('recovery_initial.pt','recovery_latest.pt','recovery_best.pt','recovery_final.pt')}
        if verify_protected()!=protected_count:
            raise RuntimeError('Protected reference files changed')
        write_outputs(output,reused+shuffled,cases,complete=True)
        metadata.update(stage='complete',finished_at=datetime.now().isoformat(),best_validation=best,
                        summary=summary,complete_checkpoint_sha256=inventory,
                        protected_files_unchanged=True)
        write_json(output/'metadata.json',metadata)
        write_json(output/'run_status.json',dict(stage='complete',best_validation=best,summary=summary))
        print(json.dumps(dict(stage='complete',best_validation=best,summary=summary)),flush=True)
    except BaseException as error:
        metadata.update(stage='failed',error=str(error),traceback=traceback.format_exc())
        write_json(output/'metadata.json',metadata)
        write_json(output/'run_status.json',dict(stage='failed',error=str(error)))
        raise


if __name__=='__main__':
    main()
