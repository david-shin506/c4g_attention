#!/usr/bin/env python3
"""Detached multi-GPU CamxTime export coordinator; resume by running this again."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from camxtime_sharding import partition_plan

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parents[1]/'C4G_prediction_dataset/CamxTime_480x480'
RUN=ROOT/'outputs/dataset_generation_ctx17'

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def now():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())

def main(args):
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    RUN.mkdir(parents=True,exist_ok=True)
    manager_lock=(RUN/'.parallel_runner.lock').open('a');fcntl.flock(manager_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    export_lock=(OUT/'.export.lock').open('a')
    # Exclusive acquisition proves all earlier single/sharded exporters stopped.
    fcntl.flock(export_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((OUT/'plan.json').read_text())['samples']
    partitions=partition_plan(plan,len(args.gpus));total=len(plan)
    assert len({r['sample_id'] for p in partitions for r in p})==total
    workers_dir=OUT/'workers'/f'shards_{len(args.gpus)}'
    workers_dir.mkdir(parents=True,exist_ok=True)
    existing=set()
    for p in (OUT/'samples').glob('*/sample.json'):
        m=json.loads(p.read_text());assert m['complete'];existing.add(m['sample_id'])
    partition_spec={'strategy':'Stable hash-sorted (scene,trajectory) round robin; all three windows of a rig share one worker',
                    'num_workers':len(args.gpus),'workers':[{'shard_id':i,'gpu':gpu,'sample_ids':[r['sample_id'] for r in part]}
                       for i,(gpu,part) in enumerate(zip(args.gpus,partitions))]}
    write(OUT/'sharding.json',partition_spec)
    for i,part in enumerate(partitions):
        done=sum(r['sample_id'] in existing for r in part)
        write(workers_dir/f'worker_{i}'/'status.json',{'state':'starting','shard_id':i,'total':len(part),'completed':done,
                     'remaining':len(part)-done,'updated_utc':now()})
    write(OUT/'index.json',{'schema_version':1,'complete':len(existing)==total,
          'samples':[{'sample_id':r['sample_id'],'path':f"samples/{r['sample_id']}/sample.json"} for r in plan if r['sample_id'] in existing]})
    # Shared reservation allows our workers, while excluding the old single-GPU runner.
    fcntl.flock(export_lock,fcntl.LOCK_SH)
    jobs=[];logs=[]
    started=time.monotonic()
    common={'pid':os.getpid(),'hostname':socket.gethostname(),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
            'num_workers':len(args.gpus),'gpus':args.gpus,'updated_utc':now(),'initial_completed':len(existing)}
    write(RUN/'launch_multigpu.json',{**common,'script':str(Path(__file__).resolve()),'python':sys.executable})
    try:
        for i,gpu in enumerate(args.gpus):
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
            log=(RUN/f'camxtime_worker_{i}.log').open('a');logs.append(log)
            command=[sys.executable,'-u',str(ROOT/'scripts/export_camxtime_vace.py'),'--output',str(OUT),
                     '--num-shards',str(len(args.gpus)),'--shard-id',str(i)]
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            jobs.append(process)
        while True:
            statuses=[]
            for i,job in enumerate(jobs):
                state=json.loads((workers_dir/f'worker_{i}'/'status.json').read_text())
                state.update(gpu=args.gpus[i],process_pid=job.pid,exit_code=job.poll())
                if job.returncode not in (None,0):state['state']='failed'
                statuses.append(state)
            done=sum(s['completed'] for s in statuses)
            failed=[s['shard_id'] for s in statuses if s['state']=='failed']
            running=any(j.poll() is None for j in jobs)
            complete=not running and not failed and done==total
            state='complete' if complete else 'running' if running else 'failed'
            eta=[s.get('estimated_remaining_hours') for s in statuses if s['remaining']]
            eta=max(eta) if eta and all(v is not None for v in eta) else None
            # Last measured window speed also updates ETA during a window.
            if eta is None:
                estimates=[(s.get('seconds_per_sample') or 0)*s['remaining']/3600 for s in statuses if s['remaining']]
                if estimates and all(v for v in estimates):eta=max(estimates)
            status={**common,'state':state,'updated_utc':now(),'completed':done,'total':total,'remaining':total-done,
                    'elapsed_seconds':time.monotonic()-started,'estimated_remaining_hours':eta,'failed_workers':failed,'workers':statuses}
            write(OUT/'status.json',status)
            write(RUN/'status.json',{**common,'state':state,'stage':f'camxtime_export_{len(jobs)}gpu','updated_utc':now(),
                  'completed':done,'total':total,'failed_workers':failed,'child_pids':[j.pid for j in jobs],
                  'estimated_remaining_hours':eta})
            if not running:
                if not complete:raise RuntimeError(f'Workers stopped before completion; failed={failed}, completed={done}/{total}')
                break
            time.sleep(10)
        entries=[]
        for r in plan:
            m=json.loads((OUT/'samples'/r['sample_id']/'sample.json').read_text());assert m['complete']
            entries.append({'sample_id':r['sample_id'],'path':f"samples/{r['sample_id']}/sample.json"})
        write(OUT/'index.json',{'schema_version':1,'complete':True,'samples':entries})
        write(RUN/'status.json',{**common,'state':'running','stage':'camxtime_validate','updated_utc':now()})
        with (RUN/'camxtime_validate.log').open('a') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/validate_camxtime_vace_export.py'),str(OUT)],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        write(RUN/'status.json',{**common,'state':'complete','stage':'complete','updated_utc':now(),'completed':total,'total':total})
    except BaseException as e:
        # A coordinator failure must not leave unmonitored children writing shared output.
        for job in jobs:
            if job.poll() is None:job.terminate()
        for job in jobs:
            if job.poll() is None:
                try:job.wait(timeout=20)
                except subprocess.TimeoutExpired:job.kill();job.wait()
        failure={'state':'failed','updated_utc':now(),'error':f'{type(e).__name__}: {e}'}
        if (OUT/'status.json').exists():
            last=json.loads((OUT/'status.json').read_text());write(OUT/'status.json',{**last,**failure})
        write(RUN/'status.json',{**common,**failure})
        raise
    finally:
        for log in logs:log.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',nargs='+',default=os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2,3').split(','))
    args=parser.parse_args()
    if len(args.gpus)<2 or len(set(args.gpus))!=len(args.gpus):parser.error('Need at least two distinct GPUs')
    main(args)
