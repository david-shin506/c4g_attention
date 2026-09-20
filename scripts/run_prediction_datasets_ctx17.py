#!/usr/bin/env python3
"""Durable sequential runner: Spring no-align, then CamxTime per-time align.
Re-run with the same Python executable to resume committed samples/frames.
"""
from pathlib import Path
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[1]/'C4G_prediction_dataset'
RUN=ROOT/'outputs/dataset_generation_ctx17'
SPRING=BASE/'Spring_480x480'
OLD=BASE/'Spring_480x480_ctx12_replaced'
CAM=BASE/'CamxTime_480x480'

def status(state,**extra):
 data={'state':state,'pid':os.getpid(),'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra}
 tmp=RUN/'status.json.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(RUN/'status.json')
 print(json.dumps(data),flush=True)

def step(name,script,*args):
 command=[sys.executable,str(ROOT/'scripts'/script),*[str(a) for a in args]]
 with (RUN/f'{name}.log').open('a') as log:
  child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  status('running',stage=name,child_pid=child.pid,command=command,log=str(RUN/f'{name}.log'))
  code=child.wait()
 if code: raise RuntimeError(f'{name} failed, exit={code}; see {RUN / (name+".log")}')

def main():
 RUN.mkdir(parents=True,exist_ok=True)
 lock=(RUN/'.runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  spec=json.loads((SPRING/'dataset.json').read_text())
  assert spec['context_count']==17 and spec['target_count']==33 and spec['pose_alignment'] is False
  state=json.loads((SPRING/'status.json').read_text())
  if state['state']!='complete':
   extra=['--clean-cache-from',OLD] if OLD.exists() else []
   step('spring_export','export_spring_vace.py','--output',SPRING,'--height',480,'--width',480,
        '--context-count',17,'--cached-rgb-from',BASE/'Spring_480x832',*extra)
  if not (SPRING/'validation.json').is_file(): step('spring_validate','validate_spring_vace_export.py',SPRING)
  assert json.loads((SPRING/'validation.json').read_text())['full_export']
  if not (SPRING/'psnr_subset_3_per_scene_report.json').is_file():
   step('spring_psnr_subset','measure_spring_rgb_psnr.py',SPRING,'--samples-per-scene',3)
  if OLD.exists():
   status('running',stage='remove_replaced_spring')
   assert OLD.parent==BASE and OLD.name=='Spring_480x480_ctx12_replaced' and not OLD.is_symlink()
   old=json.loads((OLD/'dataset.json').read_text())
   assert old['context_count']==12 and old['target_count']==23 and old['output_image_shape']==[480,480]
   archive=RUN/'replaced_spring_ctx12_metadata.zip'
   with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in OLD.glob('*.json'): z.write(p,p.relative_to(OLD))
    for p in (OLD/'samples').glob('*/sample.json'): z.write(p,p.relative_to(OLD))
   shutil.rmtree(OLD)
   (RUN/'replaced_spring_removed.json').write_text(json.dumps({'removed':str(OLD),'after_full_new_validation':True,'metadata_archive':str(archive)},indent=2)+'\n')
  state=json.loads((CAM/'status.json').read_text()) if (CAM/'status.json').exists() else {}
  if state.get('state')!='complete': step('camxtime_export','export_camxtime_vace.py','--output',CAM)
  if not (CAM/'validation.json').is_file(): step('camxtime_validate','validate_camxtime_vace_export.py',CAM)
  assert json.loads((CAM/'validation.json').read_text())['full_export']
  status('complete',spring=str(SPRING),camxtime=str(CAM))
 except BaseException as e:
  status('failed',error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':main()
