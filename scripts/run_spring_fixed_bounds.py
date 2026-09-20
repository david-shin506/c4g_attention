#!/usr/bin/env python3
"""Train to 4k, conditionally extend to 8k, then regenerate both Spring datasets."""
import fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
RUN=PROJECT/'outputs/spring_vae_feature_fixed_bounds_20260913'
BASE=PROJECT.parent.parent/'C4G_prediction_dataset'
VAE=BASE/'Spring_VAE_480x480_ctx8'
RGB=BASE/'Spring_480x480'

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def decide_extension(rows):
    baseline=[r for r in rows if r['step']==1000]
    assert len(baseline)==1,'Expected exactly one validation at step 1000'
    later=[r for r in rows if 1000<r['step']<=4000]
    assert any(r['step']==4000 for r in later),'Validation at 4000 is required'
    best=min(later,key=lambda r:r['latent_mse'])
    return {'extend_to_8000':best['latent_mse']<baseline[0]['latent_mse'],
            'step_1000_validation':baseline[0],'best_validation_1001_to_4000':best,
            'rule':'Extend only if the minimum fixed-set validation MSE after step 1000 through step 4000 is strictly below the step-1000 MSE.'}

def main():
    RUN.mkdir(exist_ok=True);os.chdir(PROJECT)
    lock=(RUN/'.pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (RUN/'supervisor.pid').write_text(str(os.getpid())+'\n')
    env=os.environ.copy();env.update(MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='9.0')
    env['PYTHONPATH']=str(PROJECT)+':'+env.get('PYTHONPATH','')
    def stage(name,**extra):
        write(RUN/'pipeline_status.json',{'state':name,'pid':os.getpid(),
              'slurm_job_id':env.get('SLURM_JOB_ID'),'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra})
        print(name,flush=True)
    def call(script,*args):
        subprocess.run([sys.executable,'-u','scripts/'+script,*map(str,args)],env=env,check=True)
    try:
        train=['--data',VAE,'--run',RUN/'training','--steps','8000','--fixed-bounds']
        decision_path=RUN/'extension_decision.json'
        if not decision_path.exists():
            stage('training_to_4000')
            call('train_spring_vae_features.py',*train,'--stop-after','4000')
            rows=[json.loads(s) for s in (RUN/'training/validation.jsonl').read_text().splitlines()]
            write(decision_path,decide_extension(rows))
        decision=json.loads(decision_path.read_text())
        if decision['extend_to_8000']:
            stage('training_to_8000',decision=decision)
            call('train_spring_vae_features.py',*train)
            chosen=RUN/'training/best.ckpt'
        else:chosen=RUN/'training/checkpoint_step_1000.ckpt'
        selected=RUN/'selected_checkpoint.ckpt'
        shutil.copy2(chosen,selected)
        write(RUN/'selection.json',{'source_checkpoint':str(chosen),'selected_checkpoint':str(selected),
              'sha256':hashlib.sha256(selected.read_bytes()).hexdigest(),'decision':decision})
        stage('exporting_vae_and_rgb',selected_checkpoint=str(selected))
        call('export_spring_vae_features.py','--output',VAE,'--checkpoint',selected,'--fixed-bounds','--save-rgb-pred')
        stage('validating_vae_and_rgb')
        call('validate_spring_vae_features.py',VAE)
        stage('exporting_rgb_480x480')
        backups=json.loads((RUN/'backup_manifest.json').read_text())
        call('export_spring_vace.py','--output',RGB,'--height','480','--width','480','--context-count','17',
             '--fixed-bounds','--cached-rgb-from',BASE/'Spring_480x832','--clean-cache-from',backups['Spring_480x480'])
        stage('validating_rgb_480x480')
        call('validate_spring_vace_export.py',RGB)
        stage('complete',selection=json.loads((RUN/'selection.json').read_text()),vae_output=str(VAE),rgb_output=str(RGB))
    except BaseException as error:
        stage('failed',error=f'{type(error).__name__}: {error}');raise

if __name__=='__main__':main()
