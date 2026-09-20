#!/usr/bin/env python3
"""Learn a 16-channel Gaussian feature stream; frozen C4G and cached Wan teacher."""
import argparse,fcntl,json,math,os,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from spring_vae_common import *
from src.model.vae_feature_lifting import VAEFeatureLifter


def atomic_checkpoint(path, payload):
    tmp=path.with_name(path.name+'.tmp');torch.save(payload,tmp);tmp.replace(path)


def infer_rows(lifter, ds, root, row, offsets):
    batch=make_batch(ds,row,lifter.encoder)
    ctx=batch['context']['index'][0].tolist()
    inputs=load_latents(root,row['scene'],ctx)[None]
    times=batch['target']['index'][0][offsets]
    gs,features=lifter(batch['context'],inputs,times)
    pred=torch.cat([render_feature(gs[int(t)],features[int(t)],batch['target'],offset)
                    for t,offset in zip(times.tolist(),offsets)],dim=0)
    gt=load_latents(root,row['scene'],times.tolist())
    return pred,gt


@torch.no_grad()
def evaluate(lifter,ds,root,rows):
    losses=[];zeros=[]
    for row in rows:
        pred,gt=infer_rows(lifter,ds,root,row,[0,7,14])
        losses.append(float(F.mse_loss(pred,gt)));zeros.append(float(gt.square().mean()))
    return {'latent_mse':sum(losses)/len(losses),'zero_feature_mse':sum(zeros)/len(zeros),'windows':len(rows),'frames':len(rows)*3}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=DATA_ROOT);p.add_argument('--run',type=Path,default=RUN_ROOT)
    p.add_argument('--steps',type=int,default=8000);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--fixed-bounds',action='store_true')
    p.add_argument('--stop-after',type=int,help='Pause at this step, preserving the configured full training horizon')
    p.add_argument('--eval-every',type=int,default=250);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(111123);np.random.seed(111123)
    run=args.run.resolve();run.mkdir(parents=True,exist_ok=True)
    lock=(run/'.train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not args.smoke:assert json.loads((args.data/'prepare_status.json').read_text())['state']=='complete'
    raw,ds,plan,_=load_dataset(fixed_bounds=args.fixed_bounds)
    val_scenes={'0014','0038'}
    training=[r for r in plan if r['scene'] not in val_scenes]
    validation=[]
    for scene in sorted(val_scenes):
        candidates=[r for r in plan if r['scene']==scene]
        validation.extend([candidates[0],candidates[len(candidates)//2]])
    config={'steps':args.steps,'lr':args.lr,'seed':111123,'data_root':str(args.data.resolve()),
            'checkpoint':str(DEFAULT_RUN/'checkpoints/epoch_2-step_45000.ckpt'),'base_step':45000,
            'training_windows':len(training),'validation_scenes':sorted(val_scenes),'validation_windows':validation,
            'train_targets_per_step':2,'target_sampling':'one context timestamp and one interpolated timestamp per step',
            'loss':'MSE against normalized 16x60x60 framewise Wan teacher latents; no feature L2 normalization',
            'frozen':'all original C4G weights and Gaussian geometry; Wan VAE teacher cached',
            'feature_architecture':'HG Instill: frozen Q/K and original x-stream; trained value/output projections, FFN, norm, 16-channel head',
            'smoke':args.smoke,
            'render_bounds':{'mode':'fixed' if args.fixed_bounds else 'baseline_scaled','near':0.1,'far':100.0,'divided_by_baseline':not args.fixed_bounds}}
    if (run/'config.json').exists():assert json.loads((run/'config.json').read_text())==config,'Run config mismatch'
    else:atomic_json(run/'config.json',config)
    encoder,_,_=build_encoder(raw,DEFAULT_RUN/'checkpoints/epoch_2-step_45000.ckpt')
    # Real-data geometry equivalence check before learning.
    probe=training[0];batch=make_batch(ds,probe,encoder);ts=batch['target']['index'][0][:2]
    with torch.no_grad():base=encoder(batch['context'],45000,target_timestamps=ts)
    lifter=VAEFeatureLifter(encoder).cuda()
    ctx=load_latents(args.data,probe['scene'],batch['context']['index'][0].tolist())[None]
    with torch.no_grad():check,_=lifter(batch['context'],ctx,ts)
    max_diff=max(float((getattr(base[int(t)],k)-getattr(check[int(t)],k)).abs().max())
                 for t in ts for k in ['means','covariances','harmonics','opacities'])
    assert max_diff<1e-5,('Geometry changed',max_diff)
    del base,check,batch,ctx
    parameters=list(lifter.decoder.trainable_parameters())
    ids={id(p) for p in parameters}
    assert all(not p.requires_grad for p in encoder.parameters() if id(p) not in ids)
    opt=torch.optim.AdamW(parameters,lr=args.lr,weight_decay=0.01)
    rng=np.random.default_rng(111123);start_step=0;best=math.inf
    latest=run/'latest.ckpt'
    if latest.exists():
        state=torch.load(str(latest),map_location='cpu');lifter.decoder.load_feature_state_dict(state['features'])
        opt.load_state_dict(state['optimizer']);start_step=state['step'];best=state['best_val_mse']
        rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
    started=time.monotonic();history=[]
    def save(step):
        payload={'step':step,'features':lifter.decoder.feature_state_dict(),'optimizer':opt.state_dict(),
                 'best_val_mse':best,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                 'cuda_rng':torch.cuda.get_rng_state_all(),'config':config}
        atomic_checkpoint(latest,payload)
    if args.smoke:
        validation=[probe]
    if start_step==0:
        baseline=evaluate(lifter,ds,args.data,validation)
        atomic_json(run/'initial_validation.json',baseline)
        print(json.dumps({'initial_validation':baseline,'geometry_max_abs_diff':max_diff,'trainable_parameters':sum(p.numel() for p in parameters)}),flush=True)
    smoke_offsets=[0,1]
    end_step=min(args.steps,args.stop_after) if args.stop_after is not None else args.steps
    if end_step<start_step:raise ValueError('Stop step precedes saved checkpoint')
    for step in range(start_step+1,end_step+1):
        row=probe if args.smoke else training[int(rng.integers(len(training)))]
        offsets=smoke_offsets if args.smoke else [int(rng.choice(np.arange(0,15,2))),int(rng.choice(np.arange(1,15,2)))]
        opt.zero_grad(set_to_none=True)
        pred,gt=infer_rows(lifter,ds,args.data,row,offsets)
        loss=F.mse_loss(pred,gt)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite feature loss')
        loss.backward()
        grad_norm=torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True)
        if step==start_step+1:
            assert float(grad_norm)>0
            assert all(p.grad is None for p in encoder.parameters() if id(p) not in ids)
        opt.step();history.append(float(loss.detach()))
        del pred,gt,loss
        if step<=5 or step%10==0 or step==args.steps:
            elapsed=time.monotonic()-started
            status={'state':'training','pid':os.getpid(),'step':step,'total_steps':args.steps,
                    'train_latent_mse':history[-1],'mean_last_10':sum(history[-10:])/len(history[-10:]),
                    'grad_norm':float(grad_norm),'elapsed_this_process_s':elapsed,
                    'seconds_per_step':elapsed/(step-start_step),
                    'estimated_remaining_s':elapsed/(step-start_step)*(args.steps-step),
                    'gpu_peak_GiB':torch.cuda.max_memory_allocated()/2**30}
            atomic_json(run/'status.json',status)
            with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
            print(json.dumps(status),flush=True)
        if step%args.eval_every==0 or step==end_step:
            result=evaluate(lifter,ds,args.data,validation)
            with (run/'validation.jsonl').open('a') as f:f.write(json.dumps({'step':step,**result})+'\n')
            if result['latent_mse']<best:
                best=result['latent_mse']
                atomic_checkpoint(run/'best.ckpt',{'step':step,'features':lifter.decoder.feature_state_dict(),'config':config,'validation':result})
            if step==1000:
                atomic_checkpoint(run/'checkpoint_step_1000.ckpt',{'step':step,'features':lifter.decoder.feature_state_dict(),'config':config,'validation':result})
            save(step)
            print(json.dumps({'step':step,'validation':result,'best_val_mse':best}),flush=True)
    if args.smoke:
        assert min(history[1:])<history[0],'Feature loss did not decrease in smoke training'
        result={'passed':True,'geometry_max_abs_diff':max_diff,'finite_nonzero_feature_gradients':True,
                'original_parameters_frozen':True,'loss_history':history,'gpu_peak_GiB':torch.cuda.max_memory_allocated()/2**30}
        atomic_json(run/'smoke_validation.json',result)
    atomic_json(run/'status.json',{'state':'complete' if end_step==args.steps else 'paused','step':end_step,'total_steps':args.steps,'best_val_mse':best,'elapsed_this_process_s':time.monotonic()-started})

if __name__=='__main__':main()
