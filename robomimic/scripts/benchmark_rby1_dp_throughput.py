"""Bounded training-throughput measurements; no checkpoints or full epochs."""
import argparse
import gc
import json
import os
from pathlib import Path

from robomimic.utils.rby1_paths import new_output, ROBOMIMIC_ROOT, resolve_training_paths
import subprocess
import threading
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from robomimic.algo import algo_factory
from robomimic.config import config_factory
from robomimic.utils import file_utils as F, obs_utils as O, train_utils as T


def gpu_monitor(stop, samples):
    while not stop.is_set():
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                samples.append((time.perf_counter(), float(result.stdout.splitlines()[0])))
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
        stop.wait(.5)


def measure(config, dataset, shape, batch_size, workers, warmup, steps, validate=False):
    gc.collect(); torch.cuda.empty_cache()
    np.random.seed(1); torch.manual_seed(1)
    model = algo_factory(config.algo_name, config, obs_key_shapes=shape['all_shapes'],
                         ac_dim=shape['ac_dim'], device=torch.device('cuda'))
    model.set_eval() if validate else model.set_train()
    torch.manual_seed(1)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=dataset.get_dataset_sampler(),
                        shuffle=dataset.get_dataset_sampler() is None, num_workers=workers, drop_last=True)
    iterator = None
    times, util_samples = [], []
    stop = threading.Event(); thread = None
    row = {'batch_size': batch_size, 'workers': workers, 'validate': validate,
           'warmup_batches': warmup, 'measured_batches': steps, 'oom': False,
           'loader_batches_per_pass': len(loader), 'error': None}
    try:
        iterator = iter(loader)
        for index in range(warmup + steps):
            if index == warmup:
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                thread = threading.Thread(target=gpu_monitor, args=(stop, util_samples), daemon=True); thread.start()
                measured_start = time.perf_counter()
            started = time.perf_counter()
            batch = next(iterator)
            batch = model.process_batch_for_training(batch)
            batch = model.postprocess_batch_for_training(batch, obs_normalization_stats=None)
            with torch.no_grad() if validate else torch.enable_grad():
                info = model.train_on_batch(batch, epoch=1, validate=validate)
            model.log_info(info)
            torch.cuda.synchronize()
            if index >= warmup:
                times.append(time.perf_counter() - started)
            del batch, info
            if index >= warmup and (index - warmup + 1) % 25 == 0:
                print(f'batch={batch_size}, workers={workers}, validation={validate}: {index-warmup+1}/{steps}, mean={np.mean(times):.4f}s', flush=True)
        measured_end = time.perf_counter()
        row.update(seconds_per_batch=float(np.mean(times)), samples_per_second=batch_size/float(np.mean(times)),
                   batch_time_median=float(np.median(times)), batch_time_p90=float(np.percentile(times,90)),
                   windows_seconds_per_batch=[float(np.mean(part)) for part in np.array_split(times,4)],
                   allocated_vram_mib=torch.cuda.max_memory_allocated()/2**20,
                   reserved_vram_mib=torch.cuda.max_memory_reserved()/2**20)
        stop.set()
        if thread is not None: thread.join(timeout=5)
        utils=[value for timestamp,value in util_samples if measured_start <= timestamp <= measured_end]
        row.update(gpu_utilization_mean_percent=float(np.mean(utils)) if utils else None,
                   gpu_utilization_median_percent=float(np.median(utils)) if utils else None,
                   gpu_utilization_p90_percent=float(np.percentile(utils,90)) if utils else None,
                   gpu_utilization_samples=len(utils))
    except torch.cuda.OutOfMemoryError as error:
        row.update(oom=True, error=str(error), allocated_vram_mib=torch.cuda.max_memory_allocated()/2**20,
                   reserved_vram_mib=torch.cuda.max_memory_reserved()/2**20)
    except Exception as error:
        row['error'] = f'{type(error).__name__}: {error}'
    finally:
        stop.set()
        if thread is not None: thread.join(timeout=5)
        if iterator is not None and workers:
            iterator._shutdown_workers()
        del iterator, loader, model
        gc.collect(); torch.cuda.empty_cache()
    print(json.dumps(row), flush=True)
    return row


def main(args):
    config_path=Path(args.config); original=config_path.read_bytes()
    raw=resolve_training_paths(json.loads(original)); config=config_factory(raw['algo_name']); config.update(raw)
    O.initialize_obs_utils_with_config(config)
    shape=F.get_shape_metadata_from_dataset(os.path.expanduser(config.train.data[0]['path']),
        action_keys=config.train.action_keys, all_obs_keys=config.all_obs_keys, ds_format=config.train.data_format)
    train,valid=T.load_data_for_training(config,shape['all_obs_keys'])
    valid.set_action_normalization_stats(train.get_action_normalization_stats())
    output=Path(args.output); output.mkdir(parents=True,exist_ok=False)
    results={'config':str(config_path.resolve()),'warmup_batches':args.warmup,'measured_training_batches':args.steps,
             'gpu':torch.cuda.get_device_name(),'train_sequences':len(train),'validation_sequences':len(valid),
             'training_rows':[], 'validation_rows':[], 'checkpoints_saved':False}
    def save(): (output/'summary.json').write_text(json.dumps(results,indent=2))
    for workers in (0,2,4):
        results['training_rows'].append(measure(config,train,shape,16,workers,args.warmup,args.steps)); save()
    viable=[row for row in results['training_rows'] if row['error'] is None]
    if not viable: raise RuntimeError('No successful batch-16 measurement')
    fastest=min(viable,key=lambda row:row['seconds_per_batch'])
    # Prefer fewer workers when their throughput is within 5% of the fastest.
    best=min((row for row in viable if row['seconds_per_batch']<=fastest['seconds_per_batch']*1.05),key=lambda row:row['workers'])
    results['selected_workers']=best['workers'];save()
    for batch_size in (32,64):
        results['training_rows'].append(measure(config,train,shape,batch_size,best['workers'],args.warmup,args.steps));save()
    for row in list(results['training_rows']):
        if row['workers']==best['workers'] and row['error'] is None:
            results['validation_rows'].append(measure(config,valid,shape,row['batch_size'],min(best['workers'],1),10,50,validate=True));save()
    assert config_path.read_bytes()==original, 'Training config changed unexpectedly'
    train.close_and_delete_hdf5_handle();valid.close_and_delete_hdf5_handle()
    print('Bounded benchmark complete:',output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(ROBOMIMIC_ROOT / "robomimic/exps/rby1_close_single_door_cartesian_diffusion_fullpass.json"))
    parser.add_argument("--output",default=str(new_output("CloseSingleDoor", "diagnostics", "rby1_throughput")))
    parser.add_argument('--warmup',type=int,default=15)
    parser.add_argument('--steps',type=int,default=100)
    args=parser.parse_args()
    if args.steps<20 or args.steps>200 or args.warmup<1 or args.warmup>50:
        parser.error('Bounded benchmark requires 20-200 measured batches and 1-50 warmup batches')
    main(args)
