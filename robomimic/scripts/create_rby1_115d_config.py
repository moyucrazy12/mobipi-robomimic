"""Count real SequenceDataset samples/loaders; change only new-run fields."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from robomimic.config import config_factory
from robomimic.utils import obs_utils as O,train_utils as T
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--work-dir',required=True,help='Directory containing successful dataset/replay reports')
parser.add_argument('--base-config',default='robomimic/exps/rby1_close_single_door_cartesian_cmd_diffusion_fullpass.json')
parser.add_argument('--output',required=True,help='New config; existing files are never overwritten')
args=parser.parse_args()
ROOT=Path(args.work_dir).expanduser().resolve()
report=json.loads((ROOT/'dataset_report.json').read_text())
gate=json.loads((ROOT/'replay_gate.json').read_text())
assert gate['passed'] and gate['demos']==6
assert gate['dataset']==report['final'] and gate['source_sha256']==report['source_sha256']
original=Path(args.base_config).expanduser()
output=Path(args.output).expanduser()
assert not output.exists()
base=json.loads(original.read_text());new=copy.deepcopy(base)
new['experiment']['name']='rby1_close_single_door_cartesian_cmd_115d_15pass'
new['train']['data'][0]['path']=report['final']
new['train']['num_epochs']=15
config=config_factory('diffusion_policy')
with config.values_unlocked():config.update(new)
config.lock();O.initialize_obs_utils_with_config(config)
keys=list(config.observation.modalities.obs.low_dim)+list(config.observation.modalities.obs.rgb)
train,valid=T.load_data_for_training(config,keys)
assert type(train).__name__=='SequenceDataset' and type(valid).__name__=='SequenceDataset'
assert len(train.demos)==100 and len(valid.demos)==15
loaders=[DataLoader(d,batch_size=16,sampler=d.get_dataset_sampler(),num_workers=0,drop_last=True) for d in [train,valid]]
counts=dict(total_demos=115,train_demos=len(train.demos),valid_demos=len(valid.demos),train_sequences=len(train),valid_sequences=len(valid),train_batches=len(loaders[0]),valid_batches=len(loaders[1]),dropped_train_sequences=len(train)%16,dropped_valid_sequences=len(valid)%16,sequence_shape_frame_plus_seq=16,batch_size=16,frame_stack=2,seq_length=15,pad_frame_stack=True,pad_seq_length=True,drop_last=True,dataset_classes=[type(train).__name__,type(valid).__name__],train_sampler=type(train.get_dataset_sampler()).__name__,valid_sampler=type(valid.get_dataset_sampler()).__name__)
assert len(train)+len(valid)==171258
# Real indexed samples exercise image loading and padding, rather than raw-count inference.
for d in [train,valid]:
    for index in [0,len(d)-1]:
        sample=d[index];assert sample['actions'].shape==(16,9)
        for k in keys:
            assert sample['obs'][k].shape[0]==16 and np.isfinite(sample['obs'][k]).all()
            if 'image' in k:assert sample['obs'][k].shape==(16,128,128,3)
new['experiment']['epoch_every_n_steps']=counts['train_batches']
new['experiment']['validation_epoch_every_n_steps']=counts['valid_batches']
assert new['train']['num_data_workers']==4 and new['train']['batch_size']==16
assert new['train']['action_keys']==['actions_eef_cmd'] and new['train']['action_config']['actions_eef_cmd']['normalization']=='min_max'
assert new['algo']==base['algo'] and new['observation']==base['observation']
assert not new['experiment']['rollout']['enabled'] and new['experiment']['save']['every_n_epochs']==1
output.write_text(json.dumps(new,indent=4)+'\n')
def changes(a,b,path=''):
    rows=[]
    if isinstance(a,dict):
        for k in a:rows+=changes(a[k],b[k],path+'.'+k if path else k)
    elif a!=b:rows.append(dict(field=path,old=a,new=b))
    return rows
diff=changes(base,new)
assert len(diff)==5
(ROOT/'sequence_counts.json').write_text(json.dumps(counts,indent=2)+'\n')
(ROOT/'config_changes.json').write_text(json.dumps(dict(original=str(original.resolve()),new_config=str(output.resolve()),changes=diff),indent=2)+'\n')
print(json.dumps(counts,indent=2));print(json.dumps(diff,indent=2))
