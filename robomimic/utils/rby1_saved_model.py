"""Saved-XML-only initialization, with asset paths relocated to this machine."""
import copy
import json
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET
import numpy as np
import robosuite
import robocasa
from robosuite.environments.base import MujocoEnv
from robocasa.environments.kitchen.kitchen import Kitchen

def local_xml(xml):
    root=ET.fromstring(xml)
    for element in root.findall('./asset/*'):
        value=element.get('file')
        if not value:continue
        for package,module in [('robocasa',robocasa),('robosuite',robosuite)]:
            token='/'+package+'/'
            if token in value:
                value=str(Path(module.__file__).parent/value.rsplit(token,1)[1]);break
        if not Path(value).is_file():raise FileNotFoundError(value)
        element.set('file',value)
    return ET.tostring(root,encoding='unicode')

def create_saved_env(metadata,group,render=False):
    xml=local_xml(group.attrs['model_file']);ep=json.loads(group.attrs['ep_meta'])
    kwargs=copy.deepcopy(metadata['env_kwargs'])
    kwargs.update(has_renderer=False,has_offscreen_renderer=render,use_camera_obs=False,ignore_done=True)
    original_load=Kitchen._load_model;original_initialize=MujocoEnv._initialize_sim
    audits=[]
    def load(self):
        self.set_ep_meta(ep);return original_load(self)
    def initialize(self,xml_string=None):
        original_initialize(self,xml_string=xml)
        model=self.sim.model._model
        assert 'robot0_world_j' not in [model.joint(j).name for j in range(model.njnt)]
        audits.append(dict(nq=model.nq,nv=model.nv,njnt=model.njnt,source='saved XML; asset paths only relocated'))
    with patch.object(Kitchen,'_load_model',load),patch.object(MujocoEnv,'_initialize_sim',initialize):
        env=robosuite.make(metadata['env_name'],**kwargs)
    env.sim.reset();env.sim.set_state_from_flattened(group['states'][0]);env.sim.forward()
    assert np.array_equal(env.sim.get_state().flatten(),group['states'][0])
    for part in env.robots[0].composite_controller.part_controllers.values():
        part.update(force=True);part.reset_goal()
    for name in ['eef_site_id','eef_cylinder_id']:
        if hasattr(env,name):env.sim.model.site_rgba[getattr(env,name)]=[0.,0.,0.,0.]
    assert env.action_dim==31
    assert env.robots[0].composite_controller_config==metadata['env_kwargs']['controller_configs']
    ik=env.robots[0].composite_controller.joint_action_policy;model=env.sim.model._model
    mapping=[dict(name=n,joint_id=int(model.joint(n).id),qpos=int(model.jnt_qposadr[model.joint(n).id]),dof=int(model.jnt_dofadr[model.joint(n).id])) for n in ik.joint_names]
    assert all(x['joint_id']==x['qpos']==x['dof'] for x in mapping)
    return env,dict(physics_compilations=audits,left_joint_mapping=mapping,saved_initial_state_exact=True)
