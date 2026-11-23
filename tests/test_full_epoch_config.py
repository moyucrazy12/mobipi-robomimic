"""Full-pass opt-in keeps historical defaults and supports constant scheduling."""
from types import SimpleNamespace
import unittest
import torch
from robomimic.algo.algo import Algo
from robomimic.config import config_factory
from robomimic.utils.torch_utils import lr_scheduler_from_optim_params


class FullEpochConfigTests(unittest.TestCase):
    def test_default_still_drops_partial_batches(self):
        self.assertTrue(config_factory('diffusion_policy').train.drop_last)

    def test_null_step_cap_constant_optimizer(self):
        params=dict(learning_rate=dict(initial=1e-4,scheduler_type='constant'),regularization=dict(L2=0))
        model=SimpleNamespace(optim_params={'policy':params},nets={'policy':torch.nn.Linear(2,2)},
            global_config=SimpleNamespace(train=SimpleNamespace(num_epochs=15),experiment=SimpleNamespace(epoch_every_n_steps=None)))
        Algo._create_optimizers(model)
        self.assertEqual(model.optimizers['policy'].param_groups[0]['lr'],1e-4)
        params['learning_rate']['scheduler_type']='cosine'
        with self.assertRaisesRegex(ValueError,'explicit total'):
            lr_scheduler_from_optim_params(params,model.nets['policy'],model.optimizers['policy'],None)


if __name__=='__main__':unittest.main()
