import os
import shutil
import json
import torch
import numpy as np
from PyBark import Bark
from pathlib import Path
from datetime import datetime
from os.path import dirname, abspath
from types import SimpleNamespace as SN

from ..utils import *

from ..envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv

class Experiment:
    def __init__(self, args, train_alg=None, policy=None):
        self.args = self.loading_args(args)
        self.setup_params()
        self.setup_notification()
        self.setup_seed(self.args.seed)
        self.setup_cuda()
        self.setup_logging()
        self.setup_envs()
        self.setup_runner(train_alg, policy)
    
    def setup_notification(self):
        if hasattr(self.args, "bark_token"):
            self.bark = Bark(self.args.bark_token)
        else:
            self.bark = None
        
    def loading_args(self, args):
        res = read_yaml(args)
        if isinstance(res, str):
            self.notification("Exp Error", res)
        
        return res

    def setup_params(self):
        self.time_token = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        self.base_dir = self.args.dir_name
        self.task = self.args.task
        self.purpose = self.args.purpose
        if self.task == "sc2":
            self.sc2_mapname = self.args.map_name
        self.alg_name = self.args.alg
        self.run_name = self.args.run

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def setup_cuda(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device("cpu")

        torch.set_num_threads(self.args.n_training_threads)

    def setup_logging(self):
        if self.task == "sc2":
            self.run_dir = Path(self.base_dir) / "results" / self.task / self.purpose / self.sc2_mapname / self.alg_name / self.run_name / self.time_token
        else:
            self.run_dir = Path(self.base_dir) / "results" / self.task / self.purpose / self.alg_name / self.run_name / self.time_token
        if not self.run_dir.exists():
            os.makedirs(str(self.run_dir))
        
        os.makedirs(os.path.join(self.run_dir, "code_backup"))
        zip_file_path(os.path.join(self.base_dir, "src"), os.path.join(self.run_dir, "code_backup"), "code.zip")
        with open(os.path.join(self.run_dir, "code_backup", "params.json"), "w") as f:
            json.dump(self.args.__dict__, f)

    def setup_envs(self):
        if self.task == "sc2":
            from ..envs.starcraft2.StarCraft2_Env import StarCraft2Env
            
            def make_env(Env, all_args, seed_amp, rank_amp, n_rollout):
                def get_env_fn(rank):
                    def init_env():
                        env = Env(all_args)
                        env.seed(all_args.seed * seed_amp + rank * rank_amp)
                        return env
                    return init_env
                if n_rollout == 1:
                    return ShareDummyVecEnv([get_env_fn(0)])
                else:
                    return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_rollout)])

            self.envs = make_env(StarCraft2Env, self.args, 1, 1000, self.args.n_rollout_threads)
            self.eval_envs = make_env(StarCraft2Env, self.args, 50000, 10000, self.args.n_eval_rollout_threads)

    def after_run(self):
        try:
            self.envs.close()
            self.eval_envs.close()
        except:
            self.notification("Exp Error", "Close Envs Error")
        self.notification("Exp Done", self.args.experiments_params)

    def notification(self, title, content):
        if self.bark is not None:
            self.bark.send(content, title=title)

    def setup_runner(self, train_alg=None, policy=None):
        from ..envs.starcraft2.smac_maps import get_map_params
        config = {
            "all_args": self.args,
            "envs": self.envs,
            "eval_envs": self.eval_envs,
            "num_agents": get_map_params(self.args.map_name)["n_agents"],
            "device": self.device,
            "run_dir": self.run_dir,
            "train_alg": train_alg,
            "policy": policy
        }
        from ..runner.smac_runner import SMACRunner as Runner
        self.runner = Runner(config)

    def training(self):
        try:
            self.runner.run()
        except:
            self.notification("Exp Error", "Please check {}".format(self.args.experiments_params))
        finally:
            self.after_run()