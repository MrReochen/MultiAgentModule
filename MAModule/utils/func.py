import os
import yaml
import collections

from types import SimpleNamespace as SN

def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def read_yaml(args):
    with open(os.path.join(args.dir_name, "src", "config", 'default.yaml'), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            return "default.yaml error: {}".format(exc)
    with open(os.path.join(args.dir_name, "src", "config", 'sc2.yaml'), "r") as f:
        try:
            env_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            return "env.yaml error: {}".format(exc)
    with open(os.path.join(args.dir_name, "src", "config", "algs", "{}.yaml".format(args.alg)), "r") as f:
        try:
            alg_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            return "alg.yaml error: {}".format(exc)
    with open(os.path.join(args.dir_name, "src", "config", "runs", "{}.yaml".format(args.run)), "r") as f:
        try:
            run_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "run.yaml error: {}".format(exc)

    config_dict = recursive_dict_update(config_dict, env_dict)
    config_dict = recursive_dict_update(config_dict, alg_dict)
    config_dict = recursive_dict_update(config_dict, run_dict)

    if args.params != "":
        added_params = args.params.split("+")
        for param in added_params:
            k, v = param.split(":")
            if k in config_dict.keys():
                if type(config_dict[k]) == type(True):
                    if v == "True":
                        config_dict[k] = True
                    else:
                        config_dict[k] = False
                else:
                    config_dict[k] = type(config_dict[k])(v)
        config_dict["experiments_params"] = args.params
    
    config_dict["alg"] = args.alg
    config_dict["dir_name"] = args.dir_name
    config_dict["map_name"] = args.map_name
    config_dict["run"] = args.run
    config_dict["task"] = args.task
    
    return SN(**config_dict)