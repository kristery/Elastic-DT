import os
import random
import time

import d4rl
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from decision_transformer.model import ElasticDecisionTransformer
from decision_transformer.utils import (
    edt_evaluate,
    get_d4rl_dataset_stats,
    get_d4rl_normalized_score,
    parse,
    base_parse,
)
from scipy.interpolate import make_interp_spline
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


 
def test(args):

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = args.dataset  # medium / medium-replay / medium-expert
    rtg_scale = args.rtg_scale  # normalize returns to go
    num_bin = args.num_bin
    top_percentile = args.top_percentile
    dt_mask = args.dt_mask
    expert_weight = args.expert_weight
    exp_loss_weight = args.exp_loss_weight

    eval_dataset = args.dataset         # medium / medium-replay / medium-expert
    eval_rtg_scale = args.rtg_scale     # normalize returns to go

    if args.env == "walker2d":
        # env_name = "Walker2d-v2"
        rtg_target = 5000
        env_d4rl_name = f"walker2d-{dataset}-v2"
        env_name = env_d4rl_name

    elif args.env == "halfcheetah":
        # env_name = "HalfCheetah-v2"
        rtg_target = 6000
        env_d4rl_name = f"halfcheetah-{dataset}-v2"
        env_name = env_d4rl_name

    elif args.env == "hopper":
        # env_name = "Hopper-v2"
        rtg_target = 3600
        env_d4rl_name = f"hopper-{dataset}-v2"
        env_name = env_d4rl_name
    elif args.env == "ant":
        # env_name = "Hopper-v2"
        rtg_target = 3600 # the value does not really matter in our method
        env_d4rl_name = f"ant-{dataset}-v2"
        env_name = env_d4rl_name
    elif args.env == "antmaze-umaze":
        rtg_target = 3600 # the value does not really matter in our method
        env_d4rl_name = f"antmaze-umaze-{dataset}-v2" if dataset == "diverse" else "antmaze-umaze-v2"
        env_name = env_d4rl_name
    else:
        raise NotImplementedError

    render = args.render                # render the env frames

    num_eval_ep = args.num_eval_ep         # num of evaluation episodes
    max_eval_ep_len = args.max_eval_ep_len # max len of one episode

    max_train_iters = args.max_train_iters
    num_updates_per_iter = args.num_updates_per_iter
    mgdt_sampling = args.mgdt_sampling


    context_len = args.context_len      # K in decision transformer
    n_blocks = args.n_blocks            # num of transformer blocks
    embed_dim = args.embed_dim          # embedding (hidden) dim of transformer
    n_heads = args.n_heads              # num of transformer heads
    dropout_p = args.dropout_p          # dropout probability
    expectile = args.expectile
    rs_steps = args.rs_steps
    state_loss_weight = args.state_loss_weight
    rs_ratio = args.rs_ratio
    real_rtg = args.real_rtg

    eval_chk_pt_dir = args.chk_pt_dir

    eval_chk_pt_name = args.chk_pt_name
    eval_chk_pt_list = [eval_chk_pt_name]

    device = torch.device(args.device)
    print("device set to: ", device)

    env_data_stats = get_d4rl_dataset_stats(env_d4rl_name)
    state_mean = np.array(env_data_stats['state_mean'])
    state_std = np.array(env_data_stats['state_std'])

    env = gym.make(env_d4rl_name)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    all_scores = []

    for eval_chk_pt_name in eval_chk_pt_list:
        model = ElasticDecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            n_blocks=n_blocks,
            h_dim=embed_dim,
            context_len=context_len,
            n_heads=n_heads,
            drop_p=dropout_p,
            env_name=env_name,
            num_bin=num_bin,
            dt_mask=dt_mask,
            rtg_scale=rtg_scale,
            real_rtg=real_rtg,
        ).to(device)

        eval_chk_pt_path = os.path.join(eval_chk_pt_dir, eval_chk_pt_name)

        # load checkpoint
        model.load_state_dict(torch.load(eval_chk_pt_path, map_location=device))

        print("model loaded from: " + eval_chk_pt_path)

        # evaluate on env
        plt.figure(figsize=(10,6))
        indices_ary = []
        
        ts = time.time()
        rtn = edt_evaluate(
            model,
            device,
            context_len,
            env,
            rtg_target,
            rtg_scale,
            num_eval_ep, # number of test trials
            max_eval_ep_len,
            state_mean,
            state_std,
            top_percentile=top_percentile,
            expert_weight=expert_weight,
            num_bin=num_bin,
            env_name=env_name,
            mgdt_sampling=True,
            rs_steps=rs_steps,
            rs_ratio=rs_ratio,
            real_rtg=real_rtg,
            render=render,
            heuristic=args.heuristic,
            heuristic_delta=args.heuristic_delta,
        )
        tf = time.time()
        print(f"rs_steps: {rs_steps}")
        print("time elapsed: " + str(tf - ts))
        print(f"num_eval_ep: {num_eval_ep}, max_eval_ep_len: {max_eval_ep_len}")
        try:
            results, indices, frames = rtn
        except:
            results, indices = rtn
        print(results, get_d4rl_normalized_score(results['eval/avg_reward'], env_d4rl_name) * 100)
        

        norm_score = get_d4rl_normalized_score(results['eval/avg_reward'], env_d4rl_name) * 100
        print("normalized d4rl score: " + format(norm_score, ".5f"))

        all_scores.append(norm_score)

    print("=" * 60)
    all_scores = np.array(all_scores)
    print("evaluated on env: " + env_d4rl_name)
    print("total num of checkpoints evaluated: " + str(len(eval_chk_pt_list)))
    print("d4rl score mean: " + format(all_scores.mean(), ".5f"))
    wandb.log({
                        "eval d4rl score": norm_score,
                    })
    print("d4rl score std: " + format(all_scores.std(), ".5f"))
    print("d4rl score var: " + format(all_scores.var(), ".5f"))
    print("=" * 60)


if __name__ == "__main__":

    args = base_parse()

    predefined_keys = ["env", "dataset", "n_heads", "n_blocks", "batch_size", "num_bin", "expectile", "seed"]
    values = args.chk_pt_name.split('_')
    arg_dict = dict(zip(predefined_keys, values))

    ###
    args.env = arg_dict["env"]
    args.dataset = arg_dict["dataset"]
    args.n_heads = int(arg_dict["n_heads"])
    args.n_blocks = int(arg_dict["n_blocks"])
    args.batch_size = int(arg_dict["batch_size"])
    args.num_bin = int(float(arg_dict["num_bin"]))
    args.expectile = float(arg_dict["expectile"])
    args.seed = int(arg_dict["seed"])
    ###
    config_file = f"configs/{args.env if args.env else 'default'}.yaml"
    cfg = OmegaConf.load(config_file)
    cfg.merge_with_dotlist([f"{k}={v}" for k, v in vars(args).items() if v is not None])

    wandb.init(project=args.project_name, config=OmegaConf.to_container(cfg, resolve=True))

    test(cfg)
