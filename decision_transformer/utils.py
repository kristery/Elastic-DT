import argparse
import math
import os
import pickle
import random
import time
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from decision_transformer.d4rl_infos import (
    D4RL_DATASET_STATS,
    REF_MAX_SCORE,
    REF_MIN_SCORE,
)
from torch.distributions.categorical import Categorical
from torch.utils.data import Dataset
from omegaconf import OmegaConf


def base_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="medium-replay")
    parser.add_argument("--rtg_scale", type=int, default=1000)

    parser.add_argument("--max_eval_ep_len", type=int, default=1000)
    parser.add_argument("--num_eval_ep", type=int, default=20)

    parser.add_argument("--dataset_dir", type=str, default="data/")
    parser.add_argument("--log_dir", type=str, default="dt_runs/")

    parser.add_argument("--context_len", type=int, default=20)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument(
        "--embed_dim", type=int, default=512
    )  # better to set it n_heads * 20
    parser.add_argument("--n_heads", type=int, default=4) 
    parser.add_argument("--dropout_p", type=float, default=0.1)

    # edt
    parser.add_argument("--ce_weight", type=float, default=0.001)
    parser.add_argument("--num_bin", type=float, default=60)
    parser.add_argument("--top_percentile", type=float, default=0.15)
    parser.add_argument("--dt_mask", action="store_true")

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wt_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)

    parser.add_argument("--max_train_iters", type=int, default=500)
    parser.add_argument("--num_updates_per_iter", type=int, default=100)
    parser.add_argument("--expert_weight", type=float, default=None)
    parser.add_argument("--mgdt_sampling", action="store_true")
    parser.add_argument("--expectile", type=float, default=0.99)

    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eval_iters", type=int, default=250)
    parser.add_argument("--rs_steps", type=int, default=2)
    parser.add_argument("--state_loss_weight", type=float, default=1.0)
    parser.add_argument("--exp_loss_weight", type=float, default=0.5)
    parser.add_argument("--rs_ratio", type=float, default=1, help="value betwee 1 and 2")
    parser.add_argument("--real_rtg", action="store_true")
    parser.add_argument("--data_ratio", type=float, default=1.0)
    parser.add_argument("--project_name", type=str, default="EDT-experiments")
    parser.add_argument("--model_save_iters", type=int, default=250) 

    parser.add_argument('--chk_pt_dir', type=str, default='dt_runs/')
    parser.add_argument('--chk_pt_name', type=str,
            default='edt_hopper-medium-replay-v2_model_23-03-06-01-28-21_best.pt')
    parser.add_argument('--render', default=False)
    parser.add_argument('--heuristic', default=False)
    parser.add_argument('--heuristic_delta', type=int, default=1)

    args = parser.parse_args()

    return args


def parse():
    args = base_parse()

    # omegaconf
    config_file = f"configs/{args.env if args.env else 'default'}.yaml"
    cfg = OmegaConf.load(config_file)
    cfg.merge_with_dotlist([f"{k}={v}" for k, v in vars(args).items() if v is not None])

    return cfg




class ModelSaver:
    def __init__(self, args):
        # self.args = vars(args)
        self.args = args
        self.best_error = float('inf')  # Initialize with infinity
        start_time = datetime.now().replace(microsecond=0)
        self.timestamp = start_time.strftime("%y-%m-%d-%H-%M-%S")
        self.dir = args.chk_pt_dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.predefined_keys = ["env", "dataset", "n_heads", "n_blocks", "batch_size", "num_bin", "expectile", "seed"]

    def generate_filename(self, epoch, is_best=False):
        filename_parts = [str(self.args[key]) for key in self.predefined_keys]
        filename_parts.append(str(self.timestamp))
        if is_best:
            filename_parts.append('best')
        else:
            filename_parts.append(str(epoch))
        filename = "_".join(filename_parts) + ".pt"
        return os.path.join(self.dir, filename)

    def save_model(self, model, epoch, error=None):
        # Always save the model at the given epoch
        filename = self.generate_filename(epoch)
        torch.save(model.state_dict(), filename)

        # If this model has lower error, save it as the best model
        if error is not None and error < self.best_error:
            self.best_error = error
            best_filename = self.generate_filename(epoch, is_best=True)
            torch.save(model.state_dict(), best_filename)


def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t + 1]
    return disc_cumsum


def get_d4rl_normalized_score(score, env_name):
    env_key = env_name.split("-")[0].lower()
    assert (
        env_key in REF_MAX_SCORE
    ), f"no reference score for {env_key} env to calculate d4rl score"
    return (score - REF_MIN_SCORE[env_key]) / (
        REF_MAX_SCORE[env_key] - REF_MIN_SCORE[env_key]
    )


def get_d4rl_dataset_stats(env_d4rl_name):
    return D4RL_DATASET_STATS[env_d4rl_name]


def encode_return(env_name, ret, scale=1.0, num_bin=120, rtg_scale=1000):
    env_key = env_name.split("-")[0].lower()
    if env_key not in REF_MAX_SCORE:
        ret_max = 100
    else:
        ret_max = REF_MAX_SCORE[env_key]
    if env_key not in REF_MIN_SCORE:
        ret_min = -20
    else:
        ret_min = REF_MIN_SCORE[env_key]
    ret_max /= rtg_scale
    ret_min /= rtg_scale
    interval = (ret_max - ret_min) / (num_bin-1)
    ret = torch.clip(ret, ret_min, ret_max)
    return ((ret - ret_min) // interval).float()


def decode_return(env_name, ret, scale=1.0, num_bin=120, rtg_scale=1000):
    env_key = env_name.split("-")[0].lower()
    if env_key not in REF_MAX_SCORE:
        ret_max = 100
    else:
        ret_max = REF_MAX_SCORE[env_key]
    if env_key not in REF_MIN_SCORE:
        ret_min = -20
    else:
        ret_min = REF_MIN_SCORE[env_key]
    ret_max /= rtg_scale
    ret_min /= rtg_scale
    interval = (ret_max - ret_min) / num_bin
    return ret * interval + ret_min


def sample_from_logits(
    logits: torch.Tensor,
    temperature: Optional[float] = 1e0,
    top_percentile: Optional[float] = None,
) -> torch.Tensor:

    if top_percentile is not None:
        percentile = torch.quantile(
            logits, top_percentile, axis=-1, keepdim=True
        )
        logits = torch.where(logits >= percentile, logits, -np.inf)
    m = Categorical(logits=temperature * logits)
    return m.sample().unsqueeze(-1)


def expert_sampling(
    logits: torch.Tensor,
    temperature: Optional[float] = 1e0,
    top_percentile: Optional[float] = None,
    expert_weight: Optional[float] = 10,
) -> torch.Tensor:
    B, T, num_bin = logits.shape
    expert_logits = (
        torch.linspace(0, 1, num_bin).repeat(B, T, 1).to(logits.device)
    )
    return sample_from_logits(
        logits + expert_weight * expert_logits, temperature, top_percentile
    )


def mgdt_logits(
    logits: torch.Tensor, opt_weight: Optional[int] = 10
) -> torch.Tensor:
    logits_opt = torch.linspace(0.0, 1.0, logits.shape[-1]).to(logits.device)
    logits_opt = logits_opt.repeat(logits.shape[1], 1).unsqueeze(0)
    return logits + opt_weight * logits_opt


def edt_evaluate(
    model,
    device,
    context_len,
    env,
    rtg_target,
    rtg_scale,
    num_eval_ep=10,
    max_test_ep_len=1000,
    state_mean=None,
    state_std=None,
    render=False,
    top_percentile=0.15,
    expert_weight=10.0,
    num_bin=120,
    env_name=None,
    mgdt_sampling=False,
    rs_steps=2,
    rs_ratio=1,
    real_rtg=False,
    heuristic=False,
    heuristic_delta=1,
    *args, 
    **kwargs
):

    eval_batch_size = 1  # required for forward pass

    indices = []
    frames = []
    results = {}
    total_reward = 0
    total_timesteps = 0

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    if state_mean is None:
        state_mean = torch.zeros((state_dim,)).to(device)
    else:
        state_mean = torch.from_numpy(state_mean).to(device)

    if state_std is None:
        state_std = torch.ones((state_dim,)).to(device)
    else:
        state_std = torch.from_numpy(state_std).to(device)

    # same as timesteps used for training the transformer
    # also, crashes if device is passed to arange()
    timesteps = torch.arange(
        start=0, end=max_test_ep_len + 2 * context_len, step=1
    )
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    model.eval()

    with torch.no_grad():

        for _ in range(num_eval_ep):

            # zeros place holders
            actions = torch.zeros(
                (eval_batch_size, max_test_ep_len + 2 * context_len, act_dim),
                dtype=torch.float32,
                device=device,
            )
            states = torch.zeros(
                (
                    eval_batch_size,
                    max_test_ep_len + 2 * context_len,
                    state_dim,
                ),
                dtype=torch.float32,
                device=device,
            )
            rewards_to_go = torch.zeros(
                (eval_batch_size, max_test_ep_len + 2 * context_len, 1),
                dtype=torch.float32,
                device=device,
            )
            rewards = torch.zeros(
                (eval_batch_size, max_test_ep_len + 2 * context_len, 1),
                dtype=torch.float32,
                device=device,
            )

            # init episode
            running_state = env.reset()
            running_reward = 0
            running_rtg = rtg_target / rtg_scale

            previous_index = None
            for t in range(max_test_ep_len):
                total_timesteps += 1

                # add state in placeholder and normalize
                states[0, t] = torch.from_numpy(running_state).to(device)
                states[0, t] = (states[0, t] - state_mean) / state_std

                # calcualate running rtg and add it in placeholder
                running_rtg = running_rtg - (running_reward / rtg_scale)
                rewards_to_go[0, t] = running_rtg
                rewards[0, t] = running_reward

                if not heuristic:
                    act, best_index = _return_search(
                        model=model,
                        timesteps=timesteps,
                        states=states,
                        actions=actions,
                        rewards_to_go=rewards_to_go,
                        rewards=rewards,
                        context_len=context_len,
                        t=t,
                        env_name=env_name,
                        top_percentile=top_percentile,
                        num_bin=num_bin,
                        rtg_scale=rtg_scale,
                        expert_weight=expert_weight,
                        mgdt_sampling=mgdt_sampling,
                        rs_steps=rs_steps,
                        rs_ratio=rs_ratio,
                        real_rtg=real_rtg
                    )
                else:
                    act, best_index = _return_search_heuristic(
                        model=model,
                        timesteps=timesteps,
                        states=states,
                        actions=actions,
                        rewards_to_go=rewards_to_go,
                        rewards=rewards,
                        context_len=context_len,
                        t=t,
                        env_name=env_name,
                        top_percentile=top_percentile,
                        num_bin=num_bin,
                        rtg_scale=rtg_scale,
                        expert_weight=expert_weight,
                        mgdt_sampling=mgdt_sampling,
                        rs_steps=rs_steps,
                        rs_ratio=rs_ratio,
                        real_rtg=real_rtg,
                        heuristic_delta=heuristic_delta,
                        previous_index=previous_index,
                    )
                    previous_index = best_index
                indices.append(best_index)


                running_state, running_reward, done, _ = env.step(
                    act.cpu().numpy()
                )

                # add action in placeholder
                actions[0, t] = act

                total_reward += running_reward

                if render:
                    frames.append(env.render(mode="rgb_array"))
                if done:
                    break

    results["eval/avg_reward"] = total_reward / num_eval_ep
    results["eval/avg_ep_len"] = total_timesteps / num_eval_ep

    if render:
        return results, indices, frames

    return results, indices


def _return_search(
    model,
    timesteps,
    states,
    actions,
    rewards_to_go,
    rewards,
    context_len,
    t,
    env_name,
    top_percentile,
    num_bin,
    rtg_scale,
    expert_weight,
    mgdt_sampling=False,
    rs_steps=2,
    rs_ratio=1,
    real_rtg=False,
    *args, 
    **kwargs
):

    # B x T x 1?
    highest_ret = -999
    estimated_rtg = None
    best_i = 0
    best_act = None

    if t < context_len:
        for i in range(0, math.ceil((t + 1)/rs_ratio), rs_steps):
            _, act_preds, ret_preds, imp_ret_preds, _ = model.forward(
                timesteps[:, i : context_len + i],
                states[:, i : context_len + i],
                actions[:, i : context_len + i],
                rewards_to_go[:, i : context_len + i],
                rewards[:, i : context_len + i],
            )

            # first sample return with optimal weight
            # this sampling is the same as mgdt
            if mgdt_sampling:
                opt_rtg = decode_return(
                    env_name,
                    expert_sampling(
                        mgdt_logits(ret_preds),
                        top_percentile=top_percentile,
                        expert_weight=expert_weight,
                    ),
                    num_bin=num_bin,
                    rtg_scale=rtg_scale,
                )

                # we should estimate it again with the estimated rtg
                _, act_preds, ret_preds, imp_ret_preds_pure, _ = model.forward(
                    timesteps[:, i : context_len + i],
                    states[:, i : context_len + i],
                    actions[:, i : context_len + i],
                    opt_rtg,
                    rewards[:, i : context_len + i],
                )

            else:
                _, act_preds, ret_preds, imp_ret_preds_pure, _ = model.forward(
                    timesteps[:, i : context_len + i],
                    states[:, i : context_len + i],
                    actions[:, i : context_len + i],
                    imp_ret_preds,
                    rewards[:, i : context_len + i],
                )

            if not real_rtg:
                imp_ret_preds = imp_ret_preds_pure

            ret_i = imp_ret_preds[:, t - i].detach().item()
            if ret_i > highest_ret:
                highest_ret = ret_i
                best_i = i
                estimated_rtg = imp_ret_preds.detach()
                best_act = act_preds[0, t - i].detach()


    else:
        for i in range(0, math.ceil(context_len/rs_ratio), rs_steps):
            _, act_preds, ret_preds, imp_ret_preds, _ = model.forward(
                timesteps[:, t - context_len + 1 + i : t + 1 + i],
                states[:, t - context_len + 1 + i : t + 1 + i],
                actions[:, t - context_len + 1 + i : t + 1 + i],
                rewards_to_go[:, t - context_len + 1 + i : t + 1 + i],
                rewards[:, t - context_len + 1 + i : t + 1 + i],
            )

            # first sample return with optimal weight
            if mgdt_sampling:
                opt_rtg = decode_return(
                    env_name,
                    expert_sampling(
                        mgdt_logits(ret_preds),
                        top_percentile=top_percentile,
                        expert_weight=expert_weight,
                    ),
                    num_bin=num_bin,
                    rtg_scale=rtg_scale,
                )

                # we should estimate the results again with the estimated return
                _, act_preds, ret_preds, imp_ret_preds_pure, _ = model.forward(
                    timesteps[:, t - context_len + 1 + i : t + 1 + i],
                    states[:, t - context_len + 1 + i : t + 1 + i],
                    actions[:, t - context_len + 1 + i : t + 1 + i],
                    opt_rtg,
                    rewards[:, t - context_len + 1 + i : t + 1 + i],
                )

            else:
                _, act_preds, ret_preds, imp_ret_preds_pure, _ = model.forward(
                    timesteps[:, t - context_len + 1 + i : t + 1 + i],
                    states[:, t - context_len + 1 + i : t + 1 + i],
                    actions[:, t - context_len + 1 + i : t + 1 + i],
                    imp_ret_preds,
                    rewards[:, t - context_len + 1 + i : t + 1 + i],
                )

            if not real_rtg:
                imp_ret_preds = imp_ret_preds_pure

            ret_i = imp_ret_preds[:, -1 - i].detach().item()
            if ret_i > highest_ret:
                highest_ret = ret_i
                best_i = i
                # estimated_rtg = imp_ret_preds.detach()
                best_act = act_preds[0, -1 - i].detach()

        # _, act_preds, _, _, _ = model.forward(
        #     timesteps[:, t - context_len + 1 + best_i : t + 1 + best_i],
        #     states[:, t - context_len + 1 + best_i : t + 1 + best_i],
        #     actions[:, t - context_len + 1 + best_i : t + 1 + best_i],
        #     estimated_rtg,
        #     rewards[:, t - context_len + 1 + best_i : t + 1 + best_i],
        # )

    return best_act, context_len - best_i


def _return_search_heuristic(
    model,
    timesteps,
    states,
    actions,
    rewards_to_go,
    rewards,
    context_len,
    t,
    env_name,
    top_percentile,
    num_bin,
    rtg_scale,
    expert_weight,
    mgdt_sampling=False,
    rs_steps=2,
    rs_ratio=1,
    real_rtg=False,
    heuristic_delta=1,
    previous_index=None,
    *args, 
    **kwargs
):

    # B x T x 1?
    highest_ret = -999
    estimated_rtg = None
    best_i = 0
    best_act = None

    if t < context_len:
        for i in range(0, math.ceil((t + 1)/rs_ratio), rs_steps):
            _, act_preds, ret_preds, imp_ret_preds, _ = model.forward(
                timesteps[:, i : context_len + i],
                states[:, i : context_len + i],
                actions[:, i : context_len + i],
                rewards_to_go[:, i : context_len + i],
                rewards[:, i : context_len + i],
            )

            if mgdt_sampling:
                opt_rtg = decode_return(
                    env_name,
                    expert_sampling(
                        mgdt_logits(ret_preds),
                        top_percentile=top_percentile,
                        expert_weight=expert_weight,
                    ),
                    num_bin=num_bin,
                    rtg_scale=rtg_scale,
                )

                # we should estimate it again with the estimated rtg
                _, act_preds, ret_preds, imp_ret_preds_pure, _ = model.forward(
                    timesteps[:, i : context_len + i],
                    states[:, i : context_len + i],
                    actions[:, i : context_len + i],
                    opt_rtg,
                    rewards[:, i : context_len + i],
                )

            else:
                _, act_preds, ret_preds, imp_ret_preds_pure, _ = model.forward(
                    timesteps[:, i : context_len + i],
                    states[:, i : context_len + i],
                    actions[:, i : context_len + i],
                    imp_ret_preds,
                    rewards[:, i : context_len + i],
                )

            if not real_rtg:
                imp_ret_preds = imp_ret_preds_pure

            ret_i = imp_ret_preds[:, t - i].detach().item()
            if ret_i > highest_ret:
                highest_ret = ret_i
                best_i = i
                estimated_rtg = imp_ret_preds.detach()
                best_act = act_preds[0, t - i].detach()


    else: # t >= context_len
        prev_best_index = context_len - previous_index

        for i in range(prev_best_index-heuristic_delta, prev_best_index+1+heuristic_delta):
            if i < 0 or i >= context_len:
                continue
            _, act_preds, ret_preds, imp_ret_preds, _ = model.forward(
                timesteps[:, t - context_len + 1 + i : t + 1 + i],
                states[:, t - context_len + 1 + i : t + 1 + i],
                actions[:, t - context_len + 1 + i : t + 1 + i],
                rewards_to_go[:, t - context_len + 1 + i : t + 1 + i],
                rewards[:, t - context_len + 1 + i : t + 1 + i],
            )

            # first sample return with optimal weight
            if mgdt_sampling:
                opt_rtg = decode_return(
                    env_name,
                    expert_sampling(
                        mgdt_logits(ret_preds),
                        top_percentile=top_percentile,
                        expert_weight=expert_weight,
                    ),
                    num_bin=num_bin,
                    rtg_scale=rtg_scale,
                )

                # we should estimate the results again with the estimated return
                _, act_preds, ret_preds, imp_ret_preds_pure, _ = model.forward(
                    timesteps[:, t - context_len + 1 + i : t + 1 + i],
                    states[:, t - context_len + 1 + i : t + 1 + i],
                    actions[:, t - context_len + 1 + i : t + 1 + i],
                    opt_rtg,
                    rewards[:, t - context_len + 1 + i : t + 1 + i],
                )

            else:
                _, act_preds, ret_preds, imp_ret_preds_pure, _ = model.forward(
                    timesteps[:, t - context_len + 1 + i : t + 1 + i],
                    states[:, t - context_len + 1 + i : t + 1 + i],
                    actions[:, t - context_len + 1 + i : t + 1 + i],
                    imp_ret_preds,
                    rewards[:, t - context_len + 1 + i : t + 1 + i],
                )

            if not real_rtg:
                imp_ret_preds = imp_ret_preds_pure

            ret_i = imp_ret_preds[:, -1 - i].detach().item()
            if ret_i > highest_ret:
                highest_ret = ret_i
                best_i = i
                # estimated_rtg = imp_ret_preds.detach()
                best_act = act_preds[0, -1 - i].detach()

    return best_act, context_len - best_i


def evaluate_on_env(
    model,
    device,
    context_len,
    env,
    rtg_target,
    rtg_scale,
    num_eval_ep=10,
    max_test_ep_len=1000,
    state_mean=None,
    state_std=None,
    render=False,
):

    eval_batch_size = 1  # required for forward pass

    results = {}
    total_reward = 0
    total_timesteps = 0

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    if state_mean is None:
        state_mean = torch.zeros((state_dim,)).to(device)
    else:
        state_mean = torch.from_numpy(state_mean).to(device)

    if state_std is None:
        state_std = torch.ones((state_dim,)).to(device)
    else:
        state_std = torch.from_numpy(state_std).to(device)

    # same as timesteps used for training the transformer
    # also, crashes if device is passed to arange()
    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    model.eval()

    with torch.no_grad():

        for _ in range(num_eval_ep):

            # zeros place holders
            actions = torch.zeros(
                (eval_batch_size, max_test_ep_len, act_dim),
                dtype=torch.float32,
                device=device,
            )
            states = torch.zeros(
                (eval_batch_size, max_test_ep_len, state_dim),
                dtype=torch.float32,
                device=device,
            )
            rewards_to_go = torch.zeros(
                (eval_batch_size, max_test_ep_len, 1),
                dtype=torch.float32,
                device=device,
            )

            # init episode
            running_state = env.reset()
            running_reward = 0
            running_rtg = rtg_target / rtg_scale

            for t in range(max_test_ep_len):

                total_timesteps += 1

                # add state in placeholder and normalize
                states[0, t] = torch.from_numpy(running_state).to(device)
                states[0, t] = (states[0, t] - state_mean) / state_std

                # calcualate running rtg and add it in placeholder
                running_rtg = running_rtg - (running_reward / rtg_scale)
                rewards_to_go[0, t] = running_rtg

                if t < context_len:
                    _, act_preds, _ = model.forward(
                        timesteps[:, :context_len],
                        states[:, :context_len],
                        actions[:, :context_len],
                        rewards_to_go[:, :context_len],
                    )
                    act = act_preds[0, t].detach()
                else:
                    _, act_preds, _ = model.forward(
                        timesteps[:, t - context_len + 1 : t + 1],
                        states[:, t - context_len + 1 : t + 1],
                        actions[:, t - context_len + 1 : t + 1],
                        rewards_to_go[:, t - context_len + 1 : t + 1],
                    )
                    act = act_preds[0, -1].detach()

                running_state, running_reward, done, _ = env.step(
                    act.cpu().numpy()
                )

                # add action in placeholder
                actions[0, t] = act

                total_reward += running_reward

                if render:
                    env.render()
                if done:
                    break

    results["eval/avg_reward"] = total_reward / num_eval_ep
    results["eval/avg_ep_len"] = total_timesteps / num_eval_ep

    return results


class D4RLTrajectoryDataset(Dataset):
    def __init__(self, dataset_path, context_len, rtg_scale, data_ratio=1.0):

        self.context_len = context_len

        # load dataset
        with open(dataset_path, "rb") as f:
            self.trajectories = pickle.load(f)
            
            size = len(self.trajectories)
            if data_ratio < 1.0:
                new_size = int(size * data_ratio)
                self.trajectories = self.trajectories[:new_size]

        # calculate min len of traj, state mean and variance
        # and returns_to_go for all traj
        min_len = 10**6
        states = []
        for traj in self.trajectories:
            traj_len = traj["observations"].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj["observations"])
            # calculate returns to go and rescale them
            traj["returns_to_go"] = (
                discount_cumsum(traj["rewards"], 1.0) / rtg_scale
            )

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = (
            np.mean(states, axis=0),
            np.std(states, axis=0) + 1e-6,
        )

        # normalize states
        for traj in self.trajectories:
            traj["observations"] = (
                traj["observations"] - self.state_mean
            ) / self.state_std

    def get_state_stats(self):
        return self.state_mean, self.state_std

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj["observations"].shape[0]

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(
                traj["observations"][si : si + self.context_len]
            )
            actions = torch.from_numpy(
                traj["actions"][si : si + self.context_len]
            )
            returns_to_go = torch.from_numpy(
                traj["returns_to_go"][si : si + self.context_len]
            )
            timesteps = torch.arange(
                start=si, end=si + self.context_len, step=1
            )

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj["observations"])
            states = torch.cat(
                [
                    states,
                    torch.zeros(
                        ([padding_len] + list(states.shape[1:])),
                        dtype=states.dtype,
                    ),
                ],
                dim=0,
            )

            actions = torch.from_numpy(traj["actions"])
            actions = torch.cat(
                [
                    actions,
                    torch.zeros(
                        ([padding_len] + list(actions.shape[1:])),
                        dtype=actions.dtype,
                    ),
                ],
                dim=0,
            )

            returns_to_go = torch.from_numpy(traj["returns_to_go"])
            returns_to_go = torch.cat(
                [
                    returns_to_go,
                    torch.zeros(
                        ([padding_len] + list(returns_to_go.shape[1:])),
                        dtype=returns_to_go.dtype,
                    ),
                ],
                dim=0,
            )

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat(
                [
                    torch.ones(traj_len, dtype=torch.long),
                    torch.zeros(padding_len, dtype=torch.long),
                ],
                dim=0,
            )

        return timesteps, states, actions, returns_to_go, traj_mask


class EDTTrajectoryDataset(D4RLTrajectoryDataset):
    def __init__(self, dataset_path, context_len, rtg_scale, data_ratio=1.0):

        self.context_len = context_len

        # load dataset
        with open(dataset_path, "rb") as f:
            self.trajectories = pickle.load(f)
            
            size = len(self.trajectories)
            if data_ratio < 1.0:
                new_size = int(size * data_ratio)
                self.trajectories = self.trajectories[:new_size]


        # calculate min len of traj, state mean and variance
        # and returns_to_go for all traj
        min_len = 10**6
        states = []
        for traj in self.trajectories:
            traj_len = traj["observations"].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj["observations"])
            # calculate returns to go and rescale them
            traj["returns_to_go"] = (
                discount_cumsum(traj["rewards"], 1.0) / rtg_scale
            )

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = (
            np.mean(states, axis=0),
            np.std(states, axis=0) + 1e-6,
        )

        # normalize states
        print(f"num of trajs: {len(self.trajectories)}")
        for traj in self.trajectories:
            traj["observations"] = (
                traj["observations"] - self.state_mean
            ) / self.state_std
            traj["next_observations"] = (
                traj["next_observations"] - self.state_mean
            ) / self.state_std

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj["observations"].shape[0]

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(
                traj["observations"][si : si + self.context_len]
            )
            next_states = torch.from_numpy(
                traj["next_observations"][si : si + self.context_len]
            )
            actions = torch.from_numpy(
                traj["actions"][si : si + self.context_len]
            )
            returns_to_go = torch.from_numpy(
                traj["returns_to_go"][si : si + self.context_len]
            )
            rewards = torch.from_numpy(
                traj["rewards"][si : si + self.context_len]
            )
            timesteps = torch.arange(
                start=si, end=si + self.context_len, step=1
            )

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj["observations"])
            states = torch.cat(
                [
                    states,
                    torch.zeros(
                        ([padding_len] + list(states.shape[1:])),
                        dtype=states.dtype,
                    ),
                ],
                dim=0,
            )

            next_states = torch.from_numpy(traj["next_observations"])
            next_states = torch.cat(
                [
                    next_states,
                    torch.zeros(
                        ([padding_len] + list(next_states.shape[1:])),
                        dtype=next_states.dtype,
                    ),
                ],
                dim=0,
            )

            actions = torch.from_numpy(traj["actions"])
            actions = torch.cat(
                [
                    actions,
                    torch.zeros(
                        ([padding_len] + list(actions.shape[1:])),
                        dtype=actions.dtype,
                    ),
                ],
                dim=0,
            )

            returns_to_go = torch.from_numpy(traj["returns_to_go"])
            returns_to_go = torch.cat(
                [
                    returns_to_go,
                    torch.zeros(
                        ([padding_len] + list(returns_to_go.shape[1:])),
                        dtype=returns_to_go.dtype,
                    ),
                ],
                dim=0,
            )

            rewards = torch.from_numpy(traj["rewards"])
            rewards = torch.cat(
                [
                    rewards,
                    torch.zeros(
                        ([padding_len] + list(rewards.shape[1:])),
                        dtype=rewards.dtype,
                    ),
                ],
                dim=0,
            )

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat(
                [
                    torch.ones(traj_len, dtype=torch.long),
                    torch.zeros(padding_len, dtype=torch.long),
                ],
                dim=0,
            )

        return (
            timesteps,
            states,
            next_states,
            actions,
            returns_to_go,
            rewards,
            traj_mask,
        )

