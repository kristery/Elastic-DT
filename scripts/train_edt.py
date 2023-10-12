import os
import random
from datetime import datetime

import d4rl
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from decision_transformer.model import ElasticDecisionTransformer
from decision_transformer.utils import (
    EDTTrajectoryDataset,
    ModelSaver,
    encode_return,
    parse,
)
from torch.utils.data import DataLoader
from omegaconf import OmegaConf


def train(args):

    scaler = torch.cuda.amp.GradScaler()
    model_saver = ModelSaver(args)

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = args.dataset  # medium / medium-replay
    rtg_scale = args.rtg_scale  # normalize returns to go
    num_bin = args.num_bin
    top_percentile = args.top_percentile
    dt_mask = args.dt_mask
    expert_weight = args.expert_weight
    exp_loss_weight = args.exp_loss_weight

    if args.env == "walker2d":
        rtg_target = 5000
        env_d4rl_name = f"walker2d-{dataset}-v2"
        env_name = env_d4rl_name
    elif args.env == "halfcheetah":
        rtg_target = 6000
        env_d4rl_name = f"halfcheetah-{dataset}-v2"
        env_name = env_d4rl_name
    elif args.env == "hopper":
        rtg_target = 3600
        env_d4rl_name = f"hopper-{dataset}-v2"
        env_name = env_d4rl_name
    elif args.env == "ant":
        rtg_target = 3600
        env_d4rl_name = f"ant-{dataset}-v2"
        env_name = env_d4rl_name
    else:
        raise NotImplementedError

    max_eval_ep_len = args.max_eval_ep_len  # max len of one episode
    num_eval_ep = args.num_eval_ep  # num of evaluation episodes

    batch_size = args.batch_size  # training batch size
    lr = args.lr  # learning rate
    wt_decay = args.wt_decay  # weight decay
    warmup_steps = args.warmup_steps  # warmup steps for lr scheduler

    # total updates = max_train_iters x num_updates_per_iter
    max_train_iters = args.max_train_iters
    num_updates_per_iter = args.num_updates_per_iter
    mgdt_sampling = args.mgdt_sampling

    context_len = args.context_len  # K in decision transformer
    n_blocks = args.n_blocks  # num of transformer blocks
    embed_dim = args.embed_dim  # embedding (hidden) dim of transformer
    n_heads = args.n_heads  # num of transformer heads
    dropout_p = args.dropout_p  # dropout probability
    expectile = args.expectile
    rs_steps = args.rs_steps
    state_loss_weight = args.state_loss_weight
    rs_ratio = args.rs_ratio
    real_rtg = args.real_rtg
    data_ratio = args.data_ratio

    eval_d4rl_score_mson = None

    # load data from this file
    dataset_path_u = os.path.join(args.dataset_dir, f"{env_d4rl_name}.pkl")

    # saves model and csv in this directory
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # training and evaluation device
    device = torch.device(args.device)

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    prefix = "edt_" + env_d4rl_name + f"_{args.seed}"

    save_model_name = prefix + "_model_" + start_time_str + ".pt"
    save_model_path = os.path.join(log_dir, save_model_name)

    log_csv_name = prefix + "_log_" + start_time_str + ".csv"
    log_csv_path = os.path.join(log_dir, log_csv_name)


    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)

    print("device set to: " + str(device))
    print("dataset path: " + dataset_path_u)
    print("model save path: " + save_model_path)
    print("log csv save path: " + log_csv_path)

    traj_dataset_u = EDTTrajectoryDataset(
        dataset_path_u, context_len, rtg_scale, data_ratio=data_ratio
    )

    traj_data_loader_u = DataLoader(
        traj_dataset_u,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    data_iter_u = iter(traj_data_loader_u)

    ## get state stats from dataset
    state_mean, state_std = traj_dataset_u.get_state_stats()

    env = gym.make(env_name)
    env.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wt_decay
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    total_updates = 0

    for i_train_iter in range(1, max_train_iters+1):

        log_action_losses = []
        log_state_losses = []
        log_exp_losses = []
        ret_ce_losses = []
        model.train()

        for _ in range(num_updates_per_iter):
            try:
                (
                    timesteps_u,
                    states_u,
                    next_states_u,
                    actions_u,
                    returns_to_go_u,
                    rewards_u,
                    traj_mask_u,
                ) = next(data_iter_u)
            except StopIteration:
                data_iter_u = iter(traj_data_loader_u)
                (
                    timesteps_u,
                    states_u,
                    next_states_u,
                    actions_u,
                    returns_to_go_u,
                    rewards_u,
                    traj_mask_u,
                ) = next(data_iter_u)

            timesteps_u = timesteps_u.to(device)  # B x T
            states_u = states_u.to(device)  # B x T x state_dim
            next_states_u = next_states_u.to(device)
            actions_u = actions_u.to(device)  # B x T x act_dim
            returns_to_go_u = returns_to_go_u.to(device).unsqueeze(
                dim=-1
            )  # B x T x 1
            rewards_u = rewards_u.to(device).unsqueeze(dim=-1)  # B x T x 1
            traj_mask_u = traj_mask_u.to(device)  # B x T

            with torch.autocast(device_type="cuda", dtype=torch.float16):

                (
                    state_preds_u,
                    action_preds_u,
                    return_preds_u,
                    imp_return_preds_u,
                    reward_preds_u,
                ) = model.forward(
                    timesteps=timesteps_u,
                    states=states_u,
                    actions=actions_u,
                    returns_to_go=returns_to_go_u,
                    rewards=rewards_u,
                )

                def cross_entropy(logits, labels):
                    # labels = F.one_hot(labels.long(), num_classes=int(num_bin)).squeeze(2)
                    labels = F.one_hot(
                        labels.long(), num_classes=int(num_bin)
                    ).squeeze()
                    criterion = nn.CrossEntropyLoss()
                    return criterion(logits, labels.float())

                # only consider non padded elements
                # action mse loss
                action_preds_u = action_preds_u.view(-1, act_dim)[
                    traj_mask_u.view(
                        -1,
                    )
                    > 0
                ]
                action_target_u = actions_u.view(-1, act_dim)[
                    traj_mask_u.view(
                        -1,
                    )
                    > 0
                ]
                action_loss = F.mse_loss(
                    action_preds_u, action_target_u, reduction="mean"
                )

                # state mse loss (optional)
                state_preds_u = state_preds_u.view(-1, state_dim)[
                    traj_mask_u.view(
                        -1,
                    )
                    > 0
                ]
                state_target_u = next_states_u.view(-1, state_dim)[
                    traj_mask_u.view(
                        -1,
                    )
                    > 0
                ]
                state_loss = F.mse_loss(
                    state_preds_u, state_target_u, reduction="mean"
                )

                # return expectile loss
                def expectile_loss(diff, expectile=0.8):
                    weight = torch.where(diff > 0, expectile, (1 - expectile))
                    return weight * (diff**2)

                imp_return_pred = imp_return_preds_u.reshape(-1, 1)[
                    traj_mask_u.view(
                        -1,
                    )
                    > 0
                ]
                imp_return_target = returns_to_go_u.reshape(-1, 1)[
                    traj_mask_u.view(
                        -1,
                    )
                    > 0
                ]

                imp_loss = expectile_loss(
                    (imp_return_target - imp_return_pred), expectile
                ).mean()


                # return cross entropy loss
                return_preds_u = return_preds_u.reshape(-1, int(num_bin))[
                    traj_mask_u.view(
                        -1,
                    )
                    > 0
                ]
                return_target_u = (
                    encode_return(
                        env_name,
                        returns_to_go_u,
                        num_bin=num_bin,
                        rtg_scale=rtg_scale,
                    )
                    .float()
                    .reshape(-1, 1)[
                        traj_mask_u.view(
                            -1,
                        )
                        > 0
                    ]
                )
                ret_ce_loss = cross_entropy(return_preds_u, return_target_u)

                edt_loss = (
                    action_loss
                    + state_loss * state_loss_weight
                    + imp_loss * exp_loss_weight
                    + args.ce_weight * ret_ce_loss
                )

            optimizer.zero_grad()
            scaler.scale(edt_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            ret_ce_losses.append(ret_ce_loss.detach().cpu().item())
            log_exp_losses.append(imp_loss.detach().cpu().item())
            log_action_losses.append(action_loss.detach().cpu().item())
            log_state_losses.append(state_loss.detach().cpu().item())

        mean_ret_loss = np.mean(ret_ce_losses)
        mean_action_loss = np.mean(log_action_losses)
        mean_state_loss = np.mean(log_state_losses)
        mean_expectile_loss = np.mean(log_exp_losses)
        time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)
        total_updates += num_updates_per_iter

        log_str = (
            "=" * 60
            + "\n"
            + "time elapsed: "
            + time_elapsed
            + "\n"
            + "num of updates: "
            + str(total_updates)
            + "\n"
            + "ret loss: "
            + format(mean_ret_loss, ".5f")
            + "\n"
            + "action loss: "
            + format(mean_action_loss, ".5f")
            + "\n"
            + "state loss: "
            + format(mean_state_loss, ".5f")
            + "\n"
            + "exp loss: "
            + format(mean_expectile_loss, ".5f")
            + "\n"
        )

        if i_train_iter % args.model_save_iters == 0:
            model_saver.save_model(model, epoch=i_train_iter)
        
        if i_train_iter % 10 == 0:
            print(log_str)

    print("=" * 60)
    print("finished training!")
    print("=" * 60)
    end_time = datetime.now().replace(microsecond=0)
    time_elapsed = str(end_time - start_time)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("started training at: " + start_time_str)
    print("finished training at: " + end_time_str)
    print("total training time: " + time_elapsed)
    print("=" * 60)


if __name__ == "__main__":

    args = parse()

    wandb.init(project=args.project_name, config=OmegaConf.to_container(args, resolve=True))

    train(args)
