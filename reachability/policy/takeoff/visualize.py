import torch
import numpy as np

import sys
sys.path.append('source/latent_safety')
import dreamerv3_torch.utils as utils

import wandb

def visualize_uncertainty_with_value(wm, ensemble, value, data, step):
    data = wm.preprocess(data)
    embed = wm.encoder(data)

    obs_step = 5

    # Observed states and reconstruction
    states, _ = wm.dynamics.observe(
        embed[:6, :obs_step], data["action"][:6, :obs_step], data["is_first"][:6, :obs_step]
    )
    recon = wm.heads["decoder"](wm.dynamics.get_feat(states))["front_cam"].mode()[:6]

    # Initial state and imagined rollout
    init = {k: v[:, -1] for k, v in states.items()}
    prior = wm.dynamics.imagine_with_action(data["action"][:6, obs_step:], init)
    openl = wm.heads["decoder"](wm.dynamics.get_feat(prior))["front_cam"].mode()

    # Ground truth
    truth = data["front_cam"][:6]

    # Handle episode endings
    row, col = torch.where(data["is_first"][:6, obs_step:] == 1.0)
    for i in range(row.size(0)):
        data["is_first"][row[i], obs_step + col[i]:] = 1.0
        openl[row[i], col[i]:] = openl[row[i], col[i] - 1]
        truth[row[i], obs_step + col[i]:] = truth[row[i], obs_step + col[i] - 1]

    # Observed and predicted video
    model = torch.cat([recon[:, :obs_step], openl], 1)
    error = (model - truth + 1.0) / 2.0
    video_pred = torch.cat([truth, model, error], 2)


    with torch.no_grad():
        feat_post = wm.dynamics.get_feat(states)
        feat_prior = wm.dynamics.get_feat(prior)
        feat = torch.cat([feat_post, feat_prior], 1)
        action = data["action"][:6]
        inputs = torch.concat([feat, action], -1)

        disagreement_ensemble = ensemble.intrinsic_reward_penn(inputs)

    video_pred = utils.concat_uncertainty_with_video(data, video_pred, disagreement_ensemble)

    with torch.no_grad():
        B, T, D = inputs.shape
        inputs = inputs.view(-1, D)
        value_function_estimate = value(inputs).clamp(-1, 1)
        value_function_estimate = value_function_estimate.view(B, T, -1)


    # Combine value predictions with video
    video_pred = utils.concat_uncertainty_with_video(data, video_pred, value_function_estimate, thr=-1, max_val=1)

    video_pred = np.clip(255 * video_pred, 0, 255).astype(np.uint8)
    B, T, H, W, C = video_pred.shape
    video_pred = video_pred.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
    wandb.log({"value_estimate_prior": wandb.Video(video_pred, fps=10, format="mp4")}, step=step)
