import os
import time
import numpy as np
import pathlib
import torch

from torch import nn
import dreamer.exploration as expl
import dreamer.models as models
import dreamer.tools as tools
from common.utils import to_np, combine_dictionaries
#from diffusers.training_utils import EMAModel
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import matplotlib.patches as patches
import io

# os.environ["MUJOCO_GL"] = "osmesa"


class Dreamer(nn.Module):
    def __init__(
        self, obs_space, act_space, config, logger, dataset
    ):
        super(Dreamer, self).__init__()
        self.dpi = config.size[0]
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        self._should_log_video = tools.Every(config.log_every_video)
        batch_steps = config.batch_size * config.batch_length
        self._batch_train_steps = int(
            config.steps_per_batch * config.train_ratio / batch_steps
        )
        print(
            f"Updating the agent for {self._batch_train_steps} every {config.steps_per_batch} env steps"
        )
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))

        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat if logger is not None else 0
        self._update_count = 0
        self._dataset = dataset

        self._wm = models.WorldModel(obs_space, act_space, self._step, config)

        if (
            config.pretrain_actor_steps > 0
            or config.pretrain_joint_steps > 0
            or config.from_ckpt is not None
        ):
            self._make_pretrain_opt()

        if config.use_ensemble:        
            self._disag_ensemble = expl.OneStepPredictor(config, self._wm)

    def pretrain_model_only(self, data, step=None):
        metrics = {}
        wm = self._wm
        data = wm.preprocess(data)
        if self._config.pretrain_annealing is None:
            recon_weight = 1.0
        elif self._config.pretrain_annealing == "linear":
            recon_weight = (
                self._config.pretrain_joint_steps - (step - 1)
            ) / self._config.pretrain_joint_steps
            recon_weight = max(0.0, recon_weight)
        else:
            print(self._config.pretrain_annealing)
            raise Exception("Annealing strategy must be None or Linear")

        with tools.RequiresGrad(wm):
            with torch.cuda.amp.autocast(wm._use_amp):
                embed = wm.encoder(data)
                # post: z_t, prior: \hat{z}_t
                post, prior = wm.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                # note: kl_loss is already sum of dyn_loss and rep_loss
                kl_loss, kl_value, dyn_loss, rep_loss = wm.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape

                losses = {}
                feat = wm.dynamics.get_feat(post)

                if (
                    self._config.recon_pretrain
                    and step <= self._config.pretrain_joint_steps
                ):
                    # preds is dictionary of all all MLP+CNN keys
                    preds = wm.heads["decoder"](feat)
                    for name, pred in preds.items():
                        loss = -pred.log_prob(data[name])
                        assert loss.shape == embed.shape[:2], (name, loss.shape)
                        losses[name] = loss
                    recon_loss = sum(losses.values())
                else:
                    recon_loss = 0

                if self._config.use_ensemble:
                    with tools.RequiresGrad(self._disag_ensemble._networks):
                        stoch = post["stoch"]
                        target = {
                            "embed": embed,
                            "stoch": stoch,
                            "deter": post["deter"],
                            "feat": feat,
                        }[self._config.disag_target]

                        inputs = feat
                        action = torch.Tensor(data["action"]).to(self._config.device)
                        if self._config.disag_action_cond:
                            inputs = torch.concat(
                                [inputs, action], -1
                            )
                        # Train the ensemble model
                        ensemble_mets = self._disag_ensemble._train_ensemble_penn(inputs, target, data["is_first"])

                    with torch.no_grad():
                        log_disagreement = self._disag_ensemble._intrinsic_reward_penn(feat, action)
                        log_disagreement_mean = log_disagreement.mean()

                model_loss = kl_loss + recon_weight * recon_loss
                metrics = self.pretrain_opt(
                    torch.mean(model_loss), self.pretrain_params
                )

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_loss"] = to_np(kl_loss)
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl_value"] = to_np(torch.mean(kl_value))
        metrics["recon_weight"] = recon_weight

        if self._config.use_ensemble:
            metrics["ensemble"] = ensemble_mets['explorer_loss']
            metrics["log_disagreement"] = to_np(log_disagreement_mean)

        with torch.cuda.amp.autocast(wm._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(wm.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(wm.dynamics.get_dist(post).entropy())
            )
        metrics = {
            f"model_only_pretrain/{k}": v for k, v in metrics.items()
        }  # Add prefix model_pretrain to all metrics
        self._update_running_metrics(metrics)
        self._maybe_log_metrics(fps_namespace="model_only_pretrain/")
        self._step += 1
        self._logger.step = self._step

    def pretrain_disagreement_ensemble_only(self, data, step=None):
        metrics = {}
        wm = self._wm
        data = wm.preprocess(data)
        
        with torch.cuda.amp.autocast(wm._use_amp):
            embed = wm.encoder(data)
            # post: z_t, prior: \hat{z}_t
            post, prior = wm.dynamics.observe(
                embed, data["action"], data["is_first"]
            )
            feat = wm.dynamics.get_feat(post)

            if self._config.use_ensemble:
                with tools.RequiresGrad(self._disag_ensemble._networks):
                    stoch = post["stoch"]
                    target = {
                        "embed": embed,
                        "stoch": stoch,
                        "deter": post["deter"],
                        "feat": feat,
                    }[self._config.disag_target]

                    inputs = feat
                    action = torch.Tensor(data["action"]).to(self._config.device)
                    if self._config.disag_action_cond:
                        inputs = torch.concat(
                            [inputs, action], -1
                        )
                    # Train the ensemble model
                    # penn
                    ensemble_mets = self._disag_ensemble._train_ensemble_penn(inputs, target, data["is_first"])
                    metrics["ensemble"] = ensemble_mets['explorer_loss']
                    
                with torch.no_grad():
                    log_disagreement = self._disag_ensemble._intrinsic_reward_penn(feat, action)    
                    log_disagreement_mean = log_disagreement.mean()
                    metrics["log_disagreement"] = to_np(log_disagreement_mean)

        metrics = {
            f"disagreement_ensemble_pretrain/{k}": v for k, v in metrics.items()
        }  # Add prefix disagreement_ensemble_pretrain to all metrics
        self._update_running_metrics(metrics)
        self._maybe_log_metrics(fps_namespace="disagreement_ensemble_pretrain/")
        self._step += 1
        self._logger.step = self._step

        return metrics

    def get_latent(self, xs, ys, thetas, imgs, lx_mlp):
        states = np.expand_dims(np.expand_dims(thetas,1),1)
        imgs = np.expand_dims(imgs, 1)
        dummy_acs = np.zeros((np.shape(xs)[0], 1, 3))
        rand_idx = 1 #np.random.randint(0, 3, np.shape(xs)[0])
        dummy_acs[np.arange(np.shape(xs)[0]), :, rand_idx] = 1
        firsts = np.ones((np.shape(xs)[0], 1))
        lasts = np.zeros((np.shape(xs)[0], 1))
        
        cos = np.cos(states)
        sin = np.sin(states)
        states = np.concatenate([cos, sin], axis=-1)
        data = {'obs_state': states, 'image': imgs, 'action': dummy_acs, 'is_first': firsts, 'is_terminal': lasts}

        data = self._wm.preprocess(data)
        embed = self._wm.encoder(data)

        post, prior = self._wm.dynamics.observe(
            embed, data["action"], data["is_first"]
            )
        feat = self._wm.dynamics.get_feat(post).detach()
        with torch.no_grad():  # Disable gradient calculation
            g_x = lx_mlp(feat).detach().cpu().numpy().squeeze()
        feat = self._wm.dynamics.get_feat(post).detach().cpu().numpy().squeeze()
        return g_x, feat, post
    
    def get_eval_plot(self, obs_mlp, theta):
        nx, ny, nz = 41, 41, 5

        v = np.zeros((nx, ny, nz))
        xs = np.linspace(-1, 1, nx)
        ys = np.linspace(-1, 1, ny)
        thetas= np.linspace(0, 2*np.pi, nz, endpoint=True)
        tn, tp, fn, fp = 0, 0, 0, 0
        it = np.nditer(v, flags=['multi_index'])
        ###
        idxs = []  
        imgs = []
        labels = []
        it = np.nditer(v, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            x = xs[idx[0]]
            y = ys[idx[1]]
            theta = thetas[idx[2]]
            if (x**2 + y**2) < (self._config.targetRadius**2):
                labels.append(1) # unsafe
            else:
                labels.append(0) # safe
            x = x - np.cos(theta)*1*0.05
            y = y - np.sin(theta)*1*0.05
            imgs.append(self.capture_image(np.array([x, y, theta])))
            idxs.append(idx)        
            it.iternext()
        idxs = np.array(idxs)
        safe_idxs = np.where(np.array(labels) == 0)
        unsafe_idxs = np.where(np.array(labels) == 1)
        x_lin = xs[idxs[:,0]]
        y_lin = ys[idxs[:,1]]
        theta_lin = thetas[idxs[:,2]]
        
        g_x = []
        ## all of this is because I can't do a forward pass with 128x128 images in one go
        num_c = 5
        chunk = int(np.shape(x_lin)[0]/num_c)
        for k in range(num_c):
            g_xlist, _, _ = self.get_latent(x_lin[k*chunk:(k+1)*chunk], y_lin[k*chunk:(k+1)*chunk], theta_lin[k*chunk:(k+1)*chunk], imgs[k*chunk:(k+1)*chunk], obs_mlp)
            g_x = g_x + g_xlist.tolist()
        g_x = np.array(g_x)
        v[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = g_x

        tp  = np.where(g_x[safe_idxs] > 0)
        fn  = np.where(g_x[safe_idxs] <= 0)
        fp  = np.where(g_x[unsafe_idxs] > 0)
        tn  = np.where(g_x[unsafe_idxs] <= 0)
        
        vmax = round(max(np.max(v), 0),1)
        vmin = round(min(np.min(v), -vmax),1)
        
        fig, axes = plt.subplots(nz, 2, figsize=(12, nz*6))
        
        for i in range(nz):
            ax = axes[i, 0]
            im = ax.imshow(
                v[:, :, i].T, interpolation='none', extent=np.array([
                -1.1, 1.1, -1.1,1.1, ]), origin="lower",
                cmap="seismic", vmin=vmin, vmax=vmax, zorder=-1
            )
            cbar = fig.colorbar(
                im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
            )
            cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
            ax.set_title(r'$g(x)$', fontsize=18)

            ax = axes[i, 1]
            im = ax.imshow(
                v[:, :, i].T > 0, interpolation='none', extent=np.array([
                -1.1, 1.1, -1.1,1.1, ]), origin="lower",
                cmap="seismic", vmin=-1, vmax=1, zorder=-1
            )
            cbar = fig.colorbar(
                im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
            )
            cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
            ax.set_title(r'$v(x)$', fontsize=18)
            fig.tight_layout()
            circle = plt.Circle((0, 0), self._config.targetRadius, fill=False, color='blue', label = 'GT boundary')

            # Add the circle to the plot
            axes[i,0].add_patch(circle)
            axes[i,0].set_aspect('equal')
            circle2 = plt.Circle((0, 0), self._config.targetRadius, fill=False, color='blue', label = 'GT boundary')

            axes[i,1].add_patch(circle2)
            axes[i,1].set_aspect('equal')

        fp_g = np.shape(fp)[1]
        fn_g = np.shape(fn)[1]
        tp_g = np.shape(tp)[1]
        tn_g = np.shape(tn)[1]
        tot = fp_g + fn_g + tp_g + tn_g
        fig.suptitle(r"$TP={:.0f}\%$ ".format(tp_g/tot * 100) + r"$TN={:.0f}\%$ ".format(tn_g/tot * 100) + r"$FP={:.0f}\%$ ".format(fp_g/tot * 100) +r"$FN={:.0f}\%$".format(fn_g/tot * 100),
            fontsize=10,)
        buf = BytesIO()

        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        plot = Image.open(buf).convert("RGB")
        return np.array(plot), tp, fn, fp, tn
 
    def train_lx(self, data, lx_mlp, lx_opt, eval=False):
        wm = self._wm
        wm.dynamics.sample = False
        data = wm.preprocess(data)
        R = self._config.targetRadius #0.5
        
        with tools.RequiresGrad(lx_mlp):
            if not eval:
                with torch.cuda.amp.autocast(wm._use_amp):
                    embed = self._wm.encoder(data)
                    post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])
                    feat = self._wm.dynamics.get_feat(post).detach() 
                    x, y, theta = data["privileged_state"][:,:,0], data["privileged_state"][:,:,1], data["privileged_state"][:,:, 2]

                    safety_data = (x**2 + y**2) - R**2
                    safe_data = torch.where(safety_data > 0)
                    unsafe_data = torch.where(safety_data <= 0)

                    safe_dataset = feat[safe_data]
                    unsafe_dataset = feat[unsafe_data]

                    pos = lx_mlp(safe_dataset)
                    neg = lx_mlp(unsafe_dataset)
                    
                    gamma = 0.75

                    if pos.size(0) > 0:
                        lx_loss = (1/pos.size(0))*torch.sum(torch.relu(gamma - pos)) #penalizes safe for being negative
                    else:
                        lx_loss = torch.FloatTensor([0]).cuda()

                    if neg.size(0) > 0:
                        lx_loss +=  (1/neg.size(0))*torch.sum(torch.relu(gamma + neg)) # penalizes unsafe for being positive
                    
                    lx_loss = lx_loss
            
                    lx_opt(torch.mean(lx_loss), lx_mlp.parameters())
                    plot_arr = None
                    score = 0
            else:
                lx_mlp.eval()
                plot_arr, tp, fn, fp, tn = self.get_eval_plot(lx_mlp, 0)
                lx_mlp.train()
                fp_num = np.shape(fp)[1]
                fn_num = np.shape(fn)[1]
                tp_num = np.shape(tp)[1]
                tn_num = np.shape(tn)[1]
                print('TP: ', tp_num)
                print('FN: ', fn_num)

                print('TN: ', tn_num)
                print('FP: ', fp_num)
            
                score = (fp_num + fn_num) / (fp_num + fn_num + tp_num + tn_num)

        return score, plot_arr
    
    def _update_running_metrics(self, metrics):
        for name, value in metrics.items():
            if name not in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

    def _maybe_log_metrics(self, video_pred_log=False, fps_namespace=""):
        if self._logger is not None:
            logged = False
            if self._should_log(self._step):
                for name, values in self._metrics.items():
                    if not np.isnan(np.mean(values)):
                        self._logger.scalar(name, float(np.mean(values)))
                        self._metrics[name] = []
                logged = True

            if video_pred_log and self._should_log_video(self._step):
                video_pred, video_pred2 = self._wm.video_pred(next(self._dataset))
                self._logger.video("train_openl_agent", to_np(video_pred))
                self._logger.video("train_openl_hand", to_np(video_pred2))
                logged = True

            if logged:
                self._logger.write(fps=True, fps_namespace=fps_namespace)

    def _make_pretrain_opt(self):
        config = self._config
        use_amp = True if config.precision == 16 else False
        if (
            config.pretrain_actor_steps + config.pretrain_joint_steps > 0
            or config.from_ckpt is not None
        ):
            # have separate lrs/eps/clips for actor and model
            # https://pytorch.org/docs/master/optim.html#per-parameter-options
            standard_kwargs = {
                "lr": config.model_lr,
                "eps": config.opt_eps,
                "clip": config.grad_clip,
                "wd": config.weight_decay,
                "opt": config.opt,
                "use_amp": use_amp,
            }
            model_params = {
                "params": list(self._wm.encoder.parameters())
                + list(self._wm.dynamics.parameters())
            }
            if config.recon_pretrain:
                model_params["params"] += list(self._wm.heads["decoder"].parameters())

            self.pretrain_params = list(model_params["params"])
            self.pretrain_opt = tools.Optimizer(
                "pretrain_opt", [model_params], **standard_kwargs
            )

            print(
                f"Optimizer pretrain has {sum(param.numel() for param in self.pretrain_params)} variables."
            )