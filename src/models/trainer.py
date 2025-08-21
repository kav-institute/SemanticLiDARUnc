# models/trainer.py
from __future__ import annotations

import os
import math
import time
import json
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# --- project modules (refactored) ---
from models.evaluator import SemanticSegmentationEvaluator

from models.temp_scaling import (
    cache_calib_logits,
    calibrate_temperature_from_cache,
)

from utils.mc_dropout import (
    set_dropout_mode,
    mc_dropout_probs,
    predictive_entropy_mc,
)

from utils.inputs import set_model_inputs

from utils.vis_cv2 import (
    visualize_semantic_segmentation_cv2,
    open_window,
    close_window,
    show_stack
)

# optional / uncertainty helpers (unchanged file)
from models.probability_helper import (
    to_alpha_concentrations,
    get_predictive_entropy,
    get_predictive_entropy_norm,
    get_aleatoric_uncertainty,
    compute_mc_reliability_bins,
    save_reliability_diagram,
    compute_entropy_error_iou,
    plot_mIOU_errorEntropy,
    compute_entropy_reliability,
    get_dirichlet_uncertainty_imgs
)

from utils.o3d_env import (
    ensure_o3d_runtime, 
    has_display
)

def create_ia_plots(args_cv2, args_o3d, save_dir=""):
    assert len(args_o3d)==2, "create_cv2_plots function args_o3d requires 2 arguments, xyz and pcl colors"
    import cv2
    img = show_stack(args_cv2, scale=1.5)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'),):
    # if (cv2.waitKey(1) & 0xFF) == ord('q'):
        import open3d as o3d
        if not has_display():
            # Headless session: save a PLY instead of opening a window
            if save_dir:
                xyz, color_bgr = args_o3d
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz.reshape(-1, 3))
                # convert BGR [0..255] to RGB [0..1]
                rgb = (color_bgr[..., ::-1].reshape(-1, 3).astype(np.float32)) / 255.0
                pcd.colors = o3d.utility.Vector3dVector(rgb)
                out = os.path.join(save_dir, "snapshot.ply")
                o3d.io.write_point_cloud(out, pcd)
                print(f"[viz] No display; saved point cloud to {out}")
            else:
                print("[viz] No display available (DISPLAY/WAYLAND_DISPLAY not set).")
            return

        # GUI path: make sure XDG_RUNTIME_DIR exists
        ensure_o3d_runtime()

        # Build and show point cloud
        xyz, color_bgr = args_o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.reshape(-1, 3))
        rgb = (color_bgr[..., ::-1].reshape(-1, 3).astype(np.float32)) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([mesh, pcd])
        

        #time.sleep(10)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(args_o3d[0].reshape(-1,3))
        # pcd.colors = o3d.utility.Vector3dVector(np.float32(args_o3d[1][...,::-1].reshape(-1,3))/255.0)

        # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # o3d.visualization.draw_geometries([mesh, pcd])

class Trainer:
    """
    Minimal, clean Trainer with:
      - standard train loop
      - test loop (optionally MC-dropout and Dirichlet-based diagnostics)
      - optional post-hoc Temperature Scaling (TS) on cached logits

    Expectation:
      - Your model exposes `forward_logits(*inputs)` for raw logits.
      - Your model's default `forward(*inputs)` returns probabilities (softmaxed),
        used as fallback when `forward_logits` does not exist.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: dict,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        visualize: bool = False,
        logging: bool = False,
        test_mask=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.visualize = visualize
        self.logging = logging

        if test_mask is None:
            test_mask = [1] * cfg["extras"]["num_classes"]

        # data / task meta
        self.num_classes = int(cfg["extras"]["num_classes"])
        self.class_names = cfg["extras"]["class_names"]
        self.class_colors = cfg["extras"]["class_colors"]

        # loss selection
        self.loss_name = cfg["extras"]["loss_function"]

        # evaluator
        self.evaluator = SemanticSegmentationEvaluator(self.num_classes, test_mask=test_mask)

        # device & writer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.save_path = cfg["extras"].get("save_path", "")
        self.writer = SummaryWriter(self.save_path) if self.logging and self.save_path else None

        # timers
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)

        # temperature value (optional, filled once calibrated)
        self.T_value: float | None = None

        # build criterion(s)
        self._init_losses()

    # ------------------------------
    # loss setup
    # ------------------------------
    def _init_losses(self):
        from models.losses import (
            TverskyLoss,
            SemanticSegmentationLoss,
            LovaszSoftmax,
            DirichletNLLLoss,
            DirichletBetaMomentLoss,
        )

        if self.loss_name == "Tversky":
            self.criterion_sem = SemanticSegmentationLoss()
            self.criterion_tversky = TverskyLoss()
        elif self.loss_name == "CE":
            self.criterion_sem = SemanticSegmentationLoss()
        elif self.loss_name == "Lovasz":
            self.criterion_lovasz = LovaszSoftmax()
        elif self.loss_name == "Dirichlet":
            self.criterion_dir_nll = DirichletNLLLoss()
            self.criterion_dir_bm = DirichletBetaMomentLoss(p=2)  # optional second term
        else:
            raise NotImplementedError(f"Unknown loss function: {self.loss_name}")

    # ------------------------------
    # training
    # ------------------------------
    def train_one_epoch(self, loader, epoch: int):
        self.model.train()
        total_loss = 0.0

        # >>> additional initialization
        # Dirichlet specific
        if self.loss_name == "Dirichlet":
            get_predictive_entropy_norm.reset()
        
        for step, (range_img, reflectivity, xyz, normals, labels) in enumerate(
            tqdm.tqdm(loader, desc=f"train {epoch+1}")
        ):            
            range_img = range_img.to(self.device)
            reflectivity = reflectivity.to(self.device)
            xyz = xyz.to(self.device)
            normals = normals.to(self.device)
            labels = labels.to(self.device) # [B,1,H,W] or [B,H,W]
            if labels.ndim == 4 and labels.shape[1] == 1:
                labels = labels.squeeze(1).long()
            else:
                labels = labels.long()

            inputs = set_model_inputs(range_img, reflectivity, xyz, normals, self.cfg)
            self._start.record()
            outputs = self.model(*inputs)  # default: probabilities (for most baselines)
            # >>> Output-kind classifier (logits / probs / log_probs) <<<
            if not hasattr(self, "_model_act_kind"):
                from models.losses import classify_output_kind
                self._model_act_kind = classify_output_kind(outputs, class_dim=1)
            self._end.record()
            torch.cuda.synchronize()

            # ---- compute loss (by selected head) ----
            if self.loss_name == "Tversky":
                loss_sem = self.criterion_sem(outputs, labels, num_classes=self.num_classes, model_act=self._model_act_kind)
                loss_t = self.criterion_tversky(outputs, labels, num_classes=self.num_classes, model_act=self._model_act_kind)
                loss = loss_sem + loss_t
            elif self.loss_name == "CE":
                loss = self.criterion_sem(outputs, labels, num_classes=self.num_classes, model_act=self._model_act_kind)
            elif self.loss_name == "Lovasz":
                loss = self.criterion_lovasz(outputs, labels)
            elif self.loss_name == "Dirichlet":
                # If your model returns raw logits for Dirichlet, point outputs there.
                loss_nll = self.criterion_dir_nll(outputs, labels)
                # loss_bm = self.criterion_dir_bm(outputs, labels)  # optional term
                loss = loss_nll
                
                # >>> additional initializations
                alpha = None
            else:
                raise RuntimeError("unreachable")

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
                
            # detach model output tensor from auto-gradient graph, and put to cpu
            outputs = outputs.detach().cpu()
            labels  = labels .detach().cpu()
            
            # get the most likely class
            outputs_argmax = torch.argmax(outputs,dim=1)
            
            if self.loss_name=="Dirichlet":
                if alpha is None: alpha = to_alpha_concentrations(outputs)
                unc_tot_batch_sum = get_predictive_entropy_norm.accumulate(alpha)
                # alternative usage: 
                    # unc_tot = get_predictive_entropy_norm(alpha)
                    # get_predictive_entropy_norm.add(unc_tot)
                    
            # logging
            if self.logging and self.writer and step % 10 == 0:
                global_step = epoch * len(loader) + step
                self.writer.add_scalar("train/loss/iter", loss.item(), global_step)
                self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], global_step)
                
            if self.visualize:
                # for visualization use first element in batch,  [B,H,W] -> [B,W,H]
                semantics_pred = (outputs_argmax).permute(0, 1, 2)[0,...].numpy()
                semantics_gt = (labels).squeeze(1).permute(0, 1, 2)[0,...].numpy()
                error_img = np.uint8(np.where(semantics_pred[...,None]!=semantics_gt[...,None], (0,0,255), (0,0,0)))
                xyz_img = (xyz).permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                reflectivity_img = (reflectivity).permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                reflectivity_img = np.uint8(255*np.concatenate(3*[reflectivity_img],axis=-1))
                prev_sem_pred = visualize_semantic_segmentation_cv2(semantics_pred, class_colors=self.class_colors)
                prev_sem_gt = visualize_semantic_segmentation_cv2(semantics_gt, class_colors=self.class_colors)
                normal_img = np.uint8(255*(normals.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()+1)/2 )
                if self.loss_name == "Dirichlet":
                    # get alpha concentration parameters form model output logits
                    if alpha is None: alpha = to_alpha_concentrations(outputs)

                    unc_tot_img, unc_a_img, unc_e_img = get_dirichlet_uncertainty_imgs(alpha)
                    create_ia_plots(args_cv2=(reflectivity_img, normal_img, prev_sem_pred, prev_sem_gt, error_img, unc_tot_img, unc_a_img, unc_e_img), 
                                    args_o3d=(xyz_img, prev_sem_pred))
                else:
                    create_ia_plots(args_cv2=(reflectivity_img, normal_img, prev_sem_pred, prev_sem_gt, error_img), 
                                    args_o3d=(xyz_img, prev_sem_pred))

        avg = total_loss / max(1, len(loader))
        print(f"[train] epoch {epoch+1}  loss={avg:.4f}")
        if self.logging and self.writer:
            self.writer.add_scalar("train/loss/epoch", avg, epoch)
            if self.loss_name=="Dirichlet":
                self.writer.add_scalar('train/unc_tot/epoch', get_predictive_entropy_norm.mean(reset=True), global_step)

    # ------------------------------
    # evaluation
    # ------------------------------
    @torch.no_grad()
    def test_one_epoch(self, loader, epoch: int):
        from models.losses import classify_output_kind  # use your canonical detector
        import torch.nn.functional as F
        EPS = 1e-12
        def _to_logits(out: torch.Tensor, kind: str) -> torch.Tensor:
            if kind == 'logits':
                return out
            elif kind == 'probs':
                return out.clamp_min(EPS).log()
            elif kind == 'log_probs':
                return out
            else:
                raise ValueError(f"Unknown output kind: {kind}")

        self.model.eval()
        self.evaluator.reset()

        # toggles
        use_dropout = bool(self.cfg["model_settings"].get("use_dropout", 0))
        want_dirichlet_metrics = (self.loss_name == "Dirichlet")

        ts_cache_mode = "mc" if use_dropout else "default"
                # MC sample count (re-use calibration setting if present)
        mc_T = int(self.cfg.get("calibration", {}).get("mc_samples", 30))
        
        # optional uncertainty plots config
        metrics_cfg = self.cfg.get("logging_settings", {}).get("metrics", {})
        do_mc_conf_acc = bool(metrics_cfg.get("McConfEmpAcc", 0))
        do_entropy_iou = bool(metrics_cfg.get("IouPlt", 0))
        do_entropy_rel = bool(metrics_cfg.get("EntErrRel", 0))

        if (do_mc_conf_acc or do_entropy_iou or do_entropy_rel) and not want_dirichlet_metrics and not use_dropout:
            # nothing to compute; silently skip
            do_mc_conf_acc = do_entropy_iou = do_entropy_rel = False

        # prepare aggregators
        n_bins = 10
        thresholds = np.linspace(0.0, 1.0, n_bins, endpoint=False)

        if do_mc_conf_acc:
            mc_hits = np.zeros(n_bins, dtype=np.float64)
            mc_tot = np.zeros(n_bins, dtype=np.float64)

        if do_entropy_iou:
            all_ious = []

        if do_entropy_rel:
            ent_err_tot = np.zeros(n_bins, dtype=np.float64)
            ent_err_err = np.zeros(n_bins, dtype=np.float64)

        for step, (range_img, reflectivity, xyz, normals, labels) in enumerate(
            tqdm.tqdm(loader, desc=f"eval {epoch+1}")
        ):
            # --- inputs/labels prep (unchanged) ---
            range_img = range_img.to(self.device)
            reflectivity = reflectivity.to(self.device)
            xyz = xyz.to(self.device)
            normals = normals.to(self.device)
            labels = labels.to(self.device)
            if labels.ndim == 4 and labels.shape[1] == 1:
                labels = labels.squeeze(1).long()
            else:
                labels = labels.long()
            inputs = set_model_inputs(range_img, reflectivity, xyz, normals, self.cfg)

            # ---- If MC-dropout is enabled, do ONLY the MC path (single evaluator update) ----
            if use_dropout:
                # (Optional) cache model output kind once for telemetry/consistency
                if not hasattr(self, "_model_act_kind") or self._model_act_kind is None:
                    peek = self.model(*inputs)
                    self._model_act_kind = classify_output_kind(peek, class_dim=1)
                    del peek

                # >>> MC sanity checks after mc_dropout_probs(...) is called
                    # - global std: mc_probs.std(dim=0).mean().item() ---> >~1e-6 
                    # - fraction of pixels that changed class: (mc_probs.argmax(dim=2) != mc_probs.argmax(dim=2)[0:1]).any(dim=0).float().mean().item() ---> >0
                
                # Decide where to apply T based on how T was calibrated
                if self.T_value is not None and ts_cache_mode == "default":
                    # Per-sample scaling: softmax(logits_like/T) then average
                    T_val = max(1e-3, float(self.T_value))
                    mc_probs = mc_dropout_probs(self.model, inputs, T=mc_T, temperature=T_val)  # [T,B,C,H,W]
                    probs = mc_probs.mean(dim=0)  # [B,C,H,W]
                else:
                    # No per-sample TS; average first. If T exists & was MC-calibrated, apply post-mean.
                    mc_probs = mc_dropout_probs(self.model, inputs, T=mc_T, temperature=None)  # [T,B,C,H,W]
                    mean_p = mc_probs.mean(dim=0).clamp_min(EPS)  # [B,C,H,W]
                    if self.T_value is not None and ts_cache_mode == "mc":
                        T_val = max(1e-3, float(self.T_value))
                        probs = torch.softmax(torch.log(mean_p) / T_val, dim=1)
                    else:
                        probs = mean_p

                preds = probs.argmax(dim=1)
                self.evaluator.update(preds, labels)

                # --- optional uncertainty metrics (unchanged logic) ---
                if (do_entropy_iou or do_entropy_rel) and self.logging:
                    ent_mc = predictive_entropy_mc(mc_probs, normalize=True)  # [B,H,W]
                    err_mask = (preds != labels).to(torch.int32)
                    if do_entropy_iou:
                        ious = compute_entropy_error_iou(ent_mc, err_mask, thresholds=thresholds)
                        all_ious.append(ious.cpu().numpy())
                    if do_entropy_rel:
                        tot, err, err_rate, ece = compute_entropy_reliability(ent_mc, err_mask, n_bins=n_bins)
                        ent_err_tot += tot
                        ent_err_err += err

                continue  # done with MC batch

            # ---- Non-MC path: your code (kept) ----
            out = self.model(*inputs)  # logits/probs/log_probs
            if not hasattr(self, "_model_act_kind") or self._model_act_kind is None:
                self._model_act_kind = classify_output_kind(out, class_dim=1)

            logits_raw = _to_logits(out, self._model_act_kind)  # un-tempered logits-like
            logits = logits_raw
            if self.T_value is not None:
                logits = logits / max(1e-3, float(self.T_value))

            probs = F.softmax(logits, dim=1)
            assert probs.shape[1] == self.num_classes, \
                f"Channel mismatch: probs C={probs.shape[1]} vs cfg num_classes={self.num_classes}"

            preds = probs.argmax(dim=1)
            self.evaluator.update(preds, labels)

            # ---- optional uncertainty metrics (Dirichlet branch uses UN-TEMPERED logits) ----
            if want_dirichlet_metrics:
                alpha = to_alpha_concentrations(logits_raw)  # use raw logits for alpha
                pred_entropy = get_predictive_entropy(alpha)  # [B,H,W]
                pred_entropy_norm = pred_entropy / math.log(self.num_classes)

                if do_mc_conf_acc:
                    hits, tot = compute_mc_reliability_bins(alpha, labels, n_bins=n_bins, n_samples=120)
                    mc_hits += hits
                    mc_tot += tot

                if do_entropy_iou:
                    err_mask = (preds != labels).to(torch.int32)
                    ious = compute_entropy_error_iou(pred_entropy_norm, err_mask, thresholds=thresholds)
                    all_ious.append(ious.cpu().numpy())

                if do_entropy_rel:
                    err_mask = (preds != labels).to(torch.int32)
                    tot, err, err_rate, ece = compute_entropy_reliability(pred_entropy_norm, err_mask, n_bins=n_bins)
                    ent_err_tot += tot
                    ent_err_err += err

        mIoU, per_class = self.evaluator.compute_final_metrics(class_names=self.class_names)
        print(f"[eval] epoch {epoch+1}  mIoU={mIoU:.4f}")

        # --- plots / logs ---
        if self.logging and self.save_path:
            out_dir = os.path.join(self.save_path, "eval")
            os.makedirs(out_dir, exist_ok=True)

            if do_mc_conf_acc and mc_tot.sum() > 0:
                emp_acc = np.divide(mc_hits, mc_tot, out=np.zeros_like(mc_hits), where=mc_tot > 0)
                edges = np.linspace(0.0, 1.0, n_bins + 1)
                centers = 0.5 * (edges[:-1] + edges[1:])
                save_reliability_diagram(
                    empirical_acc=emp_acc,
                    bin_centers=centers,
                    tot_counts=mc_tot,
                    output_path=os.path.join(out_dir, f"reliability_epoch_{epoch:06d}.png"),
                    title='Reliability diagram\n(dot area ∝ #pixels per bin — sharpness)',
                    xlabel='Predicted confidence (MC estimate)',
                    ylabel='Empirical accuracy',
                )

            if do_entropy_iou and len(all_ious) > 0:
                all_ious_np = np.stack(all_ious, axis=0)
                mean_ious = all_ious_np.mean(axis=0)
                plot_mIOU_errorEntropy(
                    mean_ious,
                    thresholds,
                    output_path=os.path.join(out_dir, f"entropy_iou_epoch_{epoch:06d}.png"),
                )

            if do_entropy_rel and ent_err_tot.sum() > 0:
                edges = np.linspace(0.0, 1.0, n_bins + 1)
                centers = 0.5 * (edges[:-1] + edges[1:])
                emp_err_rate = np.divide(ent_err_err, ent_err_tot, out=np.zeros_like(ent_err_err), where=ent_err_tot > 0)
                save_reliability_diagram(
                    empirical_acc=emp_err_rate,
                    bin_centers=centers,
                    tot_counts=ent_err_tot,
                    output_path=os.path.join(out_dir, f"entropy_error_rel_epoch_{epoch:06d}.png"),
                    title='Reliability of Entropy as Error Predictor\n(dot area ∝ #pixels per entropy bin — sharpness)',
                    xlabel='Normalized entropy',
                    ylabel='Observed error rate',
                )

        return mIoU


    # ------------------------------
    # main loop
    # ------------------------------
    def __call__(self, train_loader, val_loader):
        self.num_epochs = int(self.cfg["train_params"]["num_epochs"])
        test_every = int(self.cfg["logging_settings"]["test_every_nth_epoch"])

        # TS config (post-hoc, optional)
        cal_cfg = self.cfg.get("calibration", {})
        ts_enable = bool(cal_cfg.get("enable", False))
        ts_run_each_eval = bool(cal_cfg.get("run_each_eval", False))
        #ts_cache_mode = cal_cfg.get("cache_mode", "default")  # "default" or "mc"
        ts_cache_mode = "mc" if self.cfg['model_settings']['use_dropout'] else "default"
        ts_optimizer = cal_cfg.get("optimizer", "lbfgs")      # "lbfgs" or "adam"
        ts_lr = float(cal_cfg.get("lr", 0.05))
        ts_epochs = int(cal_cfg.get("epochs", 2))
        ts_chunk = int(cal_cfg.get("chunk_size", 1_000_000))
        ts_max_iter = int(cal_cfg.get("max_iter", 100))
        ts_save_json = cal_cfg.get("save_path", None)

        best_mIoU = -1.0

        for epoch in range(self.num_epochs):
            # ----- train -----
            if self.visualize: open_window()
            self.train_one_epoch(train_loader, epoch)
            if self.visualize: close_window()
            
            # step schedulers (if not ReduceLROnPlateau)
            if self.scheduler and not isinstance(
                self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.scheduler.step()

            # ----- periodic eval -----
            if epoch % max(1, test_every) == 0:
                mIoU = self.test_one_epoch(val_loader, epoch)

                # optional: update ReduceLROnPlateau with metric
                if self.scheduler and isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(mIoU)

                # save best checkpoint (optional)
                if self.logging and self.save_path and mIoU > best_mIoU:
                    best_mIoU = mIoU
                    ckpt_path = os.path.join(self.save_path, f"best_epoch_{epoch:06d}.pt")
                    torch.save(self.model.state_dict(), ckpt_path)

                # ----- optional post-hoc Temperature Scaling -----
                if ts_enable and (self.T_value is None or ts_run_each_eval):
                    logits_cpu, labels_cpu = cache_calib_logits(
                        model=self.model,
                        val_loader=val_loader,
                        device=self.device,
                        cfg=self.cfg,
                        ignore_index=255,
                        mode=ts_cache_mode,
                        mc_samples=int(cal_cfg.get("mc_samples", 30)),
                    )
                    self.T_value = calibrate_temperature_from_cache(
                        logits_cpu=logits_cpu,
                        labels_cpu=labels_cpu,
                        device=self.device,
                        init_T="auto",
                        optimizer_type=ts_optimizer,
                        lr=ts_lr,
                        epochs=ts_epochs,
                        chunk_size=ts_chunk,
                        max_iter_lbfgs=ts_max_iter,
                        prev_T=self.T_value,
                        save_path=ts_save_json,
                    )
                    print(f"[TS] calibrated T = {self.T_value:.4f}")

                    # (optional) re-test with calibrated T for plots
                    self.test_one_epoch(val_loader, epoch)

        # final test
        self.test_one_epoch(val_loader, self.num_epochs - 1)

        # final save
        if self.logging and self.save_path:
            torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_final.pt"))
