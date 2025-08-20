# models/tester.py
import os
import math
import json
import tqdm
import torch
import numpy as np

from utils.inputs import set_model_inputs
from models.temp_scaling import cache_calib_logits, calibrate_temperature_from_cache
from utils.mc_dropout import mc_dropout_probs, predictive_entropy_mc
from utils.reliability import reliability_diagram_from_probs

from models.evaluator import SemanticSegmentationEvaluator
# optional extras you already have
from models.probability_helper import (
    to_alpha_concentrations, get_predictive_entropy,
    compute_mc_reliability_bins, save_reliability_diagram,
    compute_entropy_error_iou, plot_mIOU_errorEntropy,
    compute_entropy_reliability
)


class Tester:
    """
    Standalone tester:
    - loads checkpoint (if provided)
    - optional temperature scaling on a given calibration loader
    - runs a test pass with (optional) MC-dropout and ECE / RC plots saved
    """
    def __init__(self, model, cfg, visualize=False, logging=False, checkpoint: str | None = None):
        self.model = model
        self.cfg = cfg
        self.visualize = visualize
        self.logging = logging

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.num_classes = cfg["extras"]["num_classes"]
        self.class_names = cfg["extras"]["class_names"]
        self.class_colors = cfg["extras"]["class_colors"]
        self.loss_function = cfg["extras"]["loss_function"]
        self.save_path = cfg["extras"].get("save_path", "")

        self.evaluator = SemanticSegmentationEvaluator(self.num_classes)

        # Load checkpoint if provided
        if checkpoint:
            try:
                sd = torch.load(checkpoint, map_location=self.device)
                self.model.load_state_dict(sd)
                print(f"[Tester] Loaded checkpoint: {checkpoint}")
            except Exception as e:
                print(f"[Tester] WARNING: failed to load checkpoint: {checkpoint}\n{e}")

        # Temperature scalar (optional; set by calibrate_temperature)
        self.T_value: float | None = None

    # ---------- Temperature scaling ----------
    @torch.no_grad()
    def cache_logits(self, calib_loader, ignore_index=255, mode="default", mc_samples=30):
        """Wrapper to use the refactored cache helper."""
        return cache_calib_logits(
            model=self.model,
            val_loader=calib_loader,
            device=self.device,
            cfg=self.cfg,
            ignore_index=ignore_index,
            mode=mode,
            mc_samples=mc_samples
        )

    def calibrate_temperature(
        self,
        calib_loader,
        ignore_index: int = 255,
        mode: str = "default",          # {"default","mc"}
        mc_samples: int = 30,
        optimizer_type: str = "lbfgs",  # {"lbfgs","adam"}
        epochs: int = 2,                # Adam only
        chunk_size: int = 1_000_000,
        max_iter_lbfgs: int = 100,
        init_T: float | str = "auto",
        save_json: str | None = None
    ):
        """Fit temperature on cached logits/labels and store in self.T_value."""
        logits_cpu, labels_cpu = self.cache_logits(
            calib_loader, ignore_index=ignore_index, mode=mode, mc_samples=mc_samples
        )
        T = calibrate_temperature_from_cache(
            logits_cpu, labels_cpu,
            device=self.device,
            init_T=init_T,
            optimizer_type=optimizer_type,
            lr=0.05,
            epochs=epochs,
            chunk_size=chunk_size,
            max_iter_lbfgs=max_iter_lbfgs,
            prev_T=self.T_value,
            save_path=save_json
        )
        self.T_value = float(T)
        print(f"[Tester] Calibrated temperature T={self.T_value:.4f}")

    # ---------- Inference helpers ----------
    @torch.no_grad()
    def _forward_probs(self, inputs):
        """
        Returns probabilities [B,C,H,W].
        Applies temperature if self.T_value is set.
        """
        # If your model.forward returns probs:
        if hasattr(self.model, "forward_logits"):
            logits = self.model.forward_logits(*inputs)
            if self.T_value is not None:
                return torch.softmax(logits / max(1e-3, self.T_value), dim=1)
            return torch.softmax(logits, dim=1)
        else:
            # model forward already returns probs (e.g., SalsaNext)
            probs = self.model(*inputs)
            if self.T_value is not None:
                # convert back to logits ~ log(p) and re-temp-scale softly
                probs = probs.clamp_min(1e-12)
                logits = torch.log(probs)
                return torch.softmax(logits / max(1e-3, self.T_value), dim=1)
            return probs

    # ---------- Testing ----------
    @torch.no_grad()
    def test_epoch(self, dataloader, epoch: int = 0):
        self.model.eval()
        self.evaluator.reset()

        use_dropout = self.cfg["model_settings"].get("use_dropout", 0)
        out_dir = os.path.join(self.save_path, "eval") if self.save_path else None
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Optional metric accumulators
        want_iou_curve = self.cfg["logging_settings"]["metrics"].get("IouPlt", 0) if self.logging else False
        want_entropy_rel = self.cfg["logging_settings"]["metrics"].get("EntErrRel", 0) if self.logging else False
        want_mc_conf_rel = self.cfg["logging_settings"]["metrics"].get("McConfEmpAcc", 0) if self.logging else False

        n_bins = 10
        thresholds = np.linspace(0.0, 1.0, n_bins, endpoint=False)

        if want_mc_conf_rel:
            mc_hits = np.zeros(n_bins)
            mc_tot  = np.zeros(n_bins)

        if want_iou_curve:
            all_ious = []

        if want_entropy_rel:
            ent_err_tot = np.zeros(n_bins)
            ent_err_err = np.zeros(n_bins)

        for batch in tqdm.tqdm(dataloader, desc=f"Testing (epoch {epoch})"):
            range_img, reflectivity, xyz, normals, semantic = batch
            range_img, reflectivity = range_img.to(self.device), reflectivity.to(self.device)
            xyz, normals = xyz.to(self.device), normals.to(self.device)
            semantic = semantic.to(self.device)

            inputs = set_model_inputs(range_img, reflectivity, xyz, normals, self.cfg)

            if use_dropout:
                # MC-dropout; if T is known, scale logits before softmax
                # and compute predictive entropy for uncertainty plots if needed
                T_s = max(1e-3, self.T_value) if self.T_value is not None else None
                probs_list = []
                if hasattr(self.model, "forward_logits"):
                    # keep BN eval; enable only dropout in your call-site if needed
                    import torch.nn as nn
                    self.model.eval()
                    for m in self.model.modules():
                        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d,
                                        nn.AlphaDropout, nn.FeatureAlphaDropout)):
                            m.train()
                    for _ in range(30):
                        logits_t = self.model.forward_logits(*inputs)
                        if T_s is not None:
                            logits_t = logits_t / T_s
                        probs_list.append(torch.softmax(logits_t, dim=1).unsqueeze(0))
                    probs_MC = torch.cat(probs_list, dim=0)             # [T,B,C,H,W]
                else:
                    probs_MC = mc_dropout_probs(self.model, inputs, T=30)  # assumes probs out
                probs = probs_MC.mean(dim=0)                              # [B,C,H,W]
                ent_mc = predictive_entropy_mc(probs_MC)                     # [B,H,W] normalized

            else:
                probs = self._forward_probs(inputs)                        # [B,C,H,W]
                ent_mc = None

            # predictions & evaluator update
            preds = probs.argmax(dim=1)                                    # [B,H,W]
            self.evaluator.update(preds, semantic)

            if not self.logging:
                continue

            # ---- optional uncertainty diagnostics (Dirichlet branch or MC) ----
            if self.loss_function == "Dirichlet":
                alpha = to_alpha_concentrations(probs)      # if probs are logits replace w/ logits->alpha
                pred_entropy = get_predictive_entropy(alpha)             # [B,H,W]
                pred_entropy_norm = pred_entropy / math.log(self.num_classes)

                error_mask = (preds != semantic.squeeze(1)).int()

                if want_mc_conf_rel:
                    hits, tot = compute_mc_reliability_bins(alpha, semantic, n_bins=n_bins, n_samples=120)
                    mc_hits += hits
                    mc_tot  += tot

                if want_iou_curve:
                    ious = compute_entropy_error_iou(pred_entropy_norm, error_mask, thresholds=thresholds)
                    all_ious.append(ious.cpu().numpy())

                if want_entropy_rel:
                    tot, err, err_rate, ece = compute_entropy_reliability(pred_entropy_norm, error_mask)
                    ent_err_tot += tot
                    ent_err_err += err

            elif use_dropout and ent_mc is not None:
                error_mask = (preds != semantic.squeeze(1)).int()
                if want_iou_curve:
                    ious = compute_entropy_error_iou(ent_mc, error_mask, thresholds=thresholds)
                    all_ious.append(ious.cpu().numpy())

        # --- results ---
        mIoU, cls_dict = self.evaluator.compute_final_metrics(class_names=self.class_names)
        print(f"[Tester] mIoU: {mIoU:.6f}")

        # --- save ECE plot on the whole test set (single pass) ---
        if self.logging and out_dir:
            ece_img_path = os.path.join(out_dir, f"ece_epoch_{epoch:06d}.png")
            # Re-run a single pass to collect probs/labels for ECE (no MC, just calibrated probs if available)
            all_probs, all_labels = [], []
            for batch in dataloader:
                range_img, reflectivity, xyz, normals, labels = batch
                range_img, reflectivity = range_img.to(self.device), reflectivity.to(self.device)
                xyz, normals = xyz.to(self.device), normals.to(self.device)
                labels = labels.to(self.device)
                inputs = set_model_inputs(range_img, reflectivity, xyz, normals, self.cfg)
                p = self._forward_probs(inputs)
                all_probs.append(p.detach())
                all_labels.append(labels.detach())
            probs_full = torch.cat(all_probs, dim=0)
            labels_full = torch.cat(all_labels, dim=0)
            ece = reliability_diagram_from_probs(
                probs_full, labels_full,
                ignore_index=255, n_bins=15,
                title="Reliability Diagram (test)",
                save_path=ece_img_path, show=False
            )
            print(f"[Tester] ECE (test): {ece:.4f}")

        # --- optional uncertainty plots ---
        if self.logging and out_dir:
            if want_mc_conf_rel:
                emp_acc = np.divide(mc_hits, mc_tot, out=np.zeros_like(mc_hits), where=mc_tot > 0)
                edges = np.linspace(0.0, 1.0, n_bins+1)
                centers = 0.5 * (edges[:-1] + edges[1:])
                path = os.path.join(out_dir, f"reliability_mc_epoch_{epoch:06d}.png")
                save_reliability_diagram(emp_acc, centers, mc_tot, output_path=path)

            if want_entropy_rel:
                emp_err = np.divide(ent_err_err, ent_err_tot, out=np.zeros_like(ent_err_err), where=ent_err_tot>0)
                edges = np.linspace(0.0, 1.0, n_bins+1)
                centers = 0.5 * (edges[:-1] + edges[1:])
                path = os.path.join(out_dir, f"entropy_error_reliability_epoch_{epoch:06d}.png")
                save_reliability_diagram(
                    empirical_acc = emp_err,
                    bin_centers   = centers,
                    tot_counts    = ent_err_tot,
                    output_path   = path,
                    title='Reliability of Entropy as Error Predictor\n(dot area ∝ #pixels per entropy bin)',
                    xlabel='Normalized Entropy',
                    ylabel='Observed Error Rate'
                )

            if want_iou_curve and len(all_ious) > 0:
                all_ious = np.stack(all_ious, axis=0)
                mean_ious = all_ious.mean(axis=0)
                path = os.path.join(out_dir, f"mean_iou_curve_epoch_{epoch:06d}.png")
                plot_mIOU_errorEntropy(mean_ious, thresholds, output_path=path)

        return mIoU, cls_dict

    # convenience wrapper
    def run(self, dataloader_test, calib_loader=None, do_calibration: bool = False,
            ts_mode: str = "default", mc_samples: int = 30):
        """
        If do_calibration=True and calib_loader is given, fit T first.
        Then test one epoch and return (mIoU, class_metrics).
        """
        if do_calibration and calib_loader is not None:
            json_out = None
            if self.save_path:
                json_out = os.path.join(self.save_path, "eval", "temperature.json")
            self.calibrate_temperature(
                calib_loader,
                mode=ts_mode,
                mc_samples=mc_samples,
                optimizer_type="lbfgs",
                max_iter_lbfgs=100,
                epochs=2,
                chunk_size=1_000_000,
                init_T="auto",
                save_json=json_out
            )
        return self.test_epoch(dataloader_test, epoch=0)
