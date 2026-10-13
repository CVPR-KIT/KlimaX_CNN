"""Inherited trainer, objectives, metrics and evidence callbacks used by V45.

The legacy standalone main retains its older experiment budget. Use train_v45.py
for the published H6 / 18-lead / 20-epoch contract.
"""
import argparse
import hashlib
import json
import math
import os
import signal
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ClimaX/src'))
sys.path.insert(0, str(ROOT / 'experiments/scripts'))

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from climax.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from data_v39 import DataModuleV39, CONTEXT_RES_KM

QUANTILES = (0.5, 0.6, 0.75, 0.9, 0.95)
QUANTILE_WEIGHTS = (1.0, 1.0, 0.75, 0.5, 0.25)
WIDE_QUANTILE = 0.6
WET_THRESHOLD_MM = 0.1                # mm/10min; matches the corpus-wide convention
FSS_KERNELS_KM = (4.0, 8.0, 16.0, 32.0)   # REQUESTED nominal sizes; see FSS_KERNELS_PX
FIXED_EFFECTIVE_BATCH = 16
FIXED_EPOCHS = 30
FIXED_HISTORY = 18

# A wide-resolution cell only counts as a trustworthy GT observation if its coverage is
# (numerically) complete; partial coverage is EXCLUDED from every wide-based loss/target/
# metric, never down-weighted as if it were a fractional-but-usable observation ("do not
# trust partial coverage as full rain").
FULL_COVERAGE_THRESHOLD = 0.999
# core_target's own physical support (gaussian_blur ~3*sigma_km=9km plus the
# peak_radius_km=6km local-max window) reaches roughly (9+6)/2km = ~8 wide (2 km) cells
# out from any given pixel; the GT validity mask passed to it is eroded by this many
# cells so a nominally-valid pixel whose blur/local-max neighbourhood dips into invalid
# territory is not silently treated as a trustworthy core label.
CORE_TARGET_VALID_ERODE_CELLS = 8


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().float().numpy().tobytes())
    return h.hexdigest()


def _km_to_odd_pixels(km, res_km):
    px = max(1, int(round(float(km) / float(res_km))))
    return px if px % 2 == 1 else px + 1


FSS_KERNELS_PX = tuple(_km_to_odd_pixels(k, CONTEXT_RES_KM) for k in FSS_KERNELS_KM)


# --------------------------------------------------------------------------------- #
# Loss terms. Every weight is a fixed, recorded constructor default (Module.__init__
# below); none are tuned against validation results.
# --------------------------------------------------------------------------------- #

def masked_quantile_pinball(pred, truth, mask, quantiles, weights, eps=1e-6):
    """pred [B,L,Q,H,W]; truth,mask [B,L,1,H,W] (broadcast over Q). Per-quantile masked
    mean pinball, combined with `weights` and renormalised by their sum -- the same
    shape of computation as climax.utils.metrics.rain_multi_quantile, without the
    lat-weighting (this grid is a fixed Cartesian ROI, not a lat/lon global grid).

    Honesty note (INTERFACE.md "avoid safe-zero losses if empty invalid support/nan"):
    the TRAINING scalar clamps its denominator to >= 1 so a batch with zero valid
    pixels contributes a harmless ~0 term instead of raising mid-epoch, but the actual
    valid pixel count is returned alongside so an all-invalid batch is VISIBLE in the
    logs, not silently reported as "0 loss = perfect".
    """
    if mask is None:
        raise ValueError('masked_quantile_pinball requires an explicit validity mask')
    pred, truth, mask = pred.float(), truth.float(), mask.float()
    valid_support = float(mask.sum().detach())
    denom = mask.sum().clamp_min(1.0)
    total = pred.new_zeros(())
    terms = {}
    for i, (q, w) in enumerate(zip(quantiles, weights)):
        residual = truth - pred[:, :, i:i + 1]
        pinball = torch.maximum(q * residual, (q - 1.0) * residual)
        term = (pinball * mask).sum() / denom
        total = total + w * term
        terms[f'pinball_q{q:g}'.replace('.', 'p')] = term.detach()
    total = total / max(sum(weights), eps)
    return total, terms, valid_support


def masked_core_loss(core_pred, core_target_soft, mask, gamma=2.0, eps=1e-6):
    """Balanced soft-target focal BCE for the core probability map. `core_target_soft`
    is a continuous [0,1] target (climax...motion_core_arch.core_target), not a hard
    label, so the focal modulation uses |target - pred| as the "how wrong" term rather
    than the usual hard-label p_t. "Balanced" means the positive (target>=0.5) and
    negative (target<0.5) populations are each renormalised to contribute 0.5 of the
    total regardless of how rare core pixels are, before the mask is applied; an
    all-one-class mask (only pos or only neg present) degrades gracefully to a plain
    masked mean over the present class, flagged via the returned valid_support.

    `core_pred`/`core_target_soft`/`mask` are cast to float32 BEFORE clamp/log: under
    bf16 autocast the model's raw core-probability output can carry values close enough
    to 0/1 that bf16's ~3-decimal-digit precision makes `clamp`+`log`/`log1p` produce
    NaN/Inf (this is exactly the failure mode, not a hypothetical one -- bf16 `log(x)`
    near a clamped boundary is numerically unsafe). The loss itself is always computed
    in fp32 regardless of the surrounding autocast dtype.
    """
    core_pred = core_pred.float()
    core_target_soft = core_target_soft.float()
    mask = mask.float()
    pred = core_pred.clamp(eps, 1.0 - eps)
    target = core_target_soft.clamp(0.0, 1.0)
    bce = -(target * torch.log(pred) + (1.0 - target) * torch.log1p(-pred))
    pt = torch.where(target >= 0.5, pred, 1.0 - pred)
    focal = (1.0 - pt).clamp_min(0.0) ** gamma
    pos = (target >= 0.5).to(pred.dtype) * mask
    neg = (target < 0.5).to(pred.dtype) * mask
    pos_count, neg_count = pos.sum().clamp_min(1.0), neg.sum().clamp_min(1.0)
    class_weight = pos * (0.5 / pos_count) + neg * (0.5 / neg_count)
    valid_support = float(mask.sum().detach())
    total = (bce * focal * class_weight).sum()
    return total, valid_support


def multiscale_structure_loss(pred, truth, mask, kernels_px, eps=1e-6):
    """Differentiable multiscale proxy for the (non-differentiable, thresholded) FSS
    reported at evaluation: local-neighbourhood-mean squared error of the RAW
    (non-binarised) field at the same physical kernel sizes, masked to neighbourhoods
    that are fully covered at that scale. This is explicitly NOT a differentiable FSS
    on binarised fields -- it optimises multiscale structure agreement, and evaluate_v39
    reports the literal thresholded FSS separately so training objective and reported
    metric are never conflated.
    """
    # pred/truth/mask are [B,L,1,H,W]; avg_pool2d needs 4D, so leads fold into batch.
    pred_f, truth_f, mask_f = pred.flatten(0, 1), truth.flatten(0, 1), mask.flatten(0, 1)
    total = pred.new_zeros(())
    terms = {}
    for k in kernels_px:
        k = int(k)
        if k <= 1:
            p, t, m = pred_f, truth_f, mask_f
        else:
            pad = k // 2
            p = F.avg_pool2d(F.pad(pred_f, (pad, pad, pad, pad), mode='replicate'), k, stride=1)
            t = F.avg_pool2d(F.pad(truth_f, (pad, pad, pad, pad), mode='replicate'), k, stride=1)
            m = F.avg_pool2d(F.pad(mask_f, (pad, pad, pad, pad), mode='replicate'), k, stride=1)
            m = (m >= (1.0 - eps)).to(pred.dtype)
        denom = m.sum().clamp_min(1.0)
        term = (((p - t) ** 2) * m).sum() / denom
        total = total + term
        terms[f'structure_k{k}'] = term.detach()
    return total / max(len(kernels_px), 1), terms


def optional_core_aux_loss(model_out, wide_truth, wide_valid, core_target_soft):
    """Required spatial strength and same-support per-window core area losses."""
    terms, total, used = {}, None, []

    strength_field = model_out.get('core_strength')
    if torch.is_tensor(strength_field) and strength_field.shape == wide_valid.shape:
        weight = core_target_soft * wide_valid
        denom = weight.sum().clamp_min(1e-6)
        per_pixel = F.smooth_l1_loss(strength_field, wide_truth, reduction='none')
        strength_term = (per_pixel * weight).sum() / denom
        terms['core_strength_aux'] = strength_term.detach()
        total = strength_term if total is None else total + strength_term
        used.append('core_strength')

    if not torch.is_tensor(strength_field) or strength_field.shape != wide_truth.shape:
        raise ValueError('required core_strength must match wide_truth')
    core_field = model_out['core'].float()
    if core_field.shape != core_target_soft.shape:
        raise ValueError('core area requires a spatial core probability field')
    # Use exactly the same observed support for predicted and target areas.
    denominator = wide_valid.sum((-1, -2, -3))
    predicted_area = (core_field * wide_valid).sum((-1, -2, -3)) / denominator.clamp_min(1)
    target_area = (core_target_soft * wide_valid).sum((-1, -2, -3)) / denominator.clamp_min(1)
    supported = (denominator > 0).float()
    area_term = (F.smooth_l1_loss(predicted_area, target_area.detach(), reduction='none') * supported).sum() / supported.sum().clamp_min(1)
    terms['core_area_aux'] = area_term.detach()
    total = strength_term + area_term
    used.append('core_area_masked')

    return total, terms, used


def core_valid_support(valid):
    """GT-only support for the complete blur + local prominence neighbourhood."""
    shape = valid.shape
    flat = (valid >= FULL_COVERAGE_THRESHOLD).float().reshape(-1, 1, *shape[-2:])
    r = CORE_TARGET_VALID_ERODE_CELLS
    # Outside the observed domain is unknown, including at its physical boundary.
    padded = F.pad(flat, (r, r, r, r), value=0.0)
    return (-F.max_pool2d(-padded, 2 * r + 1, stride=1)).reshape(shape)


class GlobalLeadMetrics:
    """Pooled (never batch-averaged) per-lead validation metrics over the WHOLE split.
    Ratios are computed once from running totals at the end -- a batch-mean of a
    per-batch CSI is a real statistical bias for rare events at small batch sizes (the
    same rationale climax.regional_forecast.module.py gives for its own epoch-total
    rain counts). A lead with zero pooled support for a given ratio is reported as NaN,
    never coerced to a "safe" 0 or 1.

    Core "centroid" metrics label a WHOLE-MAP, mass-weighted centroid (the centre of
    mass of the entire 200x200 core-probability field vs. the entire 200x200 soft core
    target, one point per window per lead) -- this is NOT a matched-object distance and
    carries no object identity/tracking of any kind (INTERFACE.md: "Dense split/merge is
    supported as map topology, not claimed object-ID tracking"). `core_centroid_defined`
    counts windows where BOTH the predicted and target mass were large enough to define
    a centroid at all (i.e. a comparison was even possible), not "correctly matched" in
    any correspondence sense.

    wide-resolution GT coverage below `FULL_COVERAGE_THRESHOLD` is treated as unusable
    (excluded, never down-weighted) for every wide-based ratio here, matching the
    training losses' own "do not trust partial coverage as full rain" rule.
    """

    def __init__(self, leads=18, wet_threshold=WET_THRESHOLD_MM, fss_kernels_px=FSS_KERNELS_PX,
                 core_bin_threshold=0.5, wide_res_km=CONTEXT_RES_KM):
        self.leads = int(leads)
        self.wet_threshold = float(wet_threshold)
        self.fss_kernels_px = tuple(int(k) for k in fss_kernels_px)
        self.core_bin_threshold = float(core_bin_threshold)
        self.wide_res_km = float(wide_res_km)
        z = lambda: np.zeros(self.leads, dtype=np.float64)
        self.tp, self.fp, self.fn = z(), z(), z()
        self.wet_pred, self.wet_truth, self.n_valid_fine = z(), z(), z()
        self.amount_pred, self.amount_truth = z(), z()
        self.fss_num = {k: z() for k in self.fss_kernels_px}
        self.fss_den = {k: z() for k in self.fss_kernels_px}
        self.core_intersection, self.core_union, self.n_valid_wide = z(), z(), z()
        self.core_pred_area_km2, self.core_truth_area_km2, self.n_windows_lead = z(), z(), z()
        self.core_strength_pred, self.core_strength_truth = z(), z()
        self.core_centroid_err_km_sum, self.core_centroid_defined = z(), z()
        self.n_windows = 0
        self._yy = self._xx = None

    def _coords(self, h, w, device):
        if self._yy is None or self._yy.shape != (h, w) or self._yy.device != device:
            cy = (torch.arange(h, device=device, dtype=torch.float64) - (h - 1) / 2.0) * self.wide_res_km
            cx = (torch.arange(w, device=device, dtype=torch.float64) - (w - 1) / 2.0) * self.wide_res_km
            self._yy, self._xx = torch.meshgrid(cy, cx, indexing='ij')
        return self._yy, self._xx

    @torch.no_grad()
    def update(self, fine_pred, truth, fine_valid, wide_pred, wide_truth, wide_valid, core_pred, core_target_soft):
        b = fine_pred.shape[0]
        self.n_windows += b
        fv = fine_valid > 0.5
        wp = (fine_pred >= self.wet_threshold) & fv
        wt = (truth >= self.wet_threshold) & fv
        hit = wp & wt
        # "do not trust partial coverage as full rain": any wide-resolution ratio here
        # treats a cell as usable only if its GT coverage is (numerically) complete.
        wv = (wide_valid >= FULL_COVERAGE_THRESHOLD).to(wide_pred.dtype)
        cv = core_valid_support(wv)
        wide_wet_pred = (wide_pred >= self.wet_threshold).to(wide_pred.dtype)
        wide_wet_truth = (wide_truth >= self.wet_threshold).to(wide_truth.dtype)
        core_bin_pred = (core_pred >= self.core_bin_threshold).to(core_pred.dtype)
        core_bin_truth = (core_target_soft >= self.core_bin_threshold).to(core_pred.dtype)
        cell_km2 = self.wide_res_km ** 2
        yy, xx = self._coords(wide_pred.shape[-2], wide_pred.shape[-1], wide_pred.device)
        # avg_pool2d needs 4D; fold leads into batch, then restore the lead axis.
        wp_f, wt_f, wv_f = wide_wet_pred.flatten(0, 1), wide_wet_truth.flatten(0, 1), wv.flatten(0, 1)
        fss_frac = {}
        for k in self.fss_kernels_px:
            if k <= 1:
                fss_frac[k] = (wide_wet_pred, wide_wet_truth, wv)
            else:
                pad = k // 2
                fp_ = F.avg_pool2d(F.pad(wp_f, (pad,) * 4, mode='replicate'), k, stride=1).unflatten(0, (b, self.leads))
                ft_ = F.avg_pool2d(F.pad(wt_f, (pad,) * 4, mode='replicate'), k, stride=1).unflatten(0, (b, self.leads))
                fm_ = F.avg_pool2d(F.pad(wv_f, (pad,) * 4, mode='replicate'), k, stride=1).unflatten(0, (b, self.leads))
                fss_frac[k] = (fp_, ft_, (fm_ >= (1.0 - 1e-6)).to(wv.dtype))
        for j in range(self.leads):
            self.tp[j] += float(hit[:, j].sum())
            self.fp[j] += float((wp[:, j] & ~wt[:, j]).sum())
            self.fn[j] += float((~wp[:, j] & wt[:, j]).sum())
            self.wet_pred[j] += float(wp[:, j].sum())
            self.wet_truth[j] += float(wt[:, j].sum())
            self.amount_pred[j] += float((fine_pred[:, j] * fv[:, j]).sum())
            self.amount_truth[j] += float((truth[:, j] * fv[:, j]).sum())
            self.n_valid_fine[j] += float(fv[:, j].sum())
            for k in self.fss_kernels_px:
                fp_, ft_, fm_ = fss_frac[k]
                m = fm_[:, j]
                self.fss_num[k][j] += float((((fp_[:, j] - ft_[:, j]) ** 2) * m).sum())
                self.fss_den[k][j] += float(((fp_[:, j] ** 2 + ft_[:, j] ** 2) * m).sum())
            cbp, cbt, m = core_bin_pred[:, j, 0], core_bin_truth[:, j, 0], (cv[:, j, 0] > 0.5)
            self.core_intersection[j] += float((cbp.bool() & cbt.bool() & m).sum())
            self.core_union[j] += float(((cbp.bool() | cbt.bool()) & m).sum())
            self.n_valid_wide[j] += float(m.sum())
            self.n_windows_lead[j] += b
            self.core_pred_area_km2[j] += float((cbp * m).sum()) * cell_km2
            self.core_truth_area_km2[j] += float((cbt * m).sum()) * cell_km2
            self.core_strength_pred[j] += float((wide_pred[:, j, 0] * cbp * m).sum())
            self.core_strength_truth[j] += float((wide_truth[:, j, 0] * cbt * m).sum())
            mv = m.to(torch.float64)
            pred_mass = core_pred[:, j, 0].double() * mv
            truth_mass = core_target_soft[:, j, 0].double() * mv
            for bi in range(b):
                pt, tt = pred_mass[bi].sum(), truth_mass[bi].sum()
                if pt <= 1e-6 or tt <= 1e-6:
                    continue
                py = (pred_mass[bi] * yy).sum() / pt
                px = (pred_mass[bi] * xx).sum() / pt
                ty = (truth_mass[bi] * yy).sum() / tt
                tx = (truth_mass[bi] * xx).sum() / tt
                self.core_centroid_err_km_sum[j] += float(torch.sqrt((py - ty) ** 2 + (px - tx) ** 2))
                self.core_centroid_defined[j] += 1.0

    def summary(self):
        def ratio(num, den):
            return [float(n / d) if d > 0 else float('nan') for n, d in zip(num, den)]

        csi = ratio(self.tp, self.tp + self.fp + self.fn)
        pod = ratio(self.tp, self.tp + self.fn)
        far = ratio(self.fp, self.tp + self.fp)
        wet_frac_truth = ratio(self.wet_truth, self.n_valid_fine)
        amount_ratio = ratio(self.amount_pred, self.amount_truth)
        fss = {k: ratio(self.fss_den[k] - self.fss_num[k], self.fss_den[k]) for k in self.fss_kernels_px}
        core_iou = ratio(self.core_intersection, self.core_union)
        core_area_ratio = ratio(self.core_pred_area_km2, self.core_truth_area_km2)
        core_strength_ratio = ratio(self.core_strength_pred, self.core_strength_truth)
        core_centroid_km = ratio(self.core_centroid_err_km_sum, self.core_centroid_defined)
        core_centroid_defined_rate = ratio(self.core_centroid_defined, self.n_windows_lead)
        pooled_tp, pooled_fp, pooled_fn = self.tp.sum(), self.fp.sum(), self.fn.sum()
        csi_global = float(pooled_tp / (pooled_tp + pooled_fp + pooled_fn)) if (pooled_tp + pooled_fp + pooled_fn) > 0 else float('nan')
        return dict(
            n_windows=self.n_windows, wet_threshold_mm=self.wet_threshold,
            csi_by_lead=csi, pod_by_lead=pod, far_by_lead=far,
            wet_fraction_truth_by_lead=wet_frac_truth, amount_ratio_by_lead=amount_ratio,
            fss_by_kernel_px_by_lead={str(k): v for k, v in fss.items()},
            # ACTUAL kernel size used (kernel_px * res_km), not the nominal requested km
            # (e.g. a requested 4 km kernel is rounded up to 3 odd pixels = 6 km actual).
            fss_kernel_km={str(k): round(k * self.wide_res_km, 3) for k in self.fss_kernels_px},
            core_iou_by_lead=core_iou, core_area_km2_ratio_by_lead=core_area_ratio,
            core_strength_ratio_by_lead=core_strength_ratio,
            core_centroid_km_by_lead=core_centroid_km,
            core_centroid_defined_rate_by_lead=core_centroid_defined_rate,
            csi_global=csi_global,
        )


class Module(pl.LightningModule):
    def __init__(self, lr=2e-4, weight_decay=1e-5, warmup_steps=2000, eta_min=1e-8,
                 forecast_steps=18, quantiles=QUANTILES, quantile_weights=QUANTILE_WEIGHTS,
                 hidden=48, fine_hidden=32, wide_quantile=WIDE_QUANTILE,
                 wet_threshold=WET_THRESHOLD_MM, core_res_km=CONTEXT_RES_KM, core_threshold=0.2,
                 core_bin_threshold=0.5, core_focal_gamma=2.0,
                 fss_kernels_km=FSS_KERNELS_KM,
                 loss_weight_fine=1.0, loss_weight_wide=0.3, loss_weight_core_bce=0.5,
                 loss_weight_core_aux=0.1, loss_weight_structure=0.2,
                 loss_weight_motion_photo=0.1, loss_weight_motion_fb=0.05,
                 loss_weight_motion_smooth=0.02, loss_weight_motion_temporal=0.02,
                 history_steps=6):
        super().__init__()
        self.save_hyperparameters()
        from climax.regional_forecast.motion_core_arch import MotionCoreNowcaster, core_target
        self._core_target_fn = core_target
        self.net = MotionCoreNowcaster(forecast_steps=forecast_steps, quantiles=tuple(quantiles),
                                       hidden=hidden, fine_hidden=fine_hidden, history_steps=history_steps)
        if float(wide_quantile) not in tuple(float(q) for q in quantiles):
            raise ValueError(f'wide_quantile {wide_quantile} must be one of the model quantiles {quantiles}')
        self._wide_quantile_index = tuple(quantiles).index(wide_quantile)
        self._fss_kernels_px = tuple(_km_to_odd_pixels(k, core_res_km) for k in fss_kernels_km)
        self._val_metrics = None
        self._last_terms = {}

    def forward(self, fine, context, valid, clock):
        return self.net(fine, context, valid, clock)

    def _core_target(self, wide_truth, wide_valid):
        # The model helper performs its own neighbourhood exclusion exactly once.
        b, leads = wide_truth.shape[:2]
        target = self._core_target_fn(wide_truth.flatten(0, 1),
            res_km=self.hparams.core_res_km, threshold=self.hparams.core_threshold,
            valid=(wide_valid.flatten(0, 1) >= FULL_COVERAGE_THRESHOLD).float())
        return target.unflatten(0, (b, leads))

    def _compute_losses(self, batch, out):
        fine_pred, wide_pred_full = out['fine'], out['wide']
        wide_pred = wide_pred_full if wide_pred_full.shape[2] == 1 else wide_pred_full[:, :, self._wide_quantile_index:self._wide_quantile_index + 1]
        core_pred = out['core']
        truth, fine_valid = batch['truth'], batch.get('fine_valid')
        if fine_valid is None:
            fine_valid = torch.ones_like(truth)
        wide_truth, wide_valid_raw = batch['wide_truth'], batch['wide_valid']
        # "do not trust partial coverage as full rain": every wide-based loss below uses
        # this thresholded mask, never the raw fractional coverage as a soft weight.
        wide_valid = (wide_valid_raw >= FULL_COVERAGE_THRESHOLD).to(wide_truth.dtype)

        fine_loss, fine_terms, fine_support = masked_quantile_pinball(
            fine_pred, truth, fine_valid, self.hparams.quantiles, self.hparams.quantile_weights)
        # wide_pred is [B,L,1,H,W] already (contract: single central q=.6 field), i.e.
        # exactly the [B,L,Q=1,H,W] shape masked_quantile_pinball expects -- no reshape.
        wide_loss, wide_terms, wide_support = masked_quantile_pinball(
            wide_pred, wide_truth, wide_valid, (self.hparams.wide_quantile,), (1.0,))

        core_soft_target = self._core_target(wide_truth, wide_valid)
        # masked_core_loss already renormalises pos/neg contributions to sum to ~1
        # (see its docstring); no further division by pixel count here.
        core_valid = core_valid_support(wide_valid)
        core_loss, core_support = masked_core_loss(core_pred, core_soft_target, core_valid,
                                                    gamma=self.hparams.core_focal_gamma)

        structure_loss, structure_terms = multiscale_structure_loss(
            wide_pred, wide_truth, wide_valid, self._fss_kernels_px)

        aux_loss, aux_terms, aux_used = optional_core_aux_loss(out, wide_truth, core_valid, core_soft_target)

        aux_losses = out.get('aux_losses', {})
        motion_weights = dict(motion_photo=self.hparams.loss_weight_motion_photo,
                              motion_fb=self.hparams.loss_weight_motion_fb,
                              motion_smooth=self.hparams.loss_weight_motion_smooth,
                              motion_temporal=self.hparams.loss_weight_motion_temporal)
        total = (self.hparams.loss_weight_fine * fine_loss
                + self.hparams.loss_weight_wide * wide_loss
                + self.hparams.loss_weight_core_bce * core_loss
                + self.hparams.loss_weight_structure * structure_loss)
        terms = {'loss/fine_pinball': fine_loss.detach(), 'loss/wide_pinball': wide_loss.detach(),
                 'loss/core_bce': core_loss.detach(), 'loss/structure': structure_loss.detach(),
                 'support/fine_valid_pixels': fine_support, 'support/wide_valid_pixels': wide_support,
                 'support/core_valid_pixels': core_support}
        for name, w in motion_weights.items():
            if name not in aux_losses:
                raise KeyError(f"model aux_losses missing required key '{name}' (INTERFACE.md contract)")
            term = aux_losses[name]
            total = total + w * term
            terms[f'loss/{name}'] = term.detach()
        if aux_loss is not None:
            total = total + self.hparams.loss_weight_core_aux * aux_loss
            terms['loss/core_aux'] = aux_loss.detach()
        terms['core_aux_fields_used'] = aux_used
        terms.update({f'loss/{k}': v for k, v in fine_terms.items()})
        terms.update({f'wide_{k}': v for k, v in wide_terms.items()})
        terms.update(structure_terms)
        terms.update(aux_terms)
        if not torch.isfinite(total):
            raise RuntimeError(f'non-finite total loss: {terms}')
        return total, terms, fine_pred, wide_pred, core_pred, core_soft_target

    def training_step(self, batch, batch_idx):
        out = self(batch['fine'], batch['context'], batch['valid'], batch['clock'])
        total, terms, *_ = self._compute_losses(batch, out)
        self._last_terms = {k: (float(v) if torch.is_tensor(v) else v) for k, v in terms.items()}
        self.log('train/loss', total, on_step=True, on_epoch=False, prog_bar=True)
        for k, v in terms.items():
            if torch.is_tensor(v) and v.numel() == 1:
                self.log(f'train/{k}', v, on_step=True, on_epoch=False, prog_bar=False)
        return total

    def on_validation_epoch_start(self):
        self._val_metrics = GlobalLeadMetrics(
            leads=self.hparams.forecast_steps, wet_threshold=self.hparams.wet_threshold,
            fss_kernels_px=self._fss_kernels_px, core_bin_threshold=self.hparams.core_bin_threshold,
            wide_res_km=self.hparams.core_res_km)
        self._val_loss_sum, self._val_loss_n = 0.0, 0
        # V45 selection adds pooled strong-rain counts without changing old metrics.
        self._val_strong_counts = np.zeros((3, self.hparams.forecast_steps), dtype=np.float64)

    def validation_step(self, batch, batch_idx):
        out = self(batch['fine'], batch['context'], batch['valid'], batch['clock'])
        total, terms, fine_pred, wide_pred, core_pred, core_soft_target = self._compute_losses(batch, out)
        fine_valid = batch.get('fine_valid')
        if fine_valid is None:
            fine_valid = torch.ones_like(batch['truth'])
        # score the fine forecast at q=.6 (the contract's "central forecast" quantile,
        # matching the wide field), NOT index 0 (q=.5).
        qi = self._wide_quantile_index
        with torch.no_grad():
            strong_pred = fine_pred[:, :, qi:qi + 1] >= 0.5
            strong_truth = batch['truth'] >= 0.5
            mask = fine_valid > 0.5
            for row, event in enumerate((strong_pred & strong_truth,
                                         strong_pred & ~strong_truth,
                                         ~strong_pred & strong_truth)):
                self._val_strong_counts[row] += (event & mask).sum((0, 2, 3, 4)).double().cpu().numpy()
        self._val_metrics.update(fine_pred[:, :, qi:qi + 1], batch['truth'], fine_valid,
                                 wide_pred, batch['wide_truth'], batch['wide_valid'],
                                 core_pred, core_soft_target)
        self._val_loss_sum += float(total.detach()) * batch['truth'].shape[0]
        self._val_loss_n += batch['truth'].shape[0]
        return total

    def on_validation_epoch_end(self):
        summary = self._val_metrics.summary()
        strong = self._val_strong_counts
        denominator = strong.sum(axis=0)
        strong_global = float(strong[0].sum() / denominator.sum()) if denominator.sum() else float('nan')
        # Index 11 is +120 minutes for ten-minute output intervals.
        late_denominator = denominator[11:].sum()
        strong_late = float(strong[0, 11:].sum() / late_denominator) if late_denominator else float('nan')
        summary.update(csi_global_threshold_0p5=strong_global,
                       csi_late120_180_threshold_0p5=strong_late,
                       strong_counts_by_lead=strong.tolist(),
                       selection_score=float(np.mean([summary['csi_global'], strong_global, strong_late])))
        self.log('val/csi_global_0p5', strong_global)
        self.log('val/csi_late120_180_0p5', strong_late)
        self.log('val/selection_score', summary['selection_score'], prog_bar=True)
        self.log('val/loss', self._val_loss_sum / max(self._val_loss_n, 1), prog_bar=True)
        self.log('val/csi_global', summary['csi_global'], prog_bar=True)
        for j in range(self.hparams.forecast_steps):
            self.log(f'val/csi_lead{j:02d}', summary['csi_by_lead'][j])
            self.log(f'val/pod_lead{j:02d}', summary['pod_by_lead'][j])
            self.log(f'val/far_lead{j:02d}', summary['far_by_lead'][j])
        self._last_val_summary = summary

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.hparams.lr,
                                      weight_decay=self.hparams.weight_decay, betas=(0.9, 0.999))
        total_steps = self.trainer.max_steps
        if not isinstance(total_steps, int) or total_steps <= 0:
            raise ValueError(f'trainer.max_steps must be a finite positive int, got {total_steps!r}')
        warmup = min(self.hparams.warmup_steps, total_steps - 1)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup, total_steps, eta_min=self.hparams.eta_min)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}}


# Top-level MotionCoreNowcaster submodules the smoke gradient check expects to see
# (climax.regional_forecast.motion_core_arch.MotionCoreNowcaster.__init__); used ONLY to
# catch a whole branch being disconnected from the graph or removed by accident, never
# to police individual step-0 zero-gradients (see Evidence.on_before_optimizer_step).
EXPECTED_BRANCHES = (
    'motion_estimator', 'wide_encoder', 'lead_embed', 'clock_proj', 'wide_from_core',
    'wide_cell', 'wide_velocity_head', 'wide_source_sink_head', 'wide_birth_head', 'core_init',
    'core_from_wide', 'core_cell', 'core_growth_head', 'core_birth_head',
    'core_death_head', 'core_strength_head', 'fine_skip_encoder',
    'fine_velocity_correction', 'fine_decoder',
)


class Evidence(pl.Callback):
    """Atomic progress.json every step (small; JSONL for validation.jsonl), source and
    manifest hashes, initial state hash, split counts, parameter count and periodic data
    fingerprints -- written once at fit start and appended to during training."""

    def __init__(self, out, smoke, smoke_steps=0):
        self.out = Path(out)
        self.smoke = bool(smoke)
        self.smoke_steps = int(smoke_steps)
        self.t0 = time.time()
        self.fingerprints = []
        self._branch_grad_sum = {}
        self._smoke_optimizer_steps = 0

    def on_fit_start(self, trainer, module):
        if self.smoke:
            # A branch entirely absent from the parameter list at all is a real wiring
            # bug (a submodule renamed/removed), checkable immediately -- unlike a
            # temporarily-zero AGGREGATE gradient, which needs the full smoke budget
            # before it means anything (see on_before_optimizer_step below).
            present = {n.split('.', 1)[0] for n, _ in module.net.named_parameters()}
            missing = [b for b in EXPECTED_BRANCHES if b not in present]
            if missing:
                raise RuntimeError(f'smoke: expected branch(es) missing from model parameters '
                                   f'entirely: {missing} (present top-level modules: {sorted(present)})')

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        if batch_idx < 5 or trainer.global_step % 500 == 0:
            self.fingerprints.append(dict(step=int(trainer.global_step),
                fine_sum=float(batch['fine'].double().sum()), truth_sum=float(batch['truth'].double().sum())))
            write_json(self.out / 'data_fingerprint.json', self.fingerprints)

    def on_before_optimizer_step(self, trainer, module, optimizer, optimizer_idx=0):
        named = [(n, p) for n, p in module.net.named_parameters() if p.grad is not None]
        if not named:
            raise RuntimeError('no parameter received a gradient')
        grads = torch.stack([p.grad.detach().float().norm() for _, p in named])
        if not torch.isfinite(grads).all():
            bad = [n for (n, _), g in zip(named, grads.tolist()) if not math.isfinite(g)]
            raise RuntimeError(f'non-finite gradient before optimizer step: {bad[:10]}')
        if self.smoke:
            self._smoke_optimizer_steps += 1
            for n, p in named:
                branch = n.split('.', 1)[0]
                self._branch_grad_sum[branch] = (self._branch_grad_sum.get(branch, 0.0)
                                                 + float(p.grad.detach().abs().sum()))
            write_json(self.out / 'gradient_report.json', dict(
                optimizer_steps=self._smoke_optimizer_steps, smoke_step_budget=self.smoke_steps,
                branch_abs_grad_sum={k: v for k, v in sorted(self._branch_grad_sum.items())},
                expected_branches=list(EXPECTED_BRANCHES)))
            # AGGREGATE over the whole smoke run, decided only once the budget is spent:
            # a branch sitting immediately behind a zero-initialised final head (several
            # of MotionCoreNowcaster's output heads are deliberately zero-init) can
            # legitimately see zero upstream gradient for the first update or two, before
            # the optimizer nudges that head's weights off exactly zero -- rejecting that
            # at step 0 would be a false positive. A branch that is STILL all-zero after
            # the full smoke budget is a real dead/disconnected branch.
            if self._smoke_optimizer_steps >= max(self.smoke_steps, 1):
                dead = [b for b in EXPECTED_BRANCHES if self._branch_grad_sum.get(b, 0.0) == 0.0]
                if dead:
                    raise RuntimeError(f'smoke: branch(es) with zero AGGREGATE gradient over '
                                       f'{self._smoke_optimizer_steps} smoke updates: {dead}; not '
                                       'learnable as wired (see gradient_report.json)')

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite training loss')
        step = int(trainer.global_step)
        if step <= 3 or step % 25 == 0:
            row = dict(step=step, epoch=int(trainer.current_epoch), loss=float(loss.detach()),
                      elapsed_seconds=round(time.time() - self.t0, 2),
                      allocated_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 3) if torch.cuda.is_available() else None,
                      utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), terms=module._last_terms)
            write_json(self.out / 'progress.json', row)
            print('PROGRESS ' + json.dumps(row), flush=True)

    def on_validation_end(self, trainer, module):
        if trainer.sanity_checking:
            return
        summary = getattr(module, '_last_val_summary', None)
        row = dict(step=int(trainer.global_step), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  summary=summary)
        with (self.out / 'validation.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=True) + '\n')
        print('VALIDATION step', trainer.global_step, 'csi_global', summary['csi_global'] if summary else None, flush=True)


class EpochSampler(pl.Callback):
    """`DayBlockShuffleSampler` needs an explicit epoch to reshuffle deterministically;
    Lightning only auto-calls `set_epoch` on samplers it wraps for DDP, which single-GPU
    training never does."""

    def __init__(self, datamodule):
        self.dm = datamodule

    def on_train_epoch_start(self, trainer, module):
        self.dm.set_epoch(trainer.current_epoch)


class HistoryInputEvidence(pl.Callback):
    """Verify actual first-batch tensors, history channels and epoch sampler."""

    def __init__(self, out, dm, history):
        self.out, self.dm, self.history = Path(out), dm, int(history)

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        if batch_idx != 0:
            return
        assert self.dm.sampler.epoch == trainer.current_epoch
        indices = list(iter(self.dm.sampler))[:self.dm.batch_size]
        expected = torch.stack([self.dm.train[i]['fine'] for i in indices])
        assert torch.equal(batch['fine'].cpu(), expected)
        assert batch['fine'].shape[1:] == (self.history, 1, 100, 100)
        assert batch['context'].shape[1:] == (self.history, 1, 200, 200)
        assert module.net.wide_encoder[0].in_channels == 2 * self.history
        assert module.net.fine_skip_encoder[0].in_channels == self.history
        rows = []
        for i in indices:
            _, date, ref = self.dm.train.refs[i]
            assert ref - self.history + 1 >= 1 and ref + 18 < 144
            rows.append(dict(date=date, reference=ref,
                input_indices=list(range(ref-self.history+1, ref+1)),
                target_indices=list(range(ref+1, ref+19))))
        write_json(self.out / 'first_batch_verified.json', dict(
            epoch=int(trainer.current_epoch), history_steps=self.history,
            exact_tensor_match=True, fine_shape=list(batch['fine'].shape),
            context_shape=list(batch['context'].shape), indices=indices, windows=rows))


class SignalStop(pl.Callback):
    """Save when possible on SIGTERM/SIGINT; SIGKILL cannot be intercepted by any
    process and is not something this callback can fix."""

    def __init__(self, out):
        self.out = Path(out)
        self._stop = False
        signal.signal(signal.SIGTERM, self._handle)
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum, frame):
        self._stop = True

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if self._stop and not trainer.should_stop:
            trainer.should_stop = True
            print('SIGNAL received; stopping after this step', flush=True)

    def on_fit_end(self, trainer, module):
        if self._stop:
            trainer.save_checkpoint(str(self.out / 'signal_stop.ckpt'))
            print('saved signal_stop.ckpt', flush=True)


def build_trainer(out, total_steps, val_every, ckpt_every, limit_val_batches, smoke, extra_callbacks,
                  accumulate_grad_batches=1, limit_train_batches=None, max_epochs=-1,
                  checkpoint_monitor='val/csi_global'):
    # save_top_k=-1: NOTHING under best/ is ever deleted (every validation-time
    # improvement-candidate checkpoint survives); ModelCheckpoint still tracks the true
    # single best-by-monitor path/score internally (best_model_path/best_model_score,
    # written to DONE.json below) even though it keeps every copy rather than replacing
    # the previous one.
    best_checkpoint = ModelCheckpoint(dirpath=out / 'best', filename='best_step{step:05d}',
        monitor=checkpoint_monitor, mode='max', save_top_k=-1, save_weights_only=False,
        auto_insert_metric_name=False)
    callbacks = [LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(dirpath=out / 'checkpoints', filename='step{step:05d}',
            every_n_train_steps=min(ckpt_every, total_steps), save_top_k=-1,
            save_weights_only=False, auto_insert_metric_name=False),
        best_checkpoint, *extra_callbacks]
    # val_check_interval is in MICRO-batch units under accumulate_grad_batches>1 (PL 1.9
    # counts training_step calls, not optimizer updates); val_every/ckpt_every above are
    # in OPTIMIZER-step units by contract, so convert here rather than at every call site.
    val_check_interval_micro = min(val_every, total_steps) * max(int(accumulate_grad_batches), 1)
    trainer = pl.Trainer(default_root_dir=str(out), accelerator='auto', devices=1,
        precision='bf16' if torch.cuda.is_available() else 32,
        max_steps=total_steps, max_epochs=max_epochs, val_check_interval=val_check_interval_micro,
        limit_train_batches=limit_train_batches if limit_train_batches is not None else 1.0,
        limit_val_batches=limit_val_batches, num_sanity_val_steps=0, gradient_clip_val=1.0,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=1 if smoke else 25, logger=CSVLogger(str(out), name='logs'),
        callbacks=callbacks, enable_progress_bar=False)
    return trainer, best_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--root', required=True, help='Local corrected reset_v2 NPZ corpus directory')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--history-steps', type=int, default=FIXED_HISTORY)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--accumulate-grad-batches', type=int, default=1)
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=FIXED_EPOCHS)
    ap.add_argument('--seed', type=int, default=3901)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--warmup-steps', type=int, default=2000)
    ap.add_argument('--ckpt-every', type=int, default=1000)
    ap.add_argument('--val-every', type=int, default=0, help='0: validate each complete epoch')
    ap.add_argument('--smoke-steps', type=int, default=0)
    a = ap.parse_args()

    smoke = bool(a.smoke_steps)
    # Both the smoke check and the full run happen on the remote GPU server
    # (INTERFACE.md/PROTOCOL.md: "GPU smoke in train separately" from gate_v39.py's
    # own CPU-only synthetic checks). CPU is never a valid target for this script.
    if not torch.cuda.is_available():
        raise RuntimeError('train_v39.py requires the remote GPU server (L40S); CPU is gate_v39.py-only')
    gpu_name = torch.cuda.get_device_name()
    if 'TITAN' in gpu_name.upper():
        raise RuntimeError('local TITAN is excluded; L40S server only')

    # Resuming is disallowed for now: DayBlockShuffleSampler's exact-once-per-epoch
    # window coverage is only guaranteed by iterating it from batch 0 of an epoch, and
    # PL 1.9 does not fast-forward a resumed mid-epoch dataloader to the exact same
    # micro-batch position, so a mid-epoch resume cannot be shown to preserve "every
    # window exactly once" -- rather than silently drop that guarantee, --out is always
    # a NEW, exclusive directory and every run starts from a fresh optimizer/sampler
    # state (no fake "resumed" DONE.json).
    try:
        a.out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise SystemExit(f'--out {a.out} already exists; train_v39.py requires a new, exclusive '
                         'output directory for every run (no resume, no appending to prior evidence)')
    pl.seed_everything(a.seed, workers=True)
    torch.set_num_threads(4)

    if a.history_steps != FIXED_HISTORY or a.epochs != FIXED_EPOCHS:
        raise ValueError('V41 requires 18 input frames and 30 epochs')
    if a.smoke_steps not in (0, 8):
        raise ValueError('Smoke must exercise exactly 8 optimizer updates')
    dm = DataModuleV39(a.root, a.manifest, batch_size=a.batch_size, num_workers=a.num_workers,
                      seed=a.seed, history=a.history_steps)
    effective_batch = a.batch_size * a.accumulate_grad_batches
    per_epoch = len(dm.train) // effective_batch
    total = a.smoke_steps or a.epochs * per_epoch
    if effective_batch != FIXED_EFFECTIVE_BATCH or a.seed != 3901 or per_epoch < 1:
        raise ValueError('Fixed effective batch16 and seed3901 are required')
    with open(a.manifest) as fh:
        audit = json.load(fh)
    if audit.get('audit_pass') is not True:
        raise ValueError('Input-history audit must pass before training')
    expected_train = sum(len(r.get('train_indices', [])) for r in audit['days'] if r['split'] == 'train')
    expected_val = sum(len(r.get('eval_indices', [])) for r in audit['days'] if r['split'] == 'val')
    if (len(dm.train), len(dm.val)) != (expected_train, expected_val):
        raise ValueError('Missing selected data files or manifest/loader count mismatch')
    if a.val_every == 0:
        a.val_every = per_epoch
    if a.val_every < 1 or a.val_every > per_epoch:
        raise ValueError('Validation interval must fit inside one epoch')

    model = Module(lr=a.lr, warmup_steps=a.warmup_steps, history_steps=a.history_steps)
    init_hash = state_hash(model)
    source_files = {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(ROOT.rglob('*.py'))}
    write_json(a.out / 'source_hashes.json', source_files)
    # limit_train_batches, in MICRO-batch units: floor(N / effective_batch) * accum, so a
    # non-1 accumulate_grad_batches never leaves a partial accumulation group dangling at
    # epoch end (e.g. batch_size=8, accumulate=2, N=137436 gives 17179 micro-batches/epoch
    # naturally, which is ODD -- taking all of them would produce 8590 optimizer updates
    # for the last partial group instead of the intended 8589; this caps it to exactly
    # 17178 micro-batches = 8589*2, matching the fixed per-epoch optimizer-step budget).
    limit_train_batches = None if smoke else per_epoch * a.accumulate_grad_batches
    metadata = dict(arch='v41_rain_dates_motion_core_history18', smoke=smoke, pid=os.getpid(), host=socket.gethostname(),
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu', seed=a.seed,
        init='scratch', init_state_sha256=init_hash, parameters=sum(p.numel() for p in model.parameters()),
        data_root=a.root, manifest=a.manifest, manifest_sha256=sha256_file(a.manifest),
        data=dm.summary(), accumulate_grad_batches=a.accumulate_grad_batches, effective_batch=effective_batch,
        epochs=a.epochs, updates_per_epoch=per_epoch, total_updates=total,
        limit_train_batches_micro=limit_train_batches,
        lr=a.lr, warmup_steps=a.warmup_steps, ckpt_every=a.ckpt_every, val_every=a.val_every,
        precision='bf16' if torch.cuda.is_available() else 'fp32', forecast_steps=18,
        history_steps=a.history_steps, history_interval_minutes=10, input_accumulation_minutes=180,
        input_timestamp_span_minutes=170, planned_updates=a.epochs * per_epoch,
        fss_kernels_requested_km=list(FSS_KERNELS_KM), fss_kernels_px=list(FSS_KERNELS_PX),
        fss_kernels_actual_km=[round(px * CONTEXT_RES_KM, 3) for px in FSS_KERNELS_PX],
        hparams=dict(model.hparams), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    write_json(a.out / 'config.json', metadata)
    print(json.dumps(metadata, indent=2), flush=True)

    trainer, best_checkpoint = build_trainer(
        a.out, total, a.smoke_steps or a.val_every, a.ckpt_every,
        limit_val_batches=2 if smoke else 1.0, smoke=smoke,
        extra_callbacks=[Evidence(a.out, smoke, a.smoke_steps), EpochSampler(dm),
                         HistoryInputEvidence(a.out, dm, a.history_steps), SignalStop(a.out)],
        accumulate_grad_batches=a.accumulate_grad_batches,
        limit_train_batches=limit_train_batches, max_epochs=a.epochs)

    start = time.time()
    trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    if int(trainer.global_step) != total:
        raise RuntimeError(f'trainer stopped at step {trainer.global_step} before the authorized '
                           f'budget of {total}; exit status cannot claim completion')
    trainer.save_checkpoint(str(a.out / 'last.ckpt'))

    if smoke:
        reloaded = Module.load_from_checkpoint(str(a.out / 'last.ckpt'), map_location='cpu', strict=True)
        reloaded.eval()
        with torch.no_grad():
            batch = {k: v[:1] for k, v in next(iter(dm.val_dataloader())).items()}
            out = reloaded(batch['fine'], batch['context'], batch['valid'], batch['clock'])
        for key in ('fine', 'wide', 'core', 'velocity', 'confidence'):
            if key not in out:
                raise RuntimeError(f'strict reload check: model output missing required key {key!r}')
            if not torch.isfinite(out[key]).all():
                raise RuntimeError(f'strict reload check: non-finite output {key!r}')
        assert int(reloaded.hparams.history_steps) == a.history_steps
        assert batch['fine'].shape[1] == batch['context'].shape[1] == a.history_steps
        assert out['fine'].shape[1:] == (18, 5, 100, 100)
        assert out['wide'].shape[1:] == (18, 1, 200, 200)
        assert state_hash(reloaded) == state_hash(model)
        write_json(a.out / 'SMOKE_ACCEPTED.json', dict(history_steps=a.history_steps,
            forecast_steps=18, optimizer_updates=int(trainer.global_step), strict_reload=True,
            restored_state_identical=True, all_expected_branches_finite_positive=True,
            fine_input_shape=list(batch['fine'].shape), wide_input_shape=list(batch['context'].shape),
            fine_output_shape=list(out['fine'].shape), wide_output_shape=list(out['wide'].shape)))
        print('SMOKE strict reload accepted', flush=True)

    write_json(a.out / 'DONE.json', dict(
        steps=int(trainer.global_step), smoke=smoke, hours=(time.time() - start) / 3600,
        last_sha256=sha256_file(a.out / 'last.ckpt'),
        best_path=best_checkpoint.best_model_path,
        best_csi=float(best_checkpoint.best_model_score) if best_checkpoint.best_model_score is not None else None))
    print('DONE', str(a.out), int(trainer.global_step), flush=True)


if __name__ == '__main__':
    main()
