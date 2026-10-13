"""Detached, transport-aware observational appearance supervision for V44.

Future observations are used ONLY in training/diagnostic labels, never as net inputs.
These labels identify rain unexplained by one-step predicted pattern transport; they
are not physical convection labels and can change as detached motion improves.
"""
import math
import torch
import torch.nn.functional as F
from climax.regional_forecast.motion_core_arch import (
    compose_backward_step, safe_grid_sample, pixel_center_grid,
)


def _erode(valid, radius):
    if radius <= 0:
        return valid.bool()
    invalid = F.pad((~valid.bool()).float(), (radius,) * 4, value=1.)
    return F.max_pool2d(invalid, 2 * radius + 1, stride=1) == 0


@torch.no_grad()
def transport_targets(last, truth, last_valid, valid, velocity, domain_km,
                      neighborhood_km, border_km, threshold=.1):
    """Return [B,L,1,H,W] targets. Existing transported wet neighborhoods ignored.

    Current dry pixels remain negative controls. Current wet pixels are positive
    only outside transported previous wet neighborhoods. Validity is conservatively
    eroded over the same neighborhood, with explicit source/path/border exclusions.
    """
    truth, last = truth.detach().float(), last.detach().float()
    b, leads, _, h, w = truth.shape
    previous = torch.cat((last[:, None], truth[:, :-1]), dim=1)
    previous_valid = torch.cat((last_valid[:, None], valid[:, :-1]), dim=1)
    previous = previous.flatten(0, 1)
    pv = ((previous_valid.flatten(0, 1) >= .999) &
          torch.isfinite(previous) & (previous >= 0))
    cv = (valid.flatten(0, 1) >= .999)
    flow = velocity.detach().float().flatten(0, 1)
    flow = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=False)
    disp, path_valid = compose_backward_step(torch.zeros_like(flow), flow, domain_km,
                                             n_substeps=3, return_support=True)
    grid = pixel_center_grid(h, w, truth.device)[None] - disp.permute(0, 2, 3, 1) * (2. / domain_km)
    warped = safe_grid_sample(torch.where(pv, previous, 0.), grid)
    warped_valid = safe_grid_sample(pv.float(), grid) >= .999
    radius = max(0, int(round(neighborhood_km / (domain_km / w))))
    # Warp occupancy, not a thresholded diluted amount: bilinear mixing of a
    # marginal .1-mm source must not falsely label transported rain as genesis.
    wet = safe_grid_sample(((previous >= threshold) & pv).float(), grid) > 1e-6
    near_wet = F.max_pool2d(wet.float(), 2 * radius + 1, stride=1, padding=radius) > 0
    trustworthy = _erode(cv & warped_valid & path_valid, radius)
    border = max(radius, int(round(border_km / (domain_km / w))))
    if border:
        trustworthy[..., :border, :] = False
        trustworthy[..., -border:, :] = False
        trustworthy[..., :, :border] = False
        trustworthy[..., :, -border:] = False
    current = truth.flatten(0, 1)
    trustworthy &= torch.isfinite(current) & (current >= 0)
    positive = trustworthy & (current >= threshold) & ~near_wet
    negative = trustworthy & (current < threshold)
    mask = positive | negative
    return dict(target_bool=positive.unflatten(0, (b, leads)),
                mask=mask.unflatten(0, (b, leads)),
                amount=torch.where(positive, current, 0.).unflatten(0, (b, leads)),
                transported=warped.unflatten(0, (b, leads)))


@torch.no_grad()
def birth_targets(batch, out, threshold=.1, include_fine=True):
    """Shared train/evaluation labels; no assumption of native fine coverage.

    Fine input coverage projects fully covered wide cells conservatively, matching
    data_v39. Future fine coverage comes from the dataset's existing fine_valid.
    """
    context_last = batch['context'][:, -1]
    wide_last_valid = batch['valid'][:, -1]
    if context_last.ndim == 3:
        context_last = context_last[:, None]
    if wide_last_valid.ndim == 3:
        wide_last_valid = wide_last_valid[:, None]
    wide = transport_targets(context_last, batch['wide_truth'], wide_last_valid,
                             batch['wide_valid'], out['velocity'], 400., 6., 20., threshold)
    if not include_fine:
        return dict(wide=wide)
    fine_last = batch['fine'][:, -1]
    if fine_last.ndim == 3:
        fine_last = fine_last[:, None]
    fine_h, fine_w = fine_last.shape[-2:]
    wide_h, wide_w = context_last.shape[-2:]
    rows = (((torch.arange(fine_h, device=fine_last.device) + .5) / fine_h * 50. + 175.) / 400. * wide_h).floor().long()
    cols = (((torch.arange(fine_w, device=fine_last.device) + .5) / fine_w * 50. + 175.) / 400. * wide_w).floor().long()
    fine_last_valid = wide_last_valid[..., rows[:, None], cols[None, :]] >= .999
    fine_valid = batch['fine_valid']
    return dict(
        wide=wide,
        fine=transport_targets(fine_last, batch['truth'], fine_last_valid,
                               fine_valid, out['fine_velocity'], 50., 1.5, 3., threshold),
    )


@torch.no_grad()
def current_occurrence_targets(truth, valid, threshold=.1):
    """Current-lead rain presence, including persistent rain and valid borders.

    This auxiliary classification target does not define an additive rain source
    or a calibrated natural-frequency probability.
    """
    current = truth.detach().float()
    mask = (valid >= .999) & torch.isfinite(current) & (current >= 0.)
    positive = mask & (current >= threshold)
    return dict(target_bool=positive, mask=mask,
                amount=torch.where(positive, current, 0.))


def balanced_location_loss(logits, target):
    """Mean positive and negative BCE, each half when both classes are present.

    BCE-with-logits keeps useful positive gradients even at very negative logits.
    Single-class/all-invalid support is explicit; all-dry batches penalize births.
    """
    logits = logits.float()
    pos = target['target_bool'].float()
    mask = target['mask'].float()
    neg = mask * (1. - pos)
    np, nn = pos.sum(), neg.sum()
    loss = F.binary_cross_entropy_with_logits(logits, pos, reduction='none')
    present = (np > 0).float() + (nn > 0).float()
    value = ((loss * pos).sum() / np.clamp_min(1.) +
             (loss * neg).sum() / nn.clamp_min(1.)) / present.clamp_min(1.)
    return value, dict(positive_pixels=np.detach(), negative_pixels=nn.detach(),
                       valid_pixels=mask.sum().detach())


def positive_amount_loss(prediction, target):
    positive = target['target_bool'].float()
    # FP32 log1p loss controls heavy-rain dynamic range; amount and probability
    # are separately supervised so a tiny occurrence probability cannot block it.
    # Match the V2 negative log-L1 scale: SmoothL1's near-zero derivative had
    # suppressed weak new rain despite positive support. V3 changes only this
    # shared positive penalty; labels, weights and negative controls stay fixed.
    loss = F.l1_loss(torch.log1p(prediction.float().clamp_min(0.)),
                    torch.log1p(target['amount'].float()), reduction='none')
    return (loss * positive).sum() / positive.sum().clamp_min(1.)


def negative_final_amount_loss(prediction, target, top_fraction=.05, truth=None):
    """Penalize final rain on occurrence-negative pixels.

    Half all-negative log-L1 mean and half largest 5% mean. log1p has useful
    derivative even below the wet threshold; hard negatives prevent spatially
    concentrated false rain from disappearing in an all-domain average. With
    ``truth`` supplied, only excess above actual rainfall is penalized, with
    zero gradient at an exact match. Without it, legacy zero-target semantics
    are retained for old diagnostic callers.
    """
    negative = target['mask'].bool() & ~target['target_bool'].bool()
    # Optional actual truth distinguishes below-threshold occurrence negatives
    # from zero rainfall. Preserve legacy callers; V45 penalizes only excess.
    excess = torch.log1p(prediction.float().clamp_min(0.))
    if truth is not None:
        excess = F.relu(excess - torch.log1p(truth.detach().float().clamp_min(0.)))
    values = excess[negative]
    count = values.numel()
    top_count = max(1, int(math.ceil(top_fraction * count))) if count else 0
    if count:
        value = .5 * values.mean() + .5 * values.topk(top_count, sorted=False).values.mean()
    else:
        value = prediction.float().sum() * 0.
    return value, dict(negative_pixels=count, hard_negative_pixels=top_count)


def dry_fine_targets(batch, target, threshold=.1):
    """Same detached labels, separately normalized on past-fine-all-dry windows."""
    with torch.no_grad():
        dry = batch['fine'].detach().flatten(1).amax(1) < threshold
        select = dry[:, None, None, None, None]
        restricted = dict(target)
        restricted['mask'] = target['mask'].bool() & select
        restricted['target_bool'] = target['target_bool'].bool() & select
        restricted['amount'] = torch.where(select, target['amount'], 0.)
    return restricted, dry.sum()
