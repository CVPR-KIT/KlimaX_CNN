"""Focused V45 readout/state checks; run with Python + torch, no trainer deps."""
import importlib.util
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source/ClimaX/src/climax/regional_forecast/motion_core_arch.py'
spec = importlib.util.spec_from_file_location('v45_motion_model', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Model = module.MotionCoreNowcaster


def run():
    torch.set_num_threads(2)
    torch.manual_seed(4501)
    model = Model(forecast_steps=2)
    anchor = torch.tensor([0., .1, .5, 2., 10.]).reshape(1, 1, 1, 5)
    decoded = torch.zeros(1, model.n_quantiles + 1, 1, 5, requires_grad=True)
    with torch.no_grad():
        decoded[:, 1:model.n_quantiles] = -6.
    result = model._fine_quantile_readout(anchor, decoded)
    torch.testing.assert_close(result[:, 1:2], anchor, atol=0., rtol=0.)
    assert (result[:, 1:] >= result[:, :-1]).all()
    assert (result >= 0).all()
    result[:, 1:2].sum().backward()
    dry_grad = float(decoded.grad[0, 0, 0, 0])
    assert dry_grad > .1, dry_grad
    assert float(decoded.grad[:, model.n_quantiles:].abs().sum()) == 0.
    wet = decoded.detach().clone()
    wet[:, :1] = 1.
    created = model._fine_quantile_readout(anchor, wet)
    assert float(created[0, 1, 0, 0]) > .5
    strong = decoded.detach().clone()
    strong[:, :1] = 100.
    strong_created = model._fine_quantile_readout(anchor, strong)
    assert float(strong_created[0, 1, 0, 0]) > 6.47
    wet[:, :1] = -1.
    reduced = model._fine_quantile_readout(anchor, wet)
    assert (reduced[:, 1:2] <= anchor).all()
    # Changing only occurrence logits cannot alter the rainfall distribution.
    probability_only = decoded.detach().clone()
    probability_only[:, model.n_quantiles:] = 100.
    torch.testing.assert_close(model._fine_quantile_readout(anchor, probability_only), result.detach())
    assert model.fine_decoder[-1].out_channels == model.n_quantiles + 1
    assert model.fine_decoder[-1].weight[0].count_nonzero() == 0
    assert model.fine_decoder[-1].bias[0] == 0
    assert model.fine_birth_additive_enabled is False

    # Deterministic material test: coarse=2, native=1, residual=-1. Both
    # observed parts must undergo the same -0.1 log amplitude each interval.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.fine_decoder[-1].bias[1:model.n_quantiles].fill_(-6.)
        model.wide_source_sink_head[-1].bias.fill_(math.atanh(-.5))
        model.wide_birth_head[-1].bias.fill_(-100.)
        model.wide_birth_probability_head[-1].bias.fill_(-100.)
    fine = torch.ones(1, 6, 1, 100, 100)
    context = torch.full((1, 6, 1, 200, 200), 2.)
    valid = torch.ones_like(context)
    clock = torch.zeros(1, 3, 4)
    with torch.no_grad():
        out = model(fine, context, valid, clock, core_scale=0., region_scale=0.)
    for lead in range(2):
        expected = torch.full_like(out['fine'][:, lead, 1], math.exp(-.1 * (lead + 1)))
        torch.testing.assert_close(out['fine'][:, lead, 1], expected, atol=2e-6, rtol=2e-6)
    assert out['fine_new_rain'].count_nonzero() == 0
    assert out['fine_birth_amount'].count_nonzero() == 0
    assert torch.isfinite(out['fine']).all()
    assert (out['fine'][:, :, 1:] >= out['fine'][:, :, :-1]).all()

    # Actual forward/backward on completely dry observations: wet target can
    # open the shared readout without an amount branch or a hidden rain floor.
    train_model = Model(forecast_steps=1)
    dry_context = torch.zeros_like(context)
    dry_fine = torch.zeros_like(fine)
    train_out = train_model(dry_fine, dry_context, valid, clock[:, :2], core_scale=0.)
    objective = -train_out['fine'][:, :, 1].mean()
    objective += .2 * F.binary_cross_entropy_with_logits(
        train_out['fine_birth_logit'], torch.ones_like(train_out['fine_birth_logit']))
    objective.backward()
    center_bias_gradient = float(train_model.fine_decoder[-1].bias.grad[0])
    assert center_bias_gradient < -.1, center_bias_gradient
    occurrence_gradient = float(train_model.fine_decoder[-1].bias.grad[-1])
    assert occurrence_gradient < 0.
    grads = [p.grad for p in train_model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    report = dict(passed=True, dry_readout_gradient=dry_grad,
                  dry_generated_q06=float(created[0, 1, 0, 0]),
                  dry_strong_q06=float(strong_created[0, 1, 0, 0]),
                  forward_center_bias_gradient=center_bias_gradient,
                  occurrence_bias_gradient=occurrence_gradient,
                  residual_expected=[math.exp(-.1), math.exp(-.2)],
                  no_unused_trainable_parameters=True)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
