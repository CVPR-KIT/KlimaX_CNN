"""V45: final fine quantiles with auxiliary current-lead occurrence supervision."""
import inspect
import torch
from train_v39 import Module as BaseModule, QUANTILES, QUANTILE_WEIGHTS, WIDE_QUANTILE
from train_v39 import WET_THRESHOLD_MM, CONTEXT_RES_KM, FSS_KERNELS_KM
from birth_objective import (birth_targets, balanced_location_loss, positive_amount_loss,
                             negative_final_amount_loss, current_occurrence_targets)


class Module(BaseModule):
    def __init__(self, lr=2e-4, weight_decay=1e-5, warmup_steps=2000, eta_min=1e-8,
                 forecast_steps=18, quantiles=QUANTILES, quantile_weights=QUANTILE_WEIGHTS,
                 hidden=48, fine_hidden=32, wide_quantile=WIDE_QUANTILE,
                 wet_threshold=WET_THRESHOLD_MM, core_res_km=CONTEXT_RES_KM, core_threshold=.2,
                 core_bin_threshold=.5, core_focal_gamma=2., fss_kernels_km=FSS_KERNELS_KM,
                 loss_weight_fine=1., loss_weight_wide=.3, loss_weight_core_bce=.5,
                 loss_weight_core_aux=.1, loss_weight_structure=.2,
                 loss_weight_motion_photo=.1, loss_weight_motion_fb=.05,
                 loss_weight_motion_smooth=.02, loss_weight_motion_temporal=.02,
                 history_steps=18, birth_supervision=True, fine_birth_enabled=True,
                 loss_weight_birth_location_wide=.2, loss_weight_birth_location_fine=0.,
                 loss_weight_birth_amount_wide=.2, loss_weight_birth_amount_fine=0.,
                 loss_weight_appearance_wide=.3, loss_weight_appearance_fine=0.,
                 loss_weight_negative_final_wide=.3, loss_weight_negative_final_fine=0.,
                 loss_weight_dry_fine_location=0., loss_weight_dry_fine_amount=0.,
                 loss_weight_dry_fine_appearance=0., loss_weight_fine_occurrence=.2,
                 fine_correction_cap=20., fine_readout_floor=.05):
        arguments = dict(locals())
        arguments.pop('self')
        arguments.pop('__class__', None)
        base_keys = set(inspect.signature(BaseModule.__init__).parameters) - {'self'}
        super().__init__(**{k: arguments[k] for k in base_keys})
        removed = ('loss_weight_birth_location_fine', 'loss_weight_birth_amount_fine',
                   'loss_weight_appearance_fine', 'loss_weight_negative_final_fine',
                   'loss_weight_dry_fine_location', 'loss_weight_dry_fine_amount',
                   'loss_weight_dry_fine_appearance')
        if any(arguments[k] != 0. for k in removed):
            raise ValueError('V45 removed fine birth/amount/final/dry objectives; set their weights to zero')
        # Explicit flat named hyperparameters preserve strict checkpoint reload,
        # including every inherited objective and the ablation controls.
        self.save_hyperparameters(arguments)
        self.net.fine_birth_enabled = bool(fine_birth_enabled)
        if fine_correction_cap <= 0. or fine_readout_floor <= 0.:
            raise ValueError('fine correction cap and readout floor must be positive')
        self.net.fine_correction_cap = float(fine_correction_cap)
        self.net.fine_readout_floor = float(fine_readout_floor)

    def _compute_losses(self, batch, out):
        total, terms, *base_outputs = super()._compute_losses(batch, out)
        if not self.hparams.birth_supervision:
            return (total, terms, *base_outputs)
        labels = birth_targets(batch, out, threshold=self.hparams.wet_threshold, include_fine=False)
        # Keep the existing wide appearance/state objectives in this candidate.
        # Fine output has a single quantile readout: no additive-amount or
        # duplicate dry-window/final-appearance objectives remain.
        wide_pred = base_outputs[1]
        appearance = positive_amount_loss(wide_pred, labels['wide'])
        negative, negative_support = negative_final_amount_loss(
            wide_pred, labels['wide'], truth=batch['wide_truth'])
        location, support = balanced_location_loss(out['rain_birth_logit'], labels['wide'])
        amount = positive_amount_loss(out['rain_birth_amount'], labels['wide'])
        for name, value in (('appearance', appearance), ('negative_final', negative),
                            ('birth_location', location), ('birth_amount', amount)):
            total = total + self.hparams[f'loss_weight_{name}_wide'] * value
            terms[f'loss/{name}_wide'] = value.detach()
        for name, value in negative_support.items():
            terms[f'support/negative_final_wide_{name}'] = value
        for name, value in support.items():
            terms[f'support/birth_wide_{name}'] = value
        if self.hparams.fine_birth_enabled:
            truth = batch['truth']
            valid = batch.get('fine_valid', torch.ones_like(truth))
            occurrence = current_occurrence_targets(truth, valid, self.hparams.wet_threshold)
            location, support = balanced_location_loss(out['fine_birth_logit'], occurrence)
            total = total + self.hparams.loss_weight_fine_occurrence * location
            terms['loss/fine_occurrence'] = location.detach()
            for name, value in support.items():
                terms[f'support/fine_occurrence_{name}'] = value
        if not torch.isfinite(total):
            raise RuntimeError(f'non-finite V45 loss: {terms}')
        return (total, terms, *base_outputs)
