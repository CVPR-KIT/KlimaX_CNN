"""V39 shared rain-pattern motion, material core/rain evolution and native fine detail.
Physical dx/dy are km per 10 minutes. Hidden dynamics use 50x50 over400km;
observed material rain/core fields retain200x200@2km, and native detail100x100@500m.
Past frames are aligned to the anchor. Observed fields use material maps;
new rain/core sources live in ground coordinates and move only after birth.
The fine readout shares evolved wide rain plus advected native detail residual.
Motion is inferred rain-pattern displacement, not verified meteorological wind.
Original implementation: Claude; reviewed mathematical/wiring corrections: Codex.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    'MotionCoreNowcaster', 'pixel_center_grid', 'roi_sampling_grid',
    'safe_grid_sample', 'compose_backward_step', 'advect_sample',
    'inverse_softplus', 'gaussian_blur', 'core_target',
]


# ------------------------------------------------------------------------
# Physical-coordinate helpers (exported, unit-tested by the trainer/gates)
# ------------------------------------------------------------------------

def pixel_center_grid(h, w, device, dtype=torch.float32):
    """Normalised (x, y) pixel-centre grid, align_corners=False convention.

    Returns [H, W, 2] with values in (-1, 1); no batch dimension. Pixel index
    i of an axis of length N maps to (i + 0.5) / N * 2 - 1, so this is valid
    as a grid_sample source/target grid for ANY field covering the same
    physical domain at ANY pixel resolution.
    """
    ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h * 2 - 1
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w * 2 - 1
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack((gx, gy), -1)


def roi_sampling_grid(roi_px, roi_width_km, domain_width_km, device, dtype=torch.float32):
    """Pixel centres of a `roi_px`-wide square ROI, expressed in a larger
    concentric domain's own [-1, 1] normalisation.

    Both squares are assumed centred on the same physical point (true for
    the 50 km fine ROI inside the 400 km wide/context/core domain in this
    model). The result can be used to grid_sample ANY field that covers the
    larger domain, at any resolution, and yields values at the ROI's pixel
    centres.
    """
    local = (torch.arange(roi_px, device=device, dtype=dtype) + 0.5) / roi_px * roi_width_km \
        - roi_width_km / 2.0
    norm = local * (2.0 / domain_width_km)
    gy, gx = torch.meshgrid(norm, norm, indexing='ij')
    return torch.stack((gx, gy), -1)


def safe_grid_sample(field, grid, padding_mode='zeros'):
    if grid.ndim == 3:
        grid = grid.unsqueeze(0).expand(field.shape[0], -1, -1, -1)
    with torch.autocast(device_type=field.device.type, enabled=False):
        out = F.grid_sample(field.float(), grid.float(), mode='bilinear',
                            padding_mode=padding_mode, align_corners=False)
    return out.to(field.dtype)


def compose_backward_step(disp_km, velocity_km_per_step, field_width_km,
                          n_substeps=3, support=None, return_support=False):
    """Trace the NEW interval, then compose with the previous backward map.

    Border displacement extends coordinates, never clamps the source map.
    Returned support separately marks trajectories leaving known map support.
    """
    with torch.autocast(device_type=disp_km.device.type, enabled=False):
        old = disp_km.float()
        b, _, h, w = old.shape
        grid = _centres(h, w, old.device)[None].expand(b, -1, -1, -1)
        vel = velocity_km_per_step.float()
        scale = 2. / float(field_width_km)
        steps = max(1, int(n_substeps))
        dt = 1. / steps
        delta = torch.zeros_like(old)
        valid_vel = torch.ones_like(vel[:, :1])
        path_ok = torch.ones_like(old[:, :1], dtype=torch.bool)
        for _ in range(steps):
            start = grid - delta.permute(0, 2, 3, 1) * scale
            v0 = safe_grid_sample(vel, start, 'border')
            mid = start - .5 * dt * v0.permute(0, 2, 3, 1) * scale
            vmid = safe_grid_sample(vel, mid, 'border')
            delta = delta + dt * vmid
            path_ok &= _supported(valid_vel, start) & _supported(valid_vel, mid)
        previous_position = grid - delta.permute(0, 2, 3, 1) * scale
        result = delta + safe_grid_sample(old, previous_position, 'border')
        old_support = torch.ones_like(old[:, :1]) if support is None else support.float()
        if old_support.ndim == 3:
            old_support = old_support[:, None]
        path_ok &= _supported(old_support, previous_position)
        original_position = grid - result.permute(0, 2, 3, 1) * scale
        path_ok &= _supported(torch.ones_like(old[:, :1]), original_position)
    return (result, path_ok) if return_support else result


def advect_sample(source_field, disp_km_field, target_grid_norm, domain_width_km,
                   padding_mode='zeros'):
    with torch.autocast(device_type=source_field.device.type, enabled=False):
        grid = target_grid_norm.float()
        if grid.ndim == 3:
            grid = grid[None].expand(source_field.shape[0], -1, -1, -1)
        local_disp = safe_grid_sample(disp_km_field.float(), grid, 'border')
        coords = grid - local_disp.permute(0, 2, 3, 1) * (2. / float(domain_width_km))
        result = safe_grid_sample(source_field.float(), coords, padding_mode)
    return result.to(source_field.dtype)


def inverse_softplus(y, beta=8.0, eps=1e-6):
    with torch.autocast(device_type=y.device.type, enabled=False):
        z = y.float().clamp_min(eps)
        return z + torch.log(-torch.expm1(-float(beta) * z)) / float(beta)


def advance_material_position(position, velocity, domain_width_km=400., n_substeps=3):
    """Forward world positions [B,H,W,2] under Eulerian velocity [B,2,H,W].

    Velocity uses border extrapolation; geographical features queried at the
    returned positions use zero outside the domain. No backward-map inversion.
    """
    position = position.float()
    step_norm = 2. / (float(domain_width_km) * n_substeps)
    for _ in range(n_substeps):
        v0 = safe_grid_sample(velocity.float(), position, 'border').permute(0, 2, 3, 1)
        vm = safe_grid_sample(velocity.float(), position + .5 * step_norm * v0,
                              'border').permute(0, 2, 3, 1)
        position = position + step_norm * vm
    return position


def evolve_ground_birth(state, step_disp, grid, growth, birth, death,
                        strength_delta, source_rate, rain_birth, core_scale=1.):
    """Transport prior generated fields one interval, then add CURRENT births.

    No initial-anchor support is applied: a new source can exist where tracing
    all the way to the observation would leave the domain. Zero padding only
    removes old generated fields that leave the domain in this one interval.
    """
    rain, heat, strength = [advect_sample(x, step_disp, grid, 400.) for x in state]
    old_prob = -torch.expm1(-heat)
    heat = (heat * torch.exp(growth - death) + birth).clamp_max(20.)
    probability = -torch.expm1(-heat)
    strength = strength * torch.exp(strength_delta) + .1 * birth
    rain = (rain * torch.exp(source_rate) + rain_birth
            + .05 * core_scale * (probability - old_prob) * strength).clamp_min(0.)
    return rain, heat, strength


def gaussian_blur(field, sigma_px):
    """Separable Gaussian blur, replicate-padded. field: [N, 1, H, W]."""
    if sigma_px <= 0:
        return field
    device, dtype = field.device, field.dtype
    radius = max(int(math.ceil(3.0 * sigma_px)), 1)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / max(sigma_px, 1e-6)) ** 2)
    k = (k / k.sum()).view(1, 1, 1, -1)
    padded = F.pad(field, (radius, radius, 0, 0), mode='replicate')
    out = F.conv2d(padded, k)
    padded = F.pad(out, (0, 0, radius, radius), mode='replicate')
    return F.conv2d(padded, k.transpose(-1, -2))


def core_target(rain, res_km=2.0, threshold=.2, sigma_km=3.0,
                sharpness=6.0, peak_radius_km=6.0, valid=None):
    """Smoothed local prominence; a uniform plateau has exactly zero score.

    Require the full blur-plus-neighbourhood footprint (8 cells at 2 km).
    This is a soft target definition, not storm tracking or segmentation.
    """
    with torch.autocast(device_type=rain.device.type, enabled=False):
        r = rain.float()
        mask = torch.isfinite(r)
        if valid is not None:
            mask = mask & (valid >= .999)
        x = torch.where(mask, r.clamp_min(0), 0.)
        sigma = max(float(sigma_km) / float(res_km), .5)
        radius = max(1, int(math.ceil(3 * sigma)))
        axis = torch.arange(-radius, radius + 1, device=r.device, dtype=torch.float32)
        kernel = torch.exp(-.5 * (axis / sigma).square())
        kernel = kernel / kernel.sum()
        smooth = F.conv2d(x, kernel.view(1, 1, 1, -1), padding=(0, radius))
        smooth = F.conv2d(smooth, kernel.view(1, 1, -1, 1), padding=(radius, 0))
        peak_r = max(1, int(math.ceil(float(peak_radius_km) / float(res_km))))
        win = 2 * peak_r + 1
        floor = -F.max_pool2d(-smooth, win, stride=1, padding=peak_r)
        prominence = (smooth - floor).clamp_min(0)
        score = torch.sigmoid(float(sharpness) * (smooth - threshold))
        score = score * (-torch.expm1(-float(sharpness) * prominence))
        footprint = radius + peak_r
        full = F.avg_pool2d(mask.float(), 2 * footprint + 1, stride=1,
                            padding=footprint, count_include_pad=True) >= 1. - 1e-6
        return torch.where(full, score, 0.).clamp(0, 1)


# ------------------------------------------------------------------------
# Small building blocks
# ------------------------------------------------------------------------

class PairMotionEstimator(nn.Module):
    """Shared conv net estimating forward+backward flow for one adjacent pair.

    The SAME weights are applied to every adjacent pair in the observed
    history (this sharing is the "shared wide multiscale learned
    motion" requirement). Dilated convs give a growing receptive field
    (multiscale) without attention. Operates on the 50x50/8 km working grid.
    Zero-initialised final layer: at init every pair estimate is exactly
    zero flow, so training starts from "no motion cue" rather than noise.
    """

    def __init__(self, hidden=32, max_speed_km10=18.0):
        super().__init__()
        self.max_speed = float(max_speed_km10)
        self.net = nn.Sequential(
            nn.Conv2d(4, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, dilation=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=4, dilation=4), nn.GELU(),
            nn.Conv2d(hidden, 4, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, rain_a, rain_b, valid_a, valid_b):
        x = torch.cat((rain_a, rain_b, valid_a, valid_b), 1)
        out = self.net(x).float().tanh() * self.max_speed
        return out[:, :2], out[:, 2:]  # forward (a->b), backward (b->a), km/10min


class ConvGRUCell(nn.Module):
    """Convolutional GRU updating features on the anchor reference grid.

    In this model the recurrent states and their source terms are material
    quantities indexed by the last observed frame. The stored states are
    updated in place; their readouts are sampled through composed backward
    maps to the future physical grid (see MotionCoreNowcaster.forward).
    """

    def __init__(self, in_ch, hidden_ch, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.hidden_ch = hidden_ch
        self.gates = nn.Conv2d(in_ch + hidden_ch, 2 * hidden_ch, kernel, padding=pad)
        self.cand = nn.Conv2d(in_ch + hidden_ch, hidden_ch, kernel, padding=pad)
        nn.init.zeros_(self.cand.weight)
        nn.init.zeros_(self.cand.bias)

    def forward(self, x, h):
        inp = torch.cat((x, h), 1)
        z, r = torch.sigmoid(self.gates(inp)).chunk(2, 1)
        c = torch.tanh(self.cand(torch.cat((x, r * h), 1)))
        return (1 - z) * h + z * c


def conv_head(in_ch, out_ch, hidden=32, zero_init=True):
    net = nn.Sequential(
        nn.Conv2d(in_ch, hidden, 3, padding=1), nn.GELU(),
        nn.Conv2d(hidden, out_ch, 1),
    )
    if zero_init:
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
    return net


# ------------------------------------------------------------------------
# Main model
# ------------------------------------------------------------------------

class MotionCoreNowcaster(nn.Module):
    """Shared wide motion, explicit rain-core state, native-detail fine readout.

    forward(fine, context, valid, clock):
      fine    [B, H, 1, 100, 100]  raw mm/10min, 500 m/px, 50 km domain
      context [B, H, 1, 200, 200]  raw mm/10min, 2 km/px, 400 km domain
      valid   [B, H, 1, 200, 200]  per-frame coverage 0..1 (context grid)
      clock   [B, 19, 4]           known absolute-time features, index 0 is
                                    the anchor (last observed) time, 1..18
                                    are the future lead times; no future rain

    H is history_steps (default 6 for original checkpoint compatibility).
    See MODEL_NOTES.md for the full walk-through, target definitions, aux
    loss normalisation, caps and known limitations.
    """

    def __init__(self, forecast_steps=18, quantiles=(0.5, 0.6, 0.75, 0.9, 0.95),
                 hidden=48, fine_hidden=32, core_hidden=16, wide_grid=50,
                 estimator_hidden=32, cond_dim=16, proj_dim=8,
                 max_prior_speed_km10=18.0, max_lead_velocity_delta_km10=6.0,
                 max_fine_correction_km10=2.0, n_substeps=3,
                 core_threshold=0.2, output_sharpness=8.0,
                 core_growth_cap=0.5, core_birth_cap=1.0, core_death_cap=1.0,
                 clock_dim=4, history_steps=6, region_channels=8,
                 fine_birth_enabled=True, fine_correction_cap=20.0,
                 fine_readout_floor=0.05):
        super().__init__()
        if isinstance(history_steps, bool) or int(history_steps) != history_steps or history_steps < 2:
            raise ValueError('history_steps must be an integer >= 2')
        self.history_steps = int(history_steps)
        if tuple(quantiles) != tuple(sorted(quantiles)):
            raise ValueError('quantiles must be sorted ascending')
        self.forecast_steps = int(forecast_steps)
        self.quantiles = tuple(float(q) for q in quantiles)
        self.n_quantiles = len(self.quantiles)
        self.fine_birth_enabled = bool(fine_birth_enabled)
        # This flag controls the occurrence auxiliary only; native additive
        # rainfall is deliberately absent in the coherent V45 readout.
        self.fine_birth_additive_enabled = False
        # Maximum signed correction in inverse-softplus linear-mm coordinates.
        # This is a numerical bound, not a calibrated physical growth limit.
        # With floor=.05/beta=8 a dry q0.6 can approach 19.86 mm/10min.
        self.fine_correction_cap = float(fine_correction_cap)
        self.fine_readout_floor = float(fine_readout_floor)
        if self.fine_correction_cap <= 0 or self.fine_readout_floor <= 0:
            raise ValueError('Fine correction cap and readout floor must be positive')
        self.hidden = int(hidden)
        self.fine_hidden = int(fine_hidden)
        self.core_hidden = int(core_hidden)
        self.wide_grid = int(wide_grid)
        self.n_substeps = int(n_substeps)
        self.output_sharpness = float(output_sharpness)
        self.core_threshold = float(core_threshold)
        self.core_growth_cap = float(core_growth_cap)
        self.core_birth_cap = float(core_birth_cap)
        self.core_death_cap = float(core_death_cap)

        # ---- fixed geometry (see module docstring) --------------------
        self.wide_domain_km = 400.0
        self.fine_domain_km = 50.0
        self.context_size = 200
        self.fine_size = 100
        self.roi_fraction = self.fine_domain_km / self.wide_domain_km

        # ---- shared motion estimator (observed pairs only) ------------
        self.motion_estimator = PairMotionEstimator(estimator_hidden, max_prior_speed_km10)
        self.max_prior_speed_km10 = float(max_prior_speed_km10)
        self.max_lead_velocity_delta_km10 = float(max_lead_velocity_delta_km10)
        self.max_fine_correction_km10 = float(max_fine_correction_km10)

        # ---- wide encoder: history frames (log1p rain + valid) -> 50x50 hidden
        self.wide_encoder = nn.Sequential(
            nn.Conv2d(2 * self.history_steps, hidden, 4, padding=1, stride=2), nn.GELU(),   # 200 -> 100
            nn.Conv2d(hidden, hidden, 4, padding=1, stride=2), nn.GELU(),  # 100 -> 50
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )

        # ---- conditioning: per-lead index + known clock ----------------
        self.cond_dim = int(cond_dim)
        self.lead_embed = nn.Parameter(torch.zeros(self.forecast_steps, self.cond_dim))
        self.clock_proj = nn.Linear(clock_dim, self.cond_dim)
        nn.init.zeros_(self.clock_proj.weight)
        nn.init.zeros_(self.clock_proj.bias)

        # ---- wide recurrent state (material, reference 50x50 grid) --------
        self.proj_dim = int(proj_dim)
        self.wide_from_core = nn.Conv2d(core_hidden, self.proj_dim, 1)
        self.wide_cell = ConvGRUCell(self.cond_dim + self.proj_dim, hidden)
        self.wide_velocity_head = conv_head(hidden, 2, hidden)
        self.wide_source_sink_head = conv_head(hidden, 1, hidden)
        self.wide_birth_head = conv_head(hidden, 1, hidden)
        nn.init.constant_(self.wide_birth_head[-1].bias, -2.0)
        self.wide_birth_probability_head = conv_head(hidden, 1, hidden)
        nn.init.constant_(self.wide_birth_probability_head[-1].bias, -4.0)

        # ---- core recurrent state (material, reference 50x50 grid) --------
        self.core_init = nn.Sequential(
            nn.Conv2d(2 + hidden, core_hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(core_hidden, core_hidden, 3, padding=1),
        )
        self.core_from_wide = nn.Conv2d(hidden, self.proj_dim, 1)
        self.core_cell = ConvGRUCell(self.cond_dim + self.proj_dim, core_hidden)
        self.core_growth_head = conv_head(core_hidden, 1, core_hidden)
        self.core_birth_head = conv_head(core_hidden, 1, core_hidden)
        self.core_death_head = conv_head(core_hidden, 1, core_hidden)
        self.core_strength_head = conv_head(core_hidden, 1, core_hidden)

        # ---- native fine detail (aligned material skip, advected at readout) -----------
        self.fine_skip_encoder = nn.Sequential(
            nn.Conv2d(self.history_steps, fine_hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(fine_hidden, fine_hidden, 3, padding=1), nn.GELU(),
        )
        fine_corr_in = hidden + fine_hidden + self.cond_dim
        self.fine_velocity_correction = conv_head(fine_corr_in, 2, fine_hidden)

        decoder_in = 1 + hidden + 1 + 1 + fine_hidden + self.cond_dim
        self.fine_decoder = nn.Sequential(
            nn.Conv2d(decoder_in, fine_hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(fine_hidden, fine_hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(fine_hidden, self.n_quantiles + 1, 1),
        )
        nn.init.normal_(self.fine_decoder[-1].weight, std=1e-3)
        nn.init.zeros_(self.fine_decoder[-1].bias)
        with torch.no_grad():
            self.fine_decoder[-1].weight[0].zero_()
            self.fine_decoder[-1].bias[1:self.n_quantiles].fill_(-6.0)
            self.fine_decoder[-1].bias[self.n_quantiles] = -4.0
        nn.init.constant_(self.core_birth_head[-1].bias, -6.0)
        nn.init.constant_(self.core_death_head[-1].bias, -4.0)

        # Learned geographical effects, not measured elevation/terrain. Never
        # transport this parameter: every query uses CURRENT world coordinates.
        self.region_embedding = nn.Parameter(torch.randn(1, region_channels, 50, 50) * .02)
        self.region_head = conv_head(region_channels + hidden + self.cond_dim, 8, 24)
        self.region_fine_head = conv_head(region_channels + fine_corr_in, 2, 24)
        for head in (self.region_head, self.region_fine_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

        self.register_buffer('_wide_pixel_grid', pixel_center_grid(wide_grid, wide_grid, 'cpu'),
                             persistent=False)
        self.register_buffer('_ctx_pixel_grid', pixel_center_grid(self.context_size, self.context_size, 'cpu'),
                             persistent=False)
        self.register_buffer('_fine_pixel_grid', pixel_center_grid(self.fine_size, self.fine_size, 'cpu'),
                             persistent=False)
        self.register_buffer(
            '_roi_in_wide',
            roi_sampling_grid(self.fine_size, self.fine_domain_km, self.wide_domain_km, 'cpu'),
            persistent=False)

    # -- helpers reaching buffers on the right device --------------------
    def _grid(self, name, device, dtype):
        return getattr(self, name).to(device=device, dtype=dtype)

    def _pool_to_wide(self, x):
        """[B, ..., 200, 200] -> [B, ..., 50, 50] by average pooling (context -> working grid)."""
        shape = x.shape[:-2]
        k = self.context_size // self.wide_grid
        return F.avg_pool2d(x.reshape(-1, 1, self.context_size, self.context_size), k).reshape(
            *shape, self.wide_grid, self.wide_grid)

    # ---------------------------------------------------------------
    def _observe(self, fine, context, valid):
        """Use observed history only; return masks, aligned history and motion.

        The prior is the componentwise median of H-1 forward pair estimates,
        sampled into anchor coordinates through the observed backward maps. It
        approximates an anchor-time Eulerian velocity; acceleration is not observed.
        Native fine validity is conservatively inherited from its containing 2km
        context cells, not a recovered native observation mask.
        """
        device, batch = fine.device, fine.shape[0]
        history = self.history_steps
        context_in, fine_in = context[:, :, 0].float(), fine[:, :, 0].float()
        valid_raw = ((valid[:, :, 0] >= .999) & torch.isfinite(context_in)
                     & (context_in >= 0)).float()
        context_raw = torch.where(valid_raw.bool(), context_in, 0.)
        native_indices = (350 + torch.arange(100, device=device)) // 4
        fine_valid = valid_raw[:, :, native_indices[:, None], native_indices[None, :]]
        fine_valid = (fine_valid.bool() & torch.isfinite(fine_in) & (fine_in >= 0)).float()
        fine_raw = torch.where(fine_valid.bool(), fine_in, 0.)
        rain_pool = self._pool_to_wide(context_raw.log1p())
        valid_pool = (self._pool_to_wide(valid_raw) >= .999).float()
        rain_pool = rain_pool * valid_pool
        pair_grid = self._grid('_wide_pixel_grid', device, torch.float32)[None].expand(batch, -1, -1, -1)
        wide_grid = self._grid('_ctx_pixel_grid', device, torch.float32)[None].expand(batch, -1, -1, -1)
        roi_grid = self._grid('_roi_in_wide', device, torch.float32)[None].expand(batch, -1, -1, -1)
        scale = 2. / self.wide_domain_km

        def sample(field, coordinates):
            with torch.autocast(device_type=device.type, enabled=False):
                return F.grid_sample(field.float(), coordinates.float(), mode='bilinear',
                                     padding_mode='zeros', align_corners=False)

        def evidence(rain, coverage):
            mean = F.avg_pool2d(rain, 3, 1, 1)
            variance = (F.avg_pool2d(rain.square(), 3, 1, 1) - mean.square()).clamp_min(0)
            texture = variance.sqrt()
            neighborhood = (F.avg_pool2d(coverage, 3, 1, 1) >= .999).float()
            return (coverage * neighborhood * rain / (rain + .05)
                    * texture / (texture + .02)).detach()

        forward_pairs, backward_pairs, photo, cycle_losses, smooth_losses = [], [], [], [], []
        source_weights, destination_supports, anchor_cycle, anchor_pair_support = [], [], None, None
        for k in range(history - 1):
            ra, rb = rain_pool[:, k:k + 1], rain_pool[:, k + 1:k + 2]
            va, vb = valid_pool[:, k:k + 1], valid_pool[:, k + 1:k + 2]
            fwd, bwd = self.motion_estimator(ra, rb, va, vb)
            forward_pairs.append(fwd.float())
            backward_pairs.append(bwd.float())
            missing_cost = (ra.amax((2, 3), keepdim=True) + rb.amax((2, 3), keepdim=True) + .1).detach()
            for source, target, source_valid, target_valid, flow, reverse in (
                    (ra, rb, va, vb, fwd, bwd), (rb, ra, vb, va, bwd, fwd)):
                coordinates = pair_grid + flow.float().permute(0, 2, 3, 1) * scale
                destination_valid = sample(target_valid, coordinates).clamp(0, 1)
                weight = evidence(source, source_valid)
                denominator = weight.sum().clamp_min(1e-6)
                error = ((sample(target, coordinates) - source).square() + 1e-6).sqrt() - .001
                # Fixed source denominator; leaving observed support cannot hide error.
                photo.append((weight * (error * destination_valid.detach()
                             + missing_cost * (1 - destination_valid))).sum() / denominator)
                cycle = flow.float() + sample(reverse, coordinates)
                cycle_losses.append((weight * (cycle.square().mean(1, keepdim=True)
                                    * destination_valid.detach())).sum() / denominator)
                source_weights.append(weight.sum().detach())
                destination_supports.append((weight * destination_valid.detach()).sum())
                dx, dy = flow[..., :, 1:] - flow[..., :, :-1], flow[..., 1:, :] - flow[..., :-1, :]
                mx = source_valid[..., :, 1:] * source_valid[..., :, :-1]
                my = source_valid[..., 1:, :] * source_valid[..., :-1, :]
                wx = mx * torch.exp(-(source[..., :, 1:] - source[..., :, :-1]).abs())
                wy = my * torch.exp(-(source[..., 1:, :] - source[..., :-1, :]).abs())
                smooth_losses.append((wx * dx.abs()).sum() / (2 * wx.sum()).clamp_min(1)
                                     + (wy * dy.abs()).sum() / (2 * wy.sum()).clamp_min(1))
                if k == history - 2 and source is rb:
                    anchor_cycle = cycle.detach().norm(dim=1, keepdim=True)
                    anchor_pair_support = (source_valid * destination_valid).detach()

        # Trace anchor positions back through each observed pair, without an inverse
        # sign shortcut. Always sample the original history image only once.
        def trace_history(anchor_grid, is_fine=False):
            grids = [None] * history
            grids[-1] = anchor_grid
            support = [None] * history
            support[-1] = sample(valid_raw[:, -1:], anchor_grid)
            fields, masks = [None] * history, [None] * history
            for k in range(history - 2, -1, -1):
                backward = sample(backward_pairs[k], grids[k + 1])
                grids[k] = grids[k + 1] + backward.permute(0, 2, 3, 1) * scale
                support[k] = support[k + 1] * sample(valid_raw[:, k:k + 1], grids[k])
            for k in range(history):
                coordinates = grids[k] * (self.wide_domain_km / self.fine_domain_km) if is_fine else grids[k]
                original = fine_raw[:, k:k + 1] if is_fine else context_raw[:, k:k + 1]
                original_valid = fine_valid[:, k:k + 1] if is_fine else valid_raw[:, k:k + 1]
                usable = (support[k] >= .999).float() * (sample(original_valid, coordinates) >= .999).float()
                fields[k], masks[k] = sample(original, coordinates) * usable, usable
            return torch.cat(fields, 1), torch.cat(masks, 1), grids

        aligned_context, aligned_valid, _ = trace_history(wide_grid)
        aligned_fine, aligned_fine_valid, _ = trace_history(roi_grid, is_fine=True)
        # Project every earlier-grid forward estimate onto the anchor 50x50 grid.
        flow_coordinates = [None] * history
        flow_coordinates[-1] = pair_grid
        for k in range(history - 2, -1, -1):
            flow_coordinates[k] = flow_coordinates[k + 1] + sample(
                backward_pairs[k], flow_coordinates[k + 1]).permute(0, 2, 3, 1) * scale
        projected = [sample(forward_pairs[k], flow_coordinates[k]) for k in range(history - 1)]
        base_prior = torch.stack(projected, 0).median(0).values
        with torch.no_grad():
            anchor_evidence = evidence(context_raw[:, -1:].log1p(), valid_raw[:, -1:])
            quality = F.interpolate(torch.exp(-anchor_cycle / 2.) * anchor_pair_support, size=(200, 200),
                                    mode='bilinear', align_corners=False)
            confidence = anchor_evidence * quality
        fixed_support = torch.stack(source_weights).sum()
        actual_support = torch.stack(destination_supports).sum()
        return dict(
            fine_raw=fine_raw, context_raw=context_raw, valid_raw=valid_raw,
            fine_valid=fine_valid, lastfinevalid=fine_valid[:, -1:],
            aligned_context=aligned_context, aligned_valid=aligned_valid,
            aligned_fine=aligned_fine, aligned_fine_valid=aligned_fine_valid,
            base_prior=base_prior, confidence=confidence,
            forward_pairs=torch.stack(forward_pairs, 1), backward_pairs=torch.stack(backward_pairs, 1),
            aux_losses=dict(motion_photo=torch.stack(photo).mean(),
                            motion_fb=torch.stack(cycle_losses).mean(),
                            motion_smooth=torch.stack(smooth_losses).mean()),
            diagnostics=dict(motion_fixed_support=fixed_support.detach(),
                             motion_destination_support=actual_support.detach(),
                             motion_support_ratio=(actual_support / fixed_support.clamp_min(1e-6)).detach(),
                             aligned_context_support=aligned_valid.mean().detach(),
                             aligned_fine_support=aligned_fine_valid.mean().detach(),
                             observed_context_support=valid_raw.mean().detach()))

    def forward(self, fine, context, valid, clock, motion_override=None, core_scale=1.0,
                region_scale=1.0):
        if fine.shape[1:] != (self.history_steps, 1, 100, 100) or context.shape[1:] != (self.history_steps, 1, 200, 200):
            raise ValueError(f'Expected {self.history_steps} native fine and wide observations')
        if (fine.shape[0] != context.shape[0] or valid.shape != context.shape
                or clock.shape != (fine.shape[0], self.forecast_steps + 1, 4)):
            raise ValueError('Coverage/clock contract mismatch')
        obs = self._observe(fine, context, valid)
        b, device = fine.shape[0], fine.device
        raw = obs['context_raw']
        fine_raw = obs['fine_raw']
        coverage = obs['valid_raw']
        prior = obs['base_prior']
        wide_h = self.wide_encoder(torch.cat((obs['aligned_context'].log1p(), obs['aligned_valid']), 1))
        # Reuse encoder weights for fixed-ground observed history. This fallback
        # gives birth heads rainfall context where the anchor trace is unknown.
        ground_history_h = self.wide_encoder(torch.cat((raw.log1p(), coverage), 1))
        fine_skip = self.fine_skip_encoder(obs['aligned_fine'].log1p())
        grid50 = self._grid('_wide_pixel_grid', device, torch.float32)
        grid200 = self._grid('_ctx_pixel_grid', device, torch.float32)
        roi = self._grid('_roi_in_wide', device, torch.float32)
        roi_b = roi[None].expand(b, -1, -1, -1)
        def up(x):
            return F.interpolate(x.float(), (200, 200), mode='bilinear', align_corners=False)
        def project(x, disp, target, support):
            value = advect_sample(x, disp, target, 400.)
            known = safe_grid_sample(support.float(), target, 'zeros')
            return value * (known >= 1. - 1e-5).to(value.dtype)
        last = raw[:, -1:]
        heat_prob = core_target(last, threshold=self.core_threshold, valid=coverage[:, -1:])
        # High-resolution material fields retain observed 2 km rain/core structure.
        heat = -torch.log1p(-heat_prob.clamp_max(.995))
        strength = last.clone()
        core_h = self.core_init(torch.cat((self._pool_to_wide(heat_prob),
                                           self._pool_to_wide(strength * heat_prob), wide_h), 1))
        ground_core_history = self.core_init(torch.cat((self._pool_to_wide(heat_prob),
                    self._pool_to_wide(strength * heat_prob), ground_history_h), 1))
        rain_material = last.clone()
        ground_birth_state = tuple(torch.zeros_like(last) for _ in range(3))
        fine_birth_state = tuple(torch.zeros_like(last) for _ in range(3))
        native_valid = obs['lastfinevalid']
        native_residual = (fine_raw[:, -1:] - safe_grid_sample(last, roi)) * native_valid
        observed_amplitude = torch.ones_like(last)
        disp = torch.zeros(b, 2, 200, 200, device=device)
        fine_disp = torch.zeros_like(disp)
        support = torch.ones(b, 1, 200, 200, device=device)
        fine_support = torch.ones_like(support)
        # Forward trajectories of anchor material centres. This is independent
        # of the Eulerian backward source map `disp`; the maps are not synonyms.
        material_position = grid50[None].expand(b, -1, -1, -1).clone()
        region = self.region_embedding.expand(b, -1, -1, -1)
        region_roi = safe_grid_sample(region, roi)
        out = {k: [] for k in ('fine', 'wide', 'core', 'core_strength', 'core_area',
                               'velocity', 'birth', 'death', 'source_sink', 'fine_velocity',
                               'rain_birth', 'generated_rain', 'fine_generated_rain',
                               'rain_birth_logit', 'rain_birth_probability', 'rain_birth_amount',
                               'fine_birth_logit', 'fine_birth_probability', 'fine_birth_amount',
                               'fine_new_rain',
                               'observed_source_support', 'fine_observed_source_support')}
        temporal = []
        previous_velocity = prior
        for j in range(self.forecast_steps):
            cond = self.lead_embed[j][None].expand(b, -1) + self.clock_proj(clock[:, j + 1])
            cw = cond[:, :, None, None].expand(-1, -1, 50, 50)
            cf = cond[:, :, None, None].expand(-1, -1, 100, 100)
            # Recurrent features and sources are material (anchor-frame) quantities.
            wide_h = self.wide_cell(torch.cat((cw, self.wide_from_core(core_h) * core_scale), 1), wide_h)
            core_h = self.core_cell(torch.cat((cw, self.core_from_wide(wide_h)), 1), core_h)
            ground_before = project(wide_h, disp, grid50, support)
            before_unknown = 1. - (safe_grid_sample(support.float(), grid50) >= 1. - 1e-5).float()
            ground_before = ground_before + before_unknown * ground_history_h
            region_before = torch.tanh(self.region_head(torch.cat((region, ground_before, cw), 1)).float()) * region_scale
            delta_material = torch.tanh(self.wide_velocity_head(wide_h).float()) * self.max_lead_velocity_delta_km10
            delta = project(delta_material, disp, grid50, support)
            velocity = prior + delta + 2. * region_before[:, :2]
            if motion_override is not None:
                if motion_override.shape != velocity.shape:
                    raise ValueError('motion_override must be B,2,50,50 km/10min')
                velocity = motion_override.float()
            temporal.append((velocity - previous_velocity).square().mean() + .1 * delta.square().mean())
            previous_velocity = velocity
            wide_features_old = project(wide_h, fine_disp, roi, fine_support)
            old_offset = safe_grid_sample(fine_disp, roi, 'border')
            native_source_old = (roi_b - old_offset.permute(0, 2, 3, 1) / 200.) * 8.
            native_features_old = safe_grid_sample(fine_skip, native_source_old)
            correction = torch.tanh(self.fine_velocity_correction(
                torch.cat((wide_features_old, native_features_old, cf), 1)).float()) * self.max_fine_correction_km10
            correction = correction + torch.tanh(self.region_fine_head(torch.cat(
                (region_roi, wide_features_old, native_features_old, cf), 1)).float()) * region_scale
            # Embed the bounded native correction into the entire 400 km velocity map.
            correction_wide = safe_grid_sample(correction, grid200 * 8., 'zeros')
            velocity_wide = up(velocity)
            fine_velocity_wide = velocity_wide + correction_wide
            # Integrate x(anchor,t)->x(anchor,t+1), sampling velocity at its
            # CURRENT world position with midpoint substeps (normalised km).
            material_position = advance_material_position(material_position, velocity_wide,
                                                            400., self.n_substeps)
            # New sources retain their own one-interval transport, independently
            # of the accumulated observation-to-future source map/support.
            step_disp = compose_backward_step(torch.zeros_like(disp), velocity_wide,
                                              400., self.n_substeps)
            fine_step_disp = compose_backward_step(torch.zeros_like(disp), fine_velocity_wide,
                                                   400., self.n_substeps)
            disp, support = compose_backward_step(disp, velocity_wide, 400., self.n_substeps,
                                                  support=support, return_support=True)
            fine_disp, fine_support = compose_backward_step(fine_disp, fine_velocity_wide, 400., self.n_substeps,
                                                            support=fine_support, return_support=True)
            ground_core_h = project(core_h, disp, grid50, support)
            ground_wide_h = project(wide_h, disp, grid50, support)
            unknown_ground = 1. - (safe_grid_sample(support.float(), grid50)
                                  >= 1. - 1e-5).float()
            ground_core_h = ground_core_h + unknown_ground * ground_core_history
            ground_wide_h = ground_wide_h + unknown_ground * ground_history_h
            region_current = torch.tanh(self.region_head(torch.cat((region, ground_wide_h, cw), 1)).float()) * region_scale
            material_region = safe_grid_sample(region_current, material_position, 'zeros')
            growth = up(torch.tanh(self.core_growth_head(core_h).float() + material_region[:, 2:3])) * self.core_growth_cap
            birth = up(F.softplus(self.core_birth_head(ground_core_h).float() + 2. * region_current[:, 6:7])) * self.core_birth_cap
            death = up(F.softplus(self.core_death_head(core_h).float() + material_region[:, 3:4])) * self.core_death_cap
            old_prob = -torch.expm1(-heat)
            heat = heat * torch.exp(growth - death)
            heat = heat.clamp_max(20.)
            probability_material = -torch.expm1(-heat)
            strength_delta = up(torch.tanh(self.core_strength_head(core_h).float() + material_region[:, 4:5])) * .1
            strength = strength * torch.exp(strength_delta)
            source_rate = up(torch.tanh(self.wide_source_sink_head(wide_h).float() + material_region[:, 5:6])) * .2
            # Both coarse observed rainfall and its native detail share the
            # same multiplicative intensity evolution in the anchor frame.
            # Core increments and generated rainfall do not scale old detail.
            observed_amplitude = observed_amplitude * torch.exp(source_rate)
            # Supervised occurrence and conditional amount have independent
            # gradients. Fixed-ground regional context modulates both heads.
            rain_birth_logit = up(self.wide_birth_probability_head(ground_wide_h).float()
                                  + 2. * region_current[:, 7:8])
            rain_birth_probability = torch.sigmoid(rain_birth_logit)
            rain_birth_amount = F.softplus(up(self.wide_birth_head(ground_wide_h).float()
                                              + .5 * region_current[:, 7:8]))
            rain_birth = rain_birth_probability * rain_birth_amount
            core_change = .05 * core_scale * (probability_material - old_prob) * strength
            rain_material = (rain_material * torch.exp(source_rate) + core_change).clamp_min(0.)
            def current_rates(displacement, known_support, ground_wide, ground_core):
                # Preserve material-derived rates on known trajectories, but
                # allow generated fields to evolve where the observation map
                # is unavailable. Those cells use fixed-ground history features.
                unknown = 1. - (safe_grid_sample(known_support.float(), grid200)
                               >= 1. - 1e-5).float()
                fallback = (up(torch.tanh(self.core_growth_head(ground_core).float())) * self.core_growth_cap,
                            up(F.softplus(self.core_death_head(ground_core).float())) * self.core_death_cap,
                            up(torch.tanh(self.core_strength_head(ground_core).float())) * .1,
                            up(torch.tanh(self.wide_source_sink_head(ground_wide).float())) * .2)
                # V42 base rates retain the original material/fallback logic.
                # Regional effects are evaluated at this ground cell, never
                # obtained by transporting fixed E along the backward map.
                base_material = (up(torch.tanh(self.core_growth_head(core_h).float())) * self.core_growth_cap,
                                 up(F.softplus(self.core_death_head(core_h).float())) * self.core_death_cap,
                                 up(torch.tanh(self.core_strength_head(core_h).float())) * .1,
                                 up(torch.tanh(self.wide_source_sink_head(wide_h).float())) * .2)
                base = [project(x, displacement, grid200, known_support) + unknown * y
                        for x, y in zip(base_material, fallback)]
                local = torch.tanh(self.region_head(torch.cat((region, ground_wide, cw), 1)).float()) * region_scale
                # Positive death multiplier; bounded additive growth/strength/source.
                return [base[0] + up(local[:, 2:3]) * .1,
                        base[1] * torch.exp(up(local[:, 3:4])),
                        base[2] + up(local[:, 4:5]) * .03,
                        base[3] + up(local[:, 5:6]) * .05]
            ground_rates = current_rates(disp, support, ground_wide_h, ground_core_h)
            fine_unknown = 1. - (safe_grid_sample(fine_support.float(), grid50)
                                >= 1. - 1e-5).float()
            fine_ground_wide = project(wide_h, fine_disp, grid50, fine_support) + fine_unknown * ground_history_h
            fine_ground_core = project(core_h, fine_disp, grid50, fine_support) + fine_unknown * ground_core_history
            fine_rates = current_rates(fine_disp, fine_support, fine_ground_wide, fine_ground_core)
            def evolve(state, step, rates):
                g, d, s, r = rates
                return evolve_ground_birth(state, step, grid200, g, birth, d, s, r,
                                           rain_birth, core_scale)
            ground_birth_state = evolve(ground_birth_state, step_disp, ground_rates)
            fine_birth_state = evolve(fine_birth_state, fine_step_disp, fine_rates)
            generated_rain, generated_heat, generated_strength = ground_birth_state
            wide = project(rain_material, disp, grid200, support) + generated_rain
            observed_prob = project(probability_material, disp, grid200, support)
            core = 1. - (1. - observed_prob) * torch.exp(-generated_heat)
            core_strength = project(strength, disp, grid200, support) + generated_strength
            # Read the SAME evolved wide rain through the fine physical source map.
            wide_anchor = (project(rain_material, fine_disp, roi, fine_support)
                           + safe_grid_sample(fine_birth_state[0], roi))
            local_offset = safe_grid_sample(fine_disp, roi, 'border')
            native_source = (roi_b - local_offset.permute(0, 2, 3, 1) / 200.) * 8.
            source_known = (safe_grid_sample(fine_support.float(), roi) >= 1. - 1e-5).float()
            evolved_native_residual = native_residual * safe_grid_sample(observed_amplitude, roi)
            detail = safe_grid_sample(evolved_native_residual, native_source) * source_known
            anchor = (wide_anchor + detail).clamp_min(0.)
            fine_features = safe_grid_sample(fine_skip, native_source) * source_known
            wide_features = project(wide_h, fine_disp, roi, fine_support)
            fine_observed_prob = project(probability_material, fine_disp, roi, fine_support)
            core_fine = (1. - (1. - fine_observed_prob)
                         * torch.exp(-safe_grid_sample(fine_birth_state[1], roi))) * core_scale
            strength_fine = (project(strength, fine_disp, roi, fine_support)
                             + safe_grid_sample(fine_birth_state[2], roi)) * core_fine
            decoded = self.fine_decoder(torch.cat((anchor.log1p(), wide_features, core_fine,
                                                   strength_fine, fine_features, cf), 1)).float()
            quantiles = self._fine_quantile_readout(anchor, decoded)
            # Current-lead occurrence auxiliary only. It does not create a
            # common rainfall floor or bypass the quantile readout.
            fine_birth_logit = decoded[:, self.n_quantiles:self.n_quantiles + 1]
            fine_birth_probability = torch.sigmoid(fine_birth_logit)
            fine_birth_amount = torch.zeros_like(fine_birth_probability)
            fine_new_rain = torch.zeros_like(fine_birth_probability)
            values = dict(fine=quantiles, wide=wide, core=core, core_strength=core_strength,
                          core_area=core.mean((1,2,3)), velocity=velocity, birth=birth, death=death,
                          source_sink=project(source_rate, disp, grid200, support),
                          rain_birth=rain_birth, generated_rain=generated_rain,
                          rain_birth_logit=rain_birth_logit,
                          rain_birth_probability=rain_birth_probability,
                          rain_birth_amount=rain_birth_amount,
                          fine_birth_logit=fine_birth_logit,
                          fine_birth_probability=fine_birth_probability,
                          fine_birth_amount=fine_birth_amount, fine_new_rain=fine_new_rain,
                          fine_generated_rain=safe_grid_sample(fine_birth_state[0], roi),
                          fine_velocity=safe_grid_sample(fine_velocity_wide, roi),
                          observed_source_support=project(coverage[:, -1:], disp, grid200, support),
                          fine_observed_source_support=project(coverage[:, -1:], fine_disp, roi, fine_support))
            for key, value in values.items():
                out[key].append(value)
        out = {k: torch.stack(v, 1) for k, v in out.items()}
        out['confidence'] = obs['confidence']
        out['aux_losses'] = dict(obs['aux_losses'], motion_temporal=torch.stack(temporal).mean())
        out['diagnostics'] = dict(base_prior_speed_kmh=prior.detach().norm(dim=1).mean()*6.,
                                  confidence_mean=obs['confidence'].detach().mean(),
                                  geometric_path_support_mean=fine_support.float().mean().detach(),
                                  observed_source_support_mean=out['fine_observed_source_support'].detach().mean())
        out['diagnostics'].update(obs['diagnostics'])
        out['wide_valid'] = coverage[:, -1:]
        return out

    def _fine_quantile_readout(self, anchor, decoded):
        """Ordered nonnegative quantiles with one bounded signed correction.

        The positive floor only stabilises the coordinate transform. It is
        subtracted analytically and is never an added rainfall amount. At
        zero correction q0.6 is exactly the anchor, including a dry anchor;
        its correction derivative remains meaningful there (~0.33 at the
        defaults), allowing generation without a separate additive branch.
        """
        center = inverse_softplus(anchor + self.fine_readout_floor, self.output_sharpness)
        # Unit slope near zero keeps the learning scale independent of cap.
        correction = self.fine_correction_cap * torch.tanh(decoded[:, :1] / self.fine_correction_cap)
        gaps = F.softplus(decoded[:, 1:self.n_quantiles])
        cumulative = torch.cumsum(torch.cat((torch.zeros_like(center), gaps), 1), 1)
        offsets = cumulative - cumulative[:, 1:2]
        base = F.softplus(center, beta=self.output_sharpness)
        delta = F.softplus(center + correction + offsets, beta=self.output_sharpness) - base
        return (anchor + delta).clamp_min(0.)


def _centres(h, w, device):
    y = (torch.arange(h, device=device, dtype=torch.float32) + .5) * (2. / h) - 1
    x = (torch.arange(w, device=device, dtype=torch.float32) + .5) * (2. / w) - 1
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    return torch.stack((xx, yy), -1)

def _supported(support, grid):
    return safe_grid_sample(support.float(), grid, 'zeros') >= 1. - 1e-5
