"""Focused V45 semantic-supervision regression tests (CPU, no trainer required)."""
import sys
import unittest
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / 'ClimaX' / 'src'))
from birth_objective import (current_occurrence_targets, balanced_location_loss,
                             negative_final_amount_loss, transport_targets)


class CoherentFineLossTests(unittest.TestCase):
    def test_persistent_rain_is_positive_at_every_lead(self):
        truth = torch.zeros(1, 3, 1, 20, 20)
        truth[:, :, :, 10, 10] = 1.
        valid = torch.ones_like(truth)
        new = current_occurrence_targets(truth, valid)
        self.assertEqual(new['target_bool'][:, :, :, 10, 10].sum().item(), 3)
        old = transport_targets(torch.zeros_like(truth[:, 0]), truth,
                                valid[:, 0], valid, torch.zeros(1, 3, 2, 20, 20),
                                50., 1.5, 3.)
        self.assertEqual(old['target_bool'][:, :, :, 10, 10].sum().item(), 1)
        logits = torch.zeros_like(truth, requires_grad=True)
        loss, _ = balanced_location_loss(logits, new)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue((logits.grad[:, :, :, 10, 10] < 0).all())

    def test_invalid_nonfinite_and_borders(self):
        truth = torch.tensor([.2, .2, float('nan'), -.1]).reshape(1, 1, 1, 1, 4)
        valid = torch.tensor([1., .5, 1., 1.]).reshape_as(truth)
        target = current_occurrence_targets(truth, valid)
        self.assertEqual(target['mask'].sum().item(), 1)
        self.assertTrue(target['target_bool'][0, 0, 0, 0, 0])
        logits = torch.zeros_like(truth, requires_grad=True)
        loss, _ = balanced_location_loss(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.equal(logits.grad[..., 1:], torch.zeros_like(logits.grad[..., 1:])))

    def test_weak_real_rain_is_not_pushed_to_zero(self):
        truth = torch.full((1, 1, 1, 2, 2), .08)
        target = current_occurrence_targets(truth, torch.ones_like(truth))
        exact = truth.clone().requires_grad_()
        loss, _ = negative_final_amount_loss(exact, target, truth=truth)
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertTrue(torch.equal(exact.grad, torch.zeros_like(exact)))
        excess = torch.full_like(truth, .12, requires_grad=True)
        loss, _ = negative_final_amount_loss(excess, target, truth=truth)
        loss.backward()
        self.assertGreater(loss.item(), 0.)
        self.assertTrue((excess.grad > 0).all())
        below = torch.full_like(truth, .04, requires_grad=True)
        loss, _ = negative_final_amount_loss(below, target, truth=truth)
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertTrue(torch.equal(below.grad, torch.zeros_like(below)))

    def test_empty_support_has_finite_zero_gradient(self):
        truth = torch.zeros(1, 2, 1, 2, 2)
        target = current_occurrence_targets(truth, torch.zeros_like(truth))
        logits = torch.zeros_like(truth, requires_grad=True)
        loss, _ = balanced_location_loss(logits, target)
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))
        rain = torch.zeros_like(truth, requires_grad=True)
        loss, _ = negative_final_amount_loss(rain, target, truth=truth)
        loss.backward()
        self.assertEqual(loss.item(), 0.)


if __name__ == '__main__':
    unittest.main()
