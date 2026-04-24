import unittest

import torch

from utils.distill_loss import DistillationLoss, DistillLossConfig


class TestTokenReshape(unittest.TestCase):
    def test_teacher_patch_reshape_shape(self):
        tokens = torch.randn(2, 576, 1024)
        mapped = DistillationLoss(DistillLossConfig())._teacher_tokens_to_map(tokens)
        self.assertEqual(mapped.shape, (2, 1024, 24, 24))

    def test_teacher_patch_reshape_order(self):
        b, n, c = 1, 576, 4
        tokens = torch.arange(b * n * c, dtype=torch.float32).reshape(b, n, c)
        mapped = DistillationLoss(DistillLossConfig())._teacher_tokens_to_map(tokens)

        # token index 0 -> position (0,0), token index 1 -> (0,1), row-major order
        self.assertTrue(torch.equal(mapped[0, :, 0, 0], tokens[0, 0, :]))
        self.assertTrue(torch.equal(mapped[0, :, 0, 1], tokens[0, 1, :]))
        self.assertTrue(torch.equal(mapped[0, :, 1, 0], tokens[0, 24, :]))


if __name__ == "__main__":
    unittest.main()
