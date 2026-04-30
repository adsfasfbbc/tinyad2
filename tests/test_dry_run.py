import unittest

import torch

from VisualAD_lib.VisualAD import VisualAD
from utils.anomaly_detection import generate_anomaly_map_from_tokens


class TestTinyClipDryRun(unittest.TestCase):
    def test_forward_and_loss(self):
        torch.manual_seed(0)
        model = VisualAD(
            embed_dim=64,
            image_resolution=28,
            vision_layers=2,
            vision_width=64,
            vision_patch_size=14,
            context_length=0,
            vocab_size=0,
            transformer_width=0,
            transformer_heads=0,
            transformer_layers=0,
            use_text=False,
        )
        model.eval()

        images = torch.randn(2, 3, 28, 28)
        features_list = [1, 2]
        output = model.encode_image(images, features_list)

        anomaly_features = output["anomaly_features"]
        normal_features = output["normal_features"]
        patch_tokens = output["patch_tokens"][-1]
        patch_start_idx = output["patch_start_idx"]

        anomaly_map = generate_anomaly_map_from_tokens(
            anomaly_features,
            normal_features,
            patch_tokens[:, patch_start_idx:, :],
            image_size=28,
        )

        self.assertEqual(anomaly_map.shape, (2, 28, 28))
        loss = anomaly_map.mean()
        self.assertFalse(torch.isnan(loss))


if __name__ == "__main__":
    unittest.main()
