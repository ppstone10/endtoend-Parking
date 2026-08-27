"""模型注册表测试。"""

import unittest

import torch

from model import available_models, build_model


class TestModelRegistry(unittest.TestCase):
    def test_registry_builds_all_declared_variants(self):
        self.assertEqual(available_models(), ("net-v0", "net-v1", "net-v2"))
        v0 = build_model("net-v0", {"bev_channels": 5, "horizon": 6})
        v1 = build_model(
            "net-v1", {"bev_channels": 5, "max_horizon": 6, "hidden_dim": 32}
        )
        v2 = build_model(
            "net-v2",
            {
                "bev_channels": 5,
                "max_horizon": 6,
                "hidden_dim": 32,
                "base_channels": 8,
                "attention_heads": 4,
            },
        )
        bev = torch.randn(2, 5, 32, 32)
        goal = torch.randn(2, 3)
        state = torch.randn(2, 2)
        self.assertEqual(v0(bev, goal, state).shape, (2, 6, 3))
        self.assertEqual(v1(bev, goal, state).shape, (2, 6, 3))
        self.assertEqual(v2(bev, goal, state).shape, (2, 6, 3))

    def test_unknown_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "未知模型"):
            build_model("net-v9")


if __name__ == "__main__":
    unittest.main()
