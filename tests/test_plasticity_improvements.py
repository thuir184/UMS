import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace

from models.LEAR import LEAR, parse_apply_to
from backbone.LEAR import LEAR as LEARBackbone
from backbone.bilora import BiLORA_Focal


class MockDataset:
    def __init__(self, n_classes=10, n_tasks=10, device='cpu'):
        self.N_CLASSES = n_classes
        self.num_classes = n_classes
        self.N_TASKS = n_tasks
        self.n_tasks = n_tasks
        self.SETTING = 'domain-il'
        self.N_CLASSES_PER_TASK = n_classes
        self.device = device
        self.train_loader = [
            (torch.randn(2, 3, 224, 224, device=self.device), torch.tensor([0, 1], device=self.device))
        ]

    def get_normalization_transform(self):
        return None


def get_base_args():
    return Namespace(
        dataset='seq-cifar10',
        model='LEAR',
        backbone='lear',
        lr=0.03,
        batch_size=4,
        n_epochs=1,
        transform_type='weak',
        n_frq=16,
        n_tasks=4,
        modulate_q=True,
        modulate_proj=True,
        ce_thresh=1e-4,
        ums_basis_rank=32,
        ums_svd_rank=8,
        ums_tau=1e-3,
        ums_rot_steps=2,
        ums_rot_lr=0.005,
        ums_head_lr=0.05,
        ums_rot_lambda_diag=0.1,
        ums_rot_lambda_anchor=0.5,
        ums_rot_lambda_support_mse=1.0,
        ums_rot_lambda_support_bce=0.1,
        ums_rot_lambda_gaussian=0.01,
        disable_task_checkpoints=True,
        checkpoint_dir='checkpoints',
        label_perc=1.0,
        optimizer='sgd',
        optim_wd=0.0,
        optim_mom=0.0,
        optim_nesterov=False,
        nowand=1,
        apply_to='9,10,11',
    )


class TestPlasticityImprovements(unittest.TestCase):
    def test_parse_apply_to(self):
        # Default/None
        self.assertEqual(parse_apply_to(None), [9, 10, 11])
        # 'all'
        self.assertEqual(parse_apply_to('all'), list(range(12)))
        self.assertEqual(parse_apply_to('ALL'), list(range(12)))
        # Comma-separated string
        self.assertEqual(parse_apply_to('9,10,11'), [9, 10, 11])
        self.assertEqual(parse_apply_to('0, 1, 2'), [0, 1, 2])
        self.assertEqual(parse_apply_to('6,7,8,9,10,11'), [6, 7, 8, 9, 10, 11])
        # List of ints
        self.assertEqual(parse_apply_to([0, 5, 11]), [0, 5, 11])
        # Invalid index raises ValueError
        with self.assertRaises(ValueError):
            parse_apply_to('12')
        with self.assertRaises(ValueError):
            parse_apply_to('-1')

    def test_improvement_1_apply_to_all_blocks(self):
        args = get_base_args()
        args.apply_to = 'all'
        dataset = MockDataset(n_classes=10)
        backbone = LEARBackbone(num_classes=10)
        loss = nn.CrossEntropyLoss()
        model = LEAR(backbone, loss, args, transform=None, dataset=dataset)
        model.to(model.device)
        model.net.to(model.device)

        self.assertEqual(model.trainable_blocks, list(range(12)))
        for i in range(12):
            block = model.net.local_vitmodel.blocks[i]
            self.assertTrue(isinstance(block.attn, BiLORA_Focal), f"Block {i} should have BiLORA_Focal")

        # Forward fusion works through all 12 blocks
        x = torch.randn(2, 3, 224, 224, device=model.device)
        out, feats = model.net(x, return_features=True)
        self.assertEqual(out.shape, (2, 10))
        self.assertEqual(feats.shape, (2, 768))

    def test_improvement_1_apply_to_subset_blocks(self):
        args = get_base_args()
        args.apply_to = '9,10,11'
        dataset = MockDataset(n_classes=10)
        backbone = LEARBackbone(num_classes=10)
        loss = nn.CrossEntropyLoss()
        model = LEAR(backbone, loss, args, transform=None, dataset=dataset)
        model.to(model.device)
        model.net.to(model.device)

        self.assertEqual(model.trainable_blocks, [9, 10, 11])
        for i in range(9):
            block = model.net.local_vitmodel.blocks[i]
            self.assertFalse(isinstance(block.attn, BiLORA_Focal), f"Block {i} should NOT have BiLORA_Focal")
        for i in [9, 10, 11]:
            block = model.net.local_vitmodel.blocks[i]
            self.assertTrue(isinstance(block.attn, BiLORA_Focal), f"Block {i} should have BiLORA_Focal")

    def test_bilora_focal_pretrained_weights_copied(self):
        args = get_base_args()
        args.apply_to = '9,10,11'
        dataset = MockDataset(n_classes=10)
        backbone = LEARBackbone(num_classes=10)
        loss = nn.CrossEntropyLoss()
        model = LEAR(backbone, loss, args, transform=None, dataset=dataset)

        for i in [9, 10, 11]:
            new_attn = model.net.local_vitmodel.blocks[i].attn
            fe_attn = model.net.feature_extractor.blocks[i].attn

            self.assertEqual(new_attn.num_heads, 12)
            self.assertAlmostEqual(new_attn.scale, 0.125, places=5)
            self.assertIsNotNone(new_attn.qkv.bias)
            self.assertIsNotNone(new_attn.proj.bias)

            self.assertTrue(torch.equal(new_attn.qkv.weight, fe_attn.qkv.weight))
            self.assertTrue(torch.equal(new_attn.qkv.bias, fe_attn.qkv.bias))
            self.assertTrue(torch.equal(new_attn.proj.weight, fe_attn.proj.weight))
            self.assertTrue(torch.equal(new_attn.proj.bias, fe_attn.proj.bias))


if __name__ == '__main__':
    unittest.main()
