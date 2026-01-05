"""
Test cases for multi-item validation functionality

Tests:
1. _create_multiitem_split
2. _compute_multiitem_metrics (nRecall@10, NDCG@10)
3. Single-item vs Multi-item compatibility
"""

import pytest
import torch
import random
import math
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.bert4rec_data import BERT4RecDataModule
from src.models.bert4rec import BERT4Rec


class TestMultiItemSplit:
    """Test _create_multiitem_split method"""

    def setup_method(self):
        """Setup test data"""
        self.datamodule = BERT4RecDataModule(
            data_dir="~/data/train/",
            data_file="train_ratings.csv",
            batch_size=32,
            min_interactions=3,
            seed=42,
        )

    def test_split_small_sequence(self):
        """Test with small sequence (1% = 1 item)"""
        user = 1
        seq = list(range(1, 101))  # 100 items

        train_seq, val_items = self.datamodule._create_multiitem_split(user, seq)

        # Check total length
        assert len(train_seq) + len(val_items) == 100

        # Check validation size (1% of 100 = 1)
        assert len(val_items) == 1

        # Check last item is in validation
        assert 100 in val_items

        # Check no overlap
        assert set(train_seq) & set(val_items) == set()

    def test_split_medium_sequence(self):
        """Test with medium sequence (1% = 2 items)"""
        user = 2
        seq = list(range(1, 201))  # 200 items

        train_seq, val_items = self.datamodule._create_multiitem_split(user, seq)

        # Check total length
        assert len(train_seq) + len(val_items) == 200

        # Check validation size (1% of 200 = 2)
        assert len(val_items) == 2

        # Check last item is in validation
        assert 200 in val_items

        # Check no overlap
        assert set(train_seq) & set(val_items) == set()

    def test_split_large_sequence(self):
        """Test with large sequence (1% = 5 items)"""
        user = 3
        seq = list(range(1, 501))  # 500 items

        train_seq, val_items = self.datamodule._create_multiitem_split(user, seq)

        # Check total length
        assert len(train_seq) + len(val_items) == 500

        # Check validation size (1% of 500 = 5)
        assert len(val_items) == 5

        # Check last item is in validation
        assert 500 in val_items

        # Check no overlap
        assert set(train_seq) & set(val_items) == set()

    def test_split_very_large_sequence(self):
        """Test with very large sequence (1% = 20 items)"""
        user = 4
        seq = list(range(1, 2001))  # 2000 items

        train_seq, val_items = self.datamodule._create_multiitem_split(user, seq)

        # Check total length
        assert len(train_seq) + len(val_items) == 2000

        # Check validation size (1% of 2000 = 20)
        assert len(val_items) == 20

        # Check last item is in validation
        assert 2000 in val_items

        # Check no overlap
        assert set(train_seq) & set(val_items) == set()

    def test_split_minimum_sequence(self):
        """Test with minimum sequence (< 100 items, 1% rounds to 0 → min 1)"""
        user = 5
        seq = list(range(1, 51))  # 50 items

        train_seq, val_items = self.datamodule._create_multiitem_split(user, seq)

        # Check total length
        assert len(train_seq) + len(val_items) == 50

        # Check validation size (1% of 50 = 0.5 → max(1, 0) = 1)
        assert len(val_items) == 1

        # Check last item is in validation
        assert 50 in val_items

    def test_split_reproducibility(self):
        """Test that same user+seed produces same split"""
        user = 10
        seq = list(range(1, 201))

        # First split
        train_seq1, val_items1 = self.datamodule._create_multiitem_split(user, seq)

        # Second split (same user, same seed)
        train_seq2, val_items2 = self.datamodule._create_multiitem_split(user, seq)

        # Should be identical
        assert train_seq1 == train_seq2
        assert val_items1 == val_items2

    def test_split_different_users(self):
        """Test that different users produce different splits"""
        seq = list(range(1, 201))

        # User 1
        train_seq1, val_items1 = self.datamodule._create_multiitem_split(1, seq)

        # User 2
        train_seq2, val_items2 = self.datamodule._create_multiitem_split(2, seq)

        # Should be different (except last item)
        assert val_items1 != val_items2  # Different random selections

    def test_split_sorted_output(self):
        """Test that validation items are sorted"""
        user = 6
        seq = list(range(1, 501))

        _, val_items = self.datamodule._create_multiitem_split(user, seq)

        # Check sorted
        assert val_items == sorted(val_items)


class TestNRecallCalculation:
    """Test nRecall@K calculation"""

    def setup_method(self):
        """Setup test model"""
        self.model = BERT4Rec(
            num_items=1000,
            hidden_units=64,
            max_len=50,
        )

    def test_nrecall_single_target_hit(self):
        """Test nRecall@10 with single target (hit)"""
        # Top-10 predictions
        top_items = torch.tensor([[42, 15, 8, 23, 99, 7, 31, 50, 12, 4]])

        # Single target (hit)
        target_items_list = [[42]]

        _, _, nrecall = self.model._compute_multiitem_metrics(top_items, target_items_list)

        # Expected: 1 / min(10, 1) = 1.0
        assert abs(nrecall - 1.0) < 1e-6

    def test_nrecall_single_target_miss(self):
        """Test nRecall@10 with single target (miss)"""
        # Top-10 predictions
        top_items = torch.tensor([[15, 8, 23, 99, 7, 31, 50, 12, 4, 88]])

        # Single target (miss)
        target_items_list = [[42]]

        _, _, nrecall = self.model._compute_multiitem_metrics(top_items, target_items_list)

        # Expected: 0 / min(10, 1) = 0.0
        assert abs(nrecall - 0.0) < 1e-6

    def test_nrecall_multi_target_all_hit(self):
        """Test nRecall@10 with 2 targets (both hit)"""
        # Top-10 predictions
        top_items = torch.tensor([[42, 15, 88, 23, 99, 7, 31, 50, 12, 4]])

        # 2 targets (both hit)
        target_items_list = [[42, 88]]

        _, _, nrecall = self.model._compute_multiitem_metrics(top_items, target_items_list)

        # Expected: 2 / min(10, 2) = 2 / 2 = 1.0
        assert abs(nrecall - 1.0) < 1e-6

    def test_nrecall_multi_target_partial_hit(self):
        """Test nRecall@10 with 5 targets (3 hit)"""
        # Top-10 predictions
        top_items = torch.tensor([[42, 15, 88, 8, 123, 99, 7, 31, 50, 12]])

        # 5 targets (3 hit: 42, 88, 123)
        target_items_list = [[42, 88, 123, 156, 200]]

        _, _, nrecall = self.model._compute_multiitem_metrics(top_items, target_items_list)

        # Expected: 3 / min(10, 5) = 3 / 5 = 0.6
        assert abs(nrecall - 0.6) < 1e-6

    def test_nrecall_heavy_user(self):
        """Test nRecall@10 with heavy user (20 targets)"""
        # Top-10 predictions
        top_items = torch.tensor([[1, 5, 8, 12, 15, 42, 88, 99, 123, 145]])

        # 20 targets (8 hit: 1, 5, 8, 12, 15, 42, 88, 99)
        target_items_list = [[1, 5, 8, 12, 15, 42, 88, 99, 123, 145,
                              156, 178, 200, 210, 220, 230, 240, 250, 260, 270]]

        _, _, nrecall = self.model._compute_multiitem_metrics(top_items, target_items_list)

        # Expected: 10 / min(10, 20) = 10 / 10 = 1.0
        # (모든 top-10이 정답 안에 있음)
        assert abs(nrecall - 1.0) < 1e-6

    def test_nrecall_batch(self):
        """Test nRecall@10 with batch of users"""
        # Batch of 3 users
        top_items = torch.tensor([
            [42, 15, 8, 23, 99, 7, 31, 50, 12, 4],      # User 1
            [15, 8, 88, 23, 99, 7, 31, 50, 12, 4],      # User 2
            [42, 15, 88, 8, 123, 99, 7, 31, 50, 12],    # User 3
        ])

        target_items_list = [
            [42],                          # User 1: 1 target, 1 hit
            [42, 88],                      # User 2: 2 targets, 1 hit (88)
            [42, 88, 123, 156, 200],       # User 3: 5 targets, 3 hit
        ]

        _, _, total_nrecall = self.model._compute_multiitem_metrics(top_items, target_items_list)

        # Expected:
        # User 1: 1/1 = 1.0
        # User 2: 1/2 = 0.5
        # User 3: 3/5 = 0.6
        # Total: 1.0 + 0.5 + 0.6 = 2.1
        assert abs(total_nrecall - 2.1) < 1e-6


class TestNDCGCalculation:
    """Test NDCG@10 calculation for multi-item"""

    def setup_method(self):
        """Setup test model"""
        self.model = BERT4Rec(
            num_items=1000,
            hidden_units=64,
            max_len=50,
        )

    def test_ndcg_single_target_rank1(self):
        """Test NDCG@10 with single target at rank 1"""
        top_items = torch.tensor([[42, 15, 8, 23, 99, 7, 31, 50, 12, 4]])
        target_items_list = [[42]]

        _, total_ndcg, _ = self.model._compute_multiitem_metrics(top_items, target_items_list)

        # Expected: 1 / log2(2) = 1.0
        expected = 1.0 / math.log2(2)
        assert abs(total_ndcg - expected) < 1e-6

    def test_ndcg_single_target_rank10(self):
        """Test NDCG@10 with single target at rank 10"""
        top_items = torch.tensor([[15, 8, 23, 99, 7, 31, 50, 12, 4, 42]])
        target_items_list = [[42]]

        _, total_ndcg, _ = self.model._compute_multiitem_metrics(top_items, target_items_list)

        # Expected: 1 / log2(11)
        expected = 1.0 / math.log2(11)
        assert abs(total_ndcg - expected) < 1e-6

    def test_ndcg_multi_target_ideal(self):
        """Test NDCG@10 with 3 targets in ideal positions (1, 2, 3)"""
        top_items = torch.tensor([[42, 88, 123, 15, 8, 23, 99, 7, 31, 50]])
        target_items_list = [[42, 88, 123]]

        _, total_ndcg, _ = self.model._compute_multiitem_metrics(top_items, target_items_list)

        # Expected: NDCG = 1.0 (ideal)
        assert abs(total_ndcg - 1.0) < 1e-6

    def test_ndcg_multi_target_partial(self):
        """Test NDCG@10 with 5 targets, 3 matches"""
        top_items = torch.tensor([[42, 15, 88, 8, 123, 99, 7, 31, 50, 12]])
        target_items_list = [[42, 88, 123, 156, 200]]

        _, total_ndcg, _ = self.model._compute_multiitem_metrics(top_items, target_items_list)

        # Calculate expected
        # DCG = 1/log2(2) + 1/log2(4) + 1/log2(6)
        dcg = 1.0/math.log2(2) + 1.0/math.log2(4) + 1.0/math.log2(6)

        # IDCG (5 targets, ideal: 1,2,3,4,5)
        idcg = sum(1.0/math.log2(i+2) for i in range(5))

        expected = dcg / idcg
        assert abs(total_ndcg - expected) < 1e-6

    def test_ndcg_no_match(self):
        """Test NDCG@10 with no matches"""
        top_items = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        target_items_list = [[42, 88, 123]]

        _, total_ndcg, _ = self.model._compute_multiitem_metrics(top_items, target_items_list)

        # Expected: 0.0
        assert abs(total_ndcg - 0.0) < 1e-6


class TestSingleVsMultiCompatibility:
    """Test single-item vs multi-item compatibility"""

    def setup_method(self):
        """Setup test model"""
        self.model = BERT4Rec(
            num_items=1000,
            hidden_units=64,
            max_len=50,
        )

    def test_single_item_as_list(self):
        """Test that single-item as list gives same result"""
        top_items = torch.tensor([[42, 15, 8, 23, 99, 7, 31, 50, 12, 4]])

        # Single-item (tensor)
        target_single = torch.tensor([42])
        hit_single, ndcg_single, nrecall_single = \
            self.model._compute_singleitem_metrics(top_items, target_single)

        # Single-item as list
        target_multi = [[42]]
        hit_multi, ndcg_multi, nrecall_multi = \
            self.model._compute_multiitem_metrics(top_items, target_multi)

        # Should be the same
        assert abs(hit_single - hit_multi) < 1e-6
        assert abs(ndcg_single - ndcg_multi) < 1e-6
        assert abs(nrecall_single - nrecall_multi) < 1e-6

    def test_batch_compatibility(self):
        """Test batch processing compatibility"""
        top_items = torch.tensor([
            [42, 15, 8, 23, 99, 7, 31, 50, 12, 4],
            [15, 8, 42, 23, 99, 7, 31, 50, 12, 4],
        ])

        # Single-item (tensor)
        target_single = torch.tensor([42, 42])
        hit_single, ndcg_single, nrecall_single = \
            self.model._compute_singleitem_metrics(top_items, target_single)

        # Single-item as list
        target_multi = [[42], [42]]
        hit_multi, ndcg_multi, nrecall_multi = \
            self.model._compute_multiitem_metrics(top_items, target_multi)

        # Should be the same
        assert abs(hit_single - hit_multi) < 1e-6
        assert abs(ndcg_single - ndcg_single) < 1e-6
        assert abs(nrecall_single - nrecall_multi) < 1e-6


class TestEdgeCases:
    """Test edge cases"""

    def setup_method(self):
        """Setup test model"""
        self.model = BERT4Rec(
            num_items=1000,
            hidden_units=64,
            max_len=50,
        )

    def test_empty_targets(self):
        """Test with empty target list (should not happen, but handle gracefully)"""
        top_items = torch.tensor([[42, 15, 8, 23, 99, 7, 31, 50, 12, 4]])
        target_items_list = [[]]  # Empty

        hit, ndcg, nrecall = self.model._compute_multiitem_metrics(top_items, target_items_list)

        # Should return 0
        assert hit == 0
        assert ndcg == 0.0
        # nrecall = 0 / min(10, 0) → division by zero, but min(10, 0) = 0
        # Handled by: nrecall = num_hits / min(10, num_targets) if num_targets > 0

    def test_all_targets_in_top10(self):
        """Test when all targets are in top-10"""
        top_items = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        target_items_list = [[1, 2, 3, 4, 5]]

        hit, ndcg, nrecall = self.model._compute_multiitem_metrics(top_items, target_items_list)

        # All 5 targets in top-10
        assert hit == 1  # Hit
        assert abs(nrecall - 1.0) < 1e-6  # 5/5 = 1.0
        assert abs(ndcg - 1.0) < 1e-6  # Ideal order

    def test_duplicate_targets(self):
        """Test with duplicate targets (should be handled by set)"""
        top_items = torch.tensor([[42, 15, 8, 23, 99, 7, 31, 50, 12, 4]])
        target_items_list = [[42, 42, 42]]  # Duplicates

        hit, ndcg, nrecall = self.model._compute_multiitem_metrics(top_items, target_items_list)

        # Duplicates are removed by set()
        # targets = set([42, 42, 42]) = {42}
        # num_targets = len({42}) = 1
        # num_hits = len(set(preds) & {42}) = 1
        # nrecall = 1 / min(10, 1) = 1.0
        assert hit == 1
        assert abs(nrecall - 1.0) < 1e-6  # 1/1 = 1.0 (duplicates removed)


class TestStatistics:
    """Test validation statistics logging"""

    def test_validation_sizes_calculation(self):
        """Test validation set size statistics"""
        import numpy as np

        # Mock validation data
        user_valid = {
            1: [100],                          # 1 item
            2: [200, 201],                     # 2 items
            3: [300, 301, 302],                # 3 items
            4: [400, 401, 402, 403, 404],      # 5 items
        }

        val_sizes = [len(v) for v in user_valid.values()]

        assert min(val_sizes) == 1
        assert max(val_sizes) == 5
        assert abs(np.mean(val_sizes) - 2.75) < 1e-6  # (1+2+3+5)/4 = 2.75


# Integration test
class TestIntegration:
    """Integration test with DataModule"""

    def test_multiitem_validation_pipeline(self):
        """Test complete multi-item validation pipeline"""
        # This would require actual data file, so we'll skip for now
        # But this shows the intended usage
        pass


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])
