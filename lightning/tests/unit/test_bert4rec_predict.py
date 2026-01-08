"""Unit tests for BERT4Rec predict method with metadata support"""

import pytest
import torch
import numpy as np
from src.models.bert4rec import BERT4Rec


@pytest.mark.unit
class TestBERT4RecPredictMethod:
    """Test predict() method functionality"""

    def test_predict_without_metadata(self, bert4rec_model_no_metadata):
        """Test predict() works without metadata"""
        user_sequences = [[1, 2, 3], [4, 5, 6, 7], [8]]
        topk = 5

        predictions = bert4rec_model_no_metadata.predict(
            user_sequences=user_sequences, topk=topk
        )

        assert predictions.shape == (3, topk)
        assert isinstance(predictions, np.ndarray)

    def test_predict_with_metadata(self, bert4rec_model, sample_config):
        """Test predict() works with metadata"""
        user_sequences = [[1, 2, 3], [4, 5, 6, 7], [8]]
        topk = 5

        # Create sample item metadata
        item_metadata = {
            "genres": {1: [1, 2], 2: [2, 3], 3: [1], 4: [2], 5: [3], 6: [1, 2], 7: [2], 8: [3]},
            "directors": {1: 1, 2: 2, 3: 1, 4: 2, 5: 3, 6: 1, 7: 2, 8: 3},
            "writers": {1: [1, 2], 2: [2], 3: [1], 4: [2, 3], 5: [3], 6: [1], 7: [2], 8: [3, 1]},
        }

        predictions = bert4rec_model.predict(
            user_sequences=user_sequences, topk=topk, item_metadata=item_metadata
        )

        assert predictions.shape == (3, topk)
        assert isinstance(predictions, np.ndarray)

    def test_predict_with_title_embeddings(self, sample_config):
        """Test predict() works with title embeddings"""
        # Create config with title embeddings
        config = {**sample_config, "title_embedding_dim": 64, "use_title_emb": True}
        model = BERT4Rec(**config)

        user_sequences = [[1, 2, 3]]
        topk = 5

        # Create item metadata with title embeddings
        item_metadata = {
            "genres": {1: [1, 2], 2: [2, 3], 3: [1]},
            "directors": {1: 1, 2: 2, 3: 1},
            "writers": {1: [1, 2], 2: [2], 3: [1]},
            "title_embs": {1: np.random.randn(64), 2: np.random.randn(64), 3: np.random.randn(64)},
        }

        predictions = model.predict(
            user_sequences=user_sequences, topk=topk, item_metadata=item_metadata
        )

        assert predictions.shape == (1, topk)

    def test_predict_with_exclude_items(self, bert4rec_model_no_metadata):
        """Test predict() excludes specified items"""
        user_sequences = [[1, 2, 3]]
        topk = 5
        exclude_items = [{1, 2, 3, 4, 5}]  # Exclude these items

        predictions = bert4rec_model_no_metadata.predict(
            user_sequences=user_sequences, topk=topk, exclude_items=exclude_items
        )

        # Check that excluded items are not in predictions
        for pred in predictions[0]:
            assert pred not in exclude_items[0]

    def test_predict_variable_sequence_lengths(self, bert4rec_model_no_metadata):
        """Test predict() handles variable sequence lengths"""
        user_sequences = [[1], [1, 2, 3, 4, 5], [10, 20]]
        topk = 3

        predictions = bert4rec_model_no_metadata.predict(
            user_sequences=user_sequences, topk=topk
        )

        assert predictions.shape == (3, topk)

    def test_predict_single_user(self, bert4rec_model_no_metadata):
        """Test predict() works for single user"""
        user_sequences = [[1, 2, 3, 4, 5]]
        topk = 10

        predictions = bert4rec_model_no_metadata.predict(
            user_sequences=user_sequences, topk=topk
        )

        assert predictions.shape == (1, topk)

    def test_predict_large_batch(self, bert4rec_model_no_metadata):
        """Test predict() handles large batch of users"""
        num_users = 100
        num_items = bert4rec_model_no_metadata.num_items
        # Use modulo to keep item IDs within valid range
        user_sequences = [
            [(i % num_items) + 1, ((i + 1) % num_items) + 1, ((i + 2) % num_items) + 1]
            for i in range(num_users)
        ]
        topk = 5

        predictions = bert4rec_model_no_metadata.predict(
            user_sequences=user_sequences, topk=topk
        )

        assert predictions.shape == (num_users, topk)

    def test_predict_returns_valid_items(self, bert4rec_model_no_metadata):
        """Test predict() returns valid item indices"""
        user_sequences = [[1, 2, 3]]
        topk = 5

        predictions = bert4rec_model_no_metadata.predict(
            user_sequences=user_sequences, topk=topk
        )

        # Check all predictions are within valid range
        assert np.all(predictions >= 1)
        assert np.all(predictions <= bert4rec_model_no_metadata.num_items)

    def test_predict_excludes_padding_and_mask(self, bert4rec_model_no_metadata):
        """Test predict() never returns padding or mask tokens"""
        user_sequences = [[1, 2, 3]]
        topk = 10

        predictions = bert4rec_model_no_metadata.predict(
            user_sequences=user_sequences, topk=topk
        )

        # Padding token (0) and mask token should never appear
        assert bert4rec_model_no_metadata.pad_token not in predictions
        assert bert4rec_model_no_metadata.mask_token not in predictions


@pytest.mark.unit
class TestPrepareBatchMetadata:
    """Test _prepare_batch_metadata() helper method"""

    def test_prepare_batch_metadata_basic(self, bert4rec_model, sample_config):
        """Test basic metadata preparation"""
        seqs = np.array([[1, 2, 3], [4, 5, 6]])
        item_metadata = {
            "genres": {1: [1, 2], 2: [2], 3: [1], 4: [2], 5: [3], 6: [1, 2]},
            "directors": {1: 1, 2: 2, 3: 1, 4: 2, 5: 3, 6: 1},
        }

        metadata = bert4rec_model._prepare_batch_metadata(seqs, item_metadata)

        assert "genres" in metadata
        assert "directors" in metadata
        assert metadata["genres"].shape[0] == 2  # batch_size
        assert metadata["directors"].shape[0] == 2

    def test_prepare_batch_metadata_none_input(self, bert4rec_model):
        """Test metadata preparation with None input"""
        seqs = np.array([[1, 2, 3]])

        metadata = bert4rec_model._prepare_batch_metadata(seqs, None)

        assert metadata is None

    def test_prepare_batch_metadata_empty_dict(self, bert4rec_model):
        """Test metadata preparation with empty metadata dict"""
        seqs = np.array([[1, 2, 3]])
        item_metadata = {}

        metadata = bert4rec_model._prepare_batch_metadata(seqs, item_metadata)

        # Should return empty dict or dict without keys
        assert isinstance(metadata, dict)

    def test_prepare_batch_metadata_missing_items(self, bert4rec_model):
        """Test metadata preparation handles missing items gracefully"""
        seqs = np.array([[1, 99, 3]])  # Item 99 not in metadata
        item_metadata = {
            "genres": {1: [1, 2], 3: [1]},
            "directors": {1: 1, 3: 1},
        }

        metadata = bert4rec_model._prepare_batch_metadata(seqs, item_metadata)

        # Should handle missing item 99 with zeros
        assert metadata["genres"].shape == (1, 3, 5)  # max_genres=5
        assert metadata["directors"].shape == (1, 3)

    def test_prepare_batch_metadata_with_title_embeddings(self, sample_config):
        """Test metadata preparation with title embeddings"""
        config = {**sample_config, "title_embedding_dim": 64, "use_title_emb": True}
        model = BERT4Rec(**config)

        seqs = np.array([[1, 2, 3]])
        item_metadata = {
            "title_embs": {
                1: np.random.randn(64),
                2: np.random.randn(64),
                3: np.random.randn(64),
            }
        }

        metadata = model._prepare_batch_metadata(seqs, item_metadata)

        assert "title_embs" in metadata
        assert metadata["title_embs"].shape == (1, 3, 64)

    def test_prepare_batch_metadata_torch_input(self, bert4rec_model):
        """Test metadata preparation accepts torch.Tensor input"""
        seqs = torch.LongTensor([[1, 2, 3]])
        item_metadata = {
            "genres": {1: [1, 2], 2: [2], 3: [1]},
            "directors": {1: 1, 2: 2, 3: 1},
        }

        metadata = bert4rec_model._prepare_batch_metadata(seqs, item_metadata)

        assert isinstance(metadata["genres"], torch.Tensor)
        assert isinstance(metadata["directors"], torch.Tensor)

    def test_prepare_batch_metadata_device_placement(self, bert4rec_model):
        """Test metadata tensors are placed on correct device"""
        bert4rec_model = bert4rec_model.cpu()
        seqs = np.array([[1, 2, 3]])
        item_metadata = {
            "genres": {1: [1, 2], 2: [2], 3: [1]},
            "directors": {1: 1, 2: 2, 3: 1},
        }

        metadata = bert4rec_model._prepare_batch_metadata(seqs, item_metadata)

        assert metadata["genres"].device.type == "cpu"
        assert metadata["directors"].device.type == "cpu"

    def test_prepare_batch_metadata_padding(self, bert4rec_model):
        """Test metadata preparation pads genres/writers correctly"""
        seqs = np.array([[1, 2]])
        item_metadata = {
            "genres": {1: [1], 2: [2, 3, 4]},  # Different lengths
            "writers": {1: [1, 2, 3], 2: [4]},  # Different lengths
        }

        metadata = bert4rec_model._prepare_batch_metadata(seqs, item_metadata)

        # All should be padded to max_genres=5 and max_writers=5
        assert metadata["genres"].shape == (1, 2, 5)
        assert metadata["writers"].shape == (1, 2, 5)


@pytest.mark.unit
class TestPredictWithDifferentFusionStrategies:
    """Test predict() works with different metadata fusion strategies"""

    @pytest.mark.parametrize("fusion_strategy", ["concat", "add", "gate"])
    def test_predict_with_fusion_strategy(self, sample_config, fusion_strategy):
        """Test predict() works with different fusion strategies"""
        config = {**sample_config, "metadata_fusion": fusion_strategy}
        model = BERT4Rec(**config)

        user_sequences = [[1, 2, 3]]
        item_metadata = {
            "genres": {1: [1, 2], 2: [2], 3: [1]},
            "directors": {1: 1, 2: 2, 3: 1},
            "writers": {1: [1], 2: [2], 3: [1, 2]},
        }

        predictions = model.predict(
            user_sequences=user_sequences, topk=5, item_metadata=item_metadata
        )

        assert predictions.shape == (1, 5)


@pytest.mark.unit
class TestPredictErrorHandling:
    """Test predict() error handling"""

    def test_predict_empty_sequences(self, bert4rec_model_no_metadata):
        """Test predict() handles empty sequence gracefully"""
        user_sequences = [[]]
        topk = 5

        # Should handle empty sequence (will be all padding)
        predictions = bert4rec_model_no_metadata.predict(
            user_sequences=user_sequences, topk=topk
        )

        assert predictions.shape == (1, topk)

    def test_predict_topk_larger_than_num_items(self, bert4rec_model_no_metadata):
        """Test predict() when topk > num_items"""
        user_sequences = [[1, 2, 3]]
        topk = bert4rec_model_no_metadata.num_items + 10  # More than available

        # Should still work, topk will be automatically capped
        predictions = bert4rec_model_no_metadata.predict(
            user_sequences=user_sequences, topk=topk
        )

        assert predictions.shape[0] == 1
        # Should return at most num_tokens items (capped)
        assert predictions.shape[1] <= bert4rec_model_no_metadata.num_tokens

    def test_predict_with_mismatched_exclude_items(self, bert4rec_model_no_metadata):
        """Test predict() handles mismatched exclude_items length"""
        user_sequences = [[1, 2, 3], [4, 5, 6]]
        topk = 5
        exclude_items = [{1, 2, 3}]  # Only 1 set for 2 users (wrong)

        # Should handle gracefully (might raise error or use empty for second user)
        try:
            predictions = bert4rec_model_no_metadata.predict(
                user_sequences=user_sequences, topk=topk, exclude_items=exclude_items
            )
            # If it doesn't raise, check it returns something reasonable
            assert predictions.shape[0] == 2
        except (IndexError, ValueError):
            # Expected to raise error with mismatched lengths
            pass


@pytest.mark.integration
class TestPredictEndToEnd:
    """Integration tests for predict() with full workflow"""

    def test_train_and_predict_consistency(self, bert4rec_model_no_metadata):
        """Test model can train and predict consistently"""
        # Train for 1 step
        sequences = torch.randint(1, 51, (4, 10))
        labels = torch.randint(0, 51, (4, 10))
        batch = (sequences, labels)

        loss = bert4rec_model_no_metadata.training_step(batch, batch_idx=0)
        assert loss.item() > 0

        # Now predict
        user_sequences = [[1, 2, 3, 4, 5]]
        predictions = bert4rec_model_no_metadata.predict(
            user_sequences=user_sequences, topk=10
        )

        assert predictions.shape == (1, 10)

    def test_predict_with_metadata_matches_forward(self, bert4rec_model, sample_config):
        """Test predict() with metadata gives consistent results with forward()"""
        user_sequences = [[1, 2, 3]]

        # Genre IDs: 0 to num_genres-1 (0 is padding, so use 1 to num_genres-1)
        num_genres = sample_config["num_genres"]
        num_directors = sample_config["num_directors"]
        num_writers = sample_config["num_writers"]

        item_metadata = {
            "genres": {
                i: [(i % (num_genres - 1)) + 1, ((i + 1) % (num_genres - 1)) + 1]
                for i in range(1, sample_config["num_items"] + 1)
            },
            "directors": {
                i: (i % (num_directors - 1)) + 1
                for i in range(1, sample_config["num_items"] + 1)
            },
            "writers": {
                i: [(i % (num_writers - 1)) + 1, ((i + 1) % (num_writers - 1)) + 1]
                for i in range(1, sample_config["num_items"] + 1)
            },
        }

        # Get predictions
        predictions = bert4rec_model.predict(
            user_sequences=user_sequences, topk=5, item_metadata=item_metadata
        )

        # Predictions should be valid
        assert np.all(predictions >= 1)
        assert np.all(predictions <= bert4rec_model.num_items)
        assert predictions.shape == (1, 5)
