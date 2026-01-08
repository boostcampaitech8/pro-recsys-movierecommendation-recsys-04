# Test Coverage Summary

## New Tests for Metadata Support in Prediction

### 1. `test_bert4rec_predict.py` - Predict Method Tests

Tests for the new `predict()` method with metadata support and `_prepare_batch_metadata()` helper.

#### TestBERT4RecPredictMethod (10 tests)
- ✅ `test_predict_without_metadata` - Basic prediction without metadata
- ✅ `test_predict_with_metadata` - Prediction with full metadata (genres, directors, writers)
- ✅ `test_predict_with_title_embeddings` - Prediction with title embeddings
- ✅ `test_predict_with_exclude_items` - Item exclusion functionality
- ✅ `test_predict_variable_sequence_lengths` - Variable length sequences
- ✅ `test_predict_single_user` - Single user prediction
- ✅ `test_predict_large_batch` - Large batch (100 users) handling
- ✅ `test_predict_returns_valid_items` - Output validation
- ✅ `test_predict_excludes_padding_and_mask` - Special token exclusion

#### TestPrepareBatchMetadata (9 tests)
- ✅ `test_prepare_batch_metadata_basic` - Basic metadata preparation
- ✅ `test_prepare_batch_metadata_none_input` - None input handling
- ✅ `test_prepare_batch_metadata_empty_dict` - Empty metadata dict
- ✅ `test_prepare_batch_metadata_missing_items` - Graceful handling of missing items
- ✅ `test_prepare_batch_metadata_with_title_embeddings` - Title embedding support
- ✅ `test_prepare_batch_metadata_torch_input` - Torch tensor input support
- ✅ `test_prepare_batch_metadata_device_placement` - Device handling (CPU/GPU)
- ✅ `test_prepare_batch_metadata_padding` - Correct padding for genres/writers

#### TestPredictWithDifferentFusionStrategies (1 test)
- ✅ `test_predict_with_fusion_strategy` - Parametrized test for concat/add/gate fusion

#### TestPredictErrorHandling (3 tests)
- ✅ `test_predict_empty_sequences` - Empty sequence handling
- ✅ `test_predict_topk_larger_than_num_items` - topk > num_items edge case
- ✅ `test_predict_with_mismatched_exclude_items` - Mismatch detection

#### TestPredictEndToEnd (2 tests)
- ✅ `test_train_and_predict_consistency` - Train → Predict workflow
- ✅ `test_predict_with_metadata_matches_forward` - Consistency check

**Total: 25 tests**

---

### 2. `test_data_module.py` - DataModule Tests (New Section)

Tests for the new `get_item_metadata()` method in BERT4RecDataModule.

#### TestGetItemMetadata (10 tests)
- ✅ `test_get_item_metadata_with_all_metadata` - Returns all metadata types
- ✅ `test_get_item_metadata_structure` - Correct dict structure
- ✅ `test_get_item_metadata_genres_are_lists` - Genre values are lists
- ✅ `test_get_item_metadata_directors_are_ints` - Director values are integers
- ✅ `test_get_item_metadata_writers_are_lists` - Writer values are lists
- ✅ `test_get_item_metadata_empty_when_no_metadata` - Returns None without metadata files
- ✅ `test_get_item_metadata_called_before_setup_fails` - Proper error before setup()
- ✅ `test_get_item_metadata_idempotent` - Multiple calls return same data
- ✅ `test_get_item_metadata_contains_valid_item_ids` - Valid item ID ranges

**Total: 10 tests**

---

## Running the Tests

### Run all new tests:
```bash
pytest tests/unit/test_bert4rec_predict.py -v
pytest tests/unit/test_data_module.py::TestGetItemMetadata -v
```

### Run specific test class:
```bash
pytest tests/unit/test_bert4rec_predict.py::TestBERT4RecPredictMethod -v
pytest tests/unit/test_bert4rec_predict.py::TestPrepareBatchMetadata -v
```

### Run with coverage:
```bash
pytest tests/unit/test_bert4rec_predict.py --cov=src.models.bert4rec --cov-report=term-missing
pytest tests/unit/test_data_module.py::TestGetItemMetadata --cov=src.data.bert4rec_data --cov-report=term-missing
```

### Run only unit tests:
```bash
pytest tests/unit/ -m unit -v
```

### Run integration tests:
```bash
pytest tests/unit/ -m integration -v
```

---

## Test Markers

- `@pytest.mark.unit` - Fast unit tests (default)
- `@pytest.mark.integration` - Slower integration tests
- `@pytest.mark.gpu` - Tests requiring GPU

---

## What These Tests Cover

### 1. Core Functionality
- ✅ Metadata preparation for inference batches
- ✅ Prediction with and without metadata
- ✅ All fusion strategies (concat, add, gate)
- ✅ Title embedding support

### 2. Edge Cases
- ✅ Empty sequences
- ✅ Variable sequence lengths
- ✅ Missing metadata for items
- ✅ topk > num_items
- ✅ Mismatched input lengths

### 3. Integration
- ✅ Train → Predict workflow
- ✅ Forward pass consistency
- ✅ DataModule integration

### 4. Data Validation
- ✅ Output shape validation
- ✅ Valid item ID ranges
- ✅ No NaN/Inf in outputs
- ✅ Special token exclusion

---

## Related Files

### Source Files Tested:
- `src/models/bert4rec.py`
  - `predict()` method
  - `_prepare_batch_metadata()` helper

- `src/data/bert4rec_data.py`
  - `get_item_metadata()` method

### Test Fixtures Used:
- `conftest.py`:
  - `sample_config` - Config with metadata
  - `sample_config_no_metadata` - Config without metadata
  - `sample_metadata` - Sample metadata tensors
  - `bert4rec_model` - Model with metadata
  - `bert4rec_model_no_metadata` - Model without metadata
  - `temp_data_dir` - Temporary test data

---

## Future Test Additions

Consider adding:
1. **Performance tests** - Benchmark predict() with large batches
2. **Memory tests** - Check for memory leaks in batch processing
3. **Concurrent tests** - Multiple predict() calls in parallel
4. **GPU tests** - Metadata preparation on GPU
5. **Real data tests** - Integration test with actual checkpoint
