# BERT4Rec Class Diagram

```mermaid
classDiagram
    %% Core Model Components
    class BERT4Rec {
        +int num_items
        +int hidden_units
        +int num_heads
        +int num_layers
        +int max_len
        +float dropout_rate
        +float mask_prob
        +float lr
        +float weight_decay
        +bool share_embeddings
        +int pad_token
        +int mask_token
        +int num_tokens
        +Embedding item_emb
        +Embedding pos_emb
        +Dropout dropout
        +LayerNorm emb_layernorm
        +ModuleList blocks
        +Linear out
        +CrossEntropyLoss criterion
        +__init__(num_items, hidden_units, num_heads, num_layers, max_len, dropout_rate, mask_prob, lr, weight_decay, share_embeddings)
        +forward(log_seqs) Tensor
        +mask_sequence(seq) tuple
        +training_step(batch, batch_idx) Tensor
        +validation_step(batch, batch_idx) dict
        +configure_optimizers() Optimizer
        +predict(user_sequences, topk, exclude_items) ndarray
        -_init_weights() void
    }

    class BERT4RecBlock {
        +MultiHeadAttention attention
        +PositionwiseFeedForward pointwise_feedforward
        +__init__(num_heads, hidden_units, dropout_rate)
        +forward(input_enc, mask) tuple
    }

    class MultiHeadAttention {
        +int num_heads
        +int hidden_units
        +int head_dim
        +Linear W_Q
        +Linear W_K
        +Linear W_V
        +Linear W_O
        +ScaledDotProductAttention attention
        +Dropout dropout
        +LayerNorm layerNorm
        +__init__(num_heads, hidden_units, dropout_rate)
        +forward(enc, mask) tuple
    }

    class ScaledDotProductAttention {
        +int head_dim
        +Dropout dropout
        +__init__(head_dim, dropout_rate)
        +forward(Q, K, V, mask) tuple
    }

    class PositionwiseFeedForward {
        +Linear W_1
        +Linear W_2
        +Dropout dropout
        +LayerNorm layerNorm
        +__init__(hidden_units, dropout_rate)
        +forward(x) Tensor
    }

    %% Data Components
    class BERT4RecDataModule {
        +str data_dir
        +str data_file
        +int batch_size
        +int max_len
        +float mask_prob
        +int min_interactions
        +int seed
        +int num_workers
        +int pad_token
        +int mask_token
        +int num_users
        +int num_items
        +dict user_train
        +dict user_valid
        +Series item2idx
        +Series user2idx
        +Series idx2item
        +Series idx2user
        +dict item_years
        +dict user_last_click_years
        +__init__(data_dir, data_file, batch_size, max_len, mask_prob, min_interactions, seed, num_workers)
        +prepare_data() void
        +setup(stage) void
        +train_dataloader() DataLoader
        +val_dataloader() DataLoader
        +get_user_sequence(user_id) list
        +get_all_sequences() dict
        +get_full_sequences() dict
        +get_future_item_sequences() dict
        -_load_item_metadata(df) void
    }

    class BERT4RecDataset {
        +dict user_sequences
        +int num_items
        +int max_len
        +float mask_prob
        +int mask_token
        +int pad_token
        +list users
        +__init__(user_sequences, num_items, max_len, mask_prob, mask_token, pad_token)
        +__len__() int
        +__getitem__(idx) tuple
        -_mask_sequence(seq) tuple
    }

    class BERT4RecValidationDataset {
        +dict user_sequences
        +dict user_targets
        +int num_items
        +int max_len
        +int mask_token
        +int pad_token
        +list users
        +__init__(user_sequences, user_targets, num_items, max_len, mask_token, pad_token)
        +__len__() int
        +__getitem__(idx) tuple
    }

    %% Relationships - Model Architecture
    BERT4Rec "1" *-- "n" BERT4RecBlock : contains
    BERT4RecBlock "1" *-- "1" MultiHeadAttention : uses
    BERT4RecBlock "1" *-- "1" PositionwiseFeedForward : uses
    MultiHeadAttention "1" *-- "1" ScaledDotProductAttention : uses

    %% Relationships - Data Pipeline
    BERT4RecDataModule "1" ..> "1" BERT4RecDataset : creates for training
    BERT4RecDataModule "1" ..> "1" BERT4RecValidationDataset : creates for validation

    %% Inheritance
    BERT4Rec --|> LightningModule : inherits
    BERT4RecDataModule --|> LightningDataModule : inherits
    BERT4RecDataset --|> Dataset : inherits
    BERT4RecValidationDataset --|> Dataset : inherits
    BERT4RecBlock --|> Module : inherits
    MultiHeadAttention --|> Module : inherits
    ScaledDotProductAttention --|> Module : inherits
    PositionwiseFeedForward --|> Module : inherits

    %% PyTorch/Lightning Base Classes
    class LightningModule {
        <<PyTorch Lightning>>
    }

    class LightningDataModule {
        <<PyTorch Lightning>>
    }

    class Module {
        <<PyTorch>>
    }

    class Dataset {
        <<PyTorch>>
    }

    %% Notes
    note for BERT4Rec "Main model implementing bidirectional\nself-attention for sequential\nrecommendation with cloze task"
    note for BERT4RecDataModule "Handles data loading, preprocessing,\nitem metadata (release years),\nand future information leakage detection"
    note for BERT4RecBlock "Transformer block with\nMulti-Head Attention + FFN"
```

## Key Design Patterns

### 1. Composition Pattern
- **BERT4Rec** composes multiple **BERT4RecBlock** modules
- Each **BERT4RecBlock** composes **MultiHeadAttention** and **PositionwiseFeedForward**
- **MultiHeadAttention** uses **ScaledDotProductAttention**

### 2. Factory Pattern
- **BERT4RecDataModule** creates appropriate dataset instances:
  - `BERT4RecDataset` for training (with masking)
  - `BERT4RecValidationDataset` for validation (with ground truth)

### 3. Strategy Pattern
- Masking strategy in `BERT4RecDataset._mask_sequence()`:
  - 80% mask token replacement
  - 10% random token replacement
  - 10% keep original

### 4. Template Method Pattern
- PyTorch Lightning's lifecycle methods:
  - `training_step()`, `validation_step()`
  - `configure_optimizers()`
  - `prepare_data()`, `setup()`

## Special Features

### Token Management
- `pad_token = 0`: Padding
- `mask_token = num_items + 1`: Masked position marker
- Item indices: `1 ~ num_items`

### Future Information Leakage Prevention
- `item_years`: Maps items to release years
- `user_last_click_years`: Tracks user's last interaction year
- `get_future_item_sequences()`: Identifies items released after user's last click

### Embedding Sharing
- Optional weight sharing between input embeddings and output layer
- Reduces parameters and improves efficiency
