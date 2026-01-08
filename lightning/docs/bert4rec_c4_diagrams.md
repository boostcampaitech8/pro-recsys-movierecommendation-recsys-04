# BERT4Rec C4 Model Diagrams

## Level 1: System Context Diagram

```mermaid
C4Context
    title System Context diagram for Movie Recommendation System

    Person(user, "ML Engineer", "Trains and deploys recommendation models")

    System_Boundary(recsys, "Movie Recommendation System") {
        System(ai_models, "AI Models", "MultiVAE, EASE, BERT4Rec, SASRec, Ensemble models")
        System(data_storage, "Data Storage", "Training data and metadata files")
        System(model_storage, "Model Storage", "Saved model checkpoints")
        System(tensorboard, "TensorBoard", "Training visualization and monitoring")
        System(data_explorer, "Data Explorer", "Exploratory data analysis tools")
    }

    System_Ext(huggingface, "HuggingFace Hub", "Pre-trained language models for embeddings")

    Rel(user, ai_models, "Trains models, generates recommendations")
    Rel(user, data_explorer, "Explores and analyzes data")
    Rel(ai_models, data_storage, "Reads training data from")
    Rel(ai_models, model_storage, "Saves/loads models to/from")
    Rel(ai_models, tensorboard, "Logs metrics to")
    Rel(user, tensorboard, "Monitors training progress")
    Rel(ai_models, huggingface, "Downloads pre-trained models from")
```

## Level 2: Container Diagram

```mermaid
C4Container
    title Container diagram for BERT4Rec System

    Person(user, "ML Engineer", "Trains and evaluates models")

    Container_Boundary(bert4rec_system, "BERT4Rec System") {
        Container(preprocess_script, "Preprocess Script", "Python", "Generates title embeddings from raw text using Sentence-BERT")
        Container(train_script, "Training Script", "Python/Hydra", "Orchestrates training pipeline with configuration management")
        Container(predict_script, "Prediction Script", "Python", "Generates recommendations for users")
        Container(data_explorer, "Data Explorer", "Jupyter/Python", "Interactive data analysis and visualization")
        Container(model, "BERT4Rec Model", "PyTorch Lightning", "Transformer-based sequential recommendation model")
        Container(data_module, "Data Module", "PyTorch Lightning", "Handles data loading, preprocessing, and batching")
        Container(utils, "Utilities", "Python", "Metrics calculation and helper functions")
    }

    System_Ext(huggingface, "HuggingFace Hub", "Model Repository", "Sentence-BERT pre-trained models")
    System_Ext(config, "Configuration", "YAML/Hydra", "Model and training hyperparameters")
    System_Ext(data_files, "Training Data", "CSV", "User-item interaction sequences")
    System_Ext(metadata_raw, "Raw Metadata", "TSV", "Item titles, genres, directors, writers, years")
    System_Ext(title_embeddings, "Title Embeddings", "TSV", "Pre-computed title embedding vectors")
    System_Ext(checkpoints, "Checkpoints", "PyTorch", "Trained model weights")
    System_Ext(tensorboard, "TensorBoard", "Visualization", "Training metrics and logs")

    Rel(user, preprocess_script, "Runs preprocessing (one-time)")
    Rel(user, train_script, "Executes training")
    Rel(user, predict_script, "Generates recommendations")
    Rel(user, data_explorer, "Analyzes data")

    Rel(preprocess_script, huggingface, "Downloads Sentence-BERT model")
    Rel(preprocess_script, metadata_raw, "Reads title text")
    Rel(preprocess_script, title_embeddings, "Writes embedding vectors")

    Rel(train_script, config, "Reads configuration")
    Rel(train_script, model, "Initializes and trains")
    Rel(train_script, data_module, "Uses for data loading")
    Rel(predict_script, model, "Loads trained model")
    Rel(predict_script, data_module, "Uses for inference data")

    Rel(data_module, data_files, "Loads interaction data")
    Rel(data_module, metadata_raw, "Loads genres, directors, writers, years")
    Rel(data_module, title_embeddings, "Loads pre-computed embeddings")
    Rel(data_explorer, data_files, "Analyzes interaction patterns")
    Rel(data_explorer, metadata_raw, "Explores metadata distributions")

    Rel(model, utils, "Uses for metrics")
    Rel(train_script, checkpoints, "Saves best models")
    Rel(predict_script, checkpoints, "Loads models from")
    Rel(train_script, tensorboard, "Logs metrics")
```

## Level 3: Component Diagram

```mermaid
C4Component
    title Component diagram for BERT4Rec Model Container

    Container_Boundary(model_container, "BERT4Rec Model") {
        Component(bert4rec, "BERT4Rec Module", "LightningModule", "Main model orchestrating training/validation")
        Component(transformer_block, "Transformer Blocks", "nn.Module", "Stacked transformer layers with self-attention")
        Component(attention, "Multi-Head Attention", "nn.Module", "Bidirectional self-attention mechanism")
        Component(ffn, "Feed-Forward Network", "nn.Module", "Position-wise feed-forward with GELU")
        Component(embeddings, "Embeddings", "nn.Embedding", "Item and positional embeddings")
        Component(metadata_emb, "Metadata Embeddings", "nn.Embedding", "Genre, director, writer, title embeddings")
        Component(fusion, "Metadata Fusion", "nn.Module", "Concat/Add/Gate fusion for metadata")
        Component(output_layer, "Output Layer", "Linear/Embedding", "Prediction layer for next items")
    }

    Container_Boundary(data_container, "Data Module") {
        Component(data_module, "BERT4RecDataModule", "LightningDataModule", "Orchestrates data pipeline")
        Component(train_dataset, "Training Dataset", "Dataset", "Applies BERT-style masking")
        Component(val_dataset, "Validation Dataset", "Dataset", "Sequences with [MASK] at end")
        Component(data_loader, "DataLoader", "PyTorch", "Batches and loads data")
        Component(metadata_loader, "Metadata Loader", "Python", "Loads genre, director, writer, title metadata")
    }

    Container_Boundary(utils_container, "Utilities") {
        Component(metrics, "Metrics", "Python", "Hit@K, NDCG@K calculation")
        Component(paths, "Path Utils", "Python", "Directory management")
        Component(recommend, "Recommender", "Python", "Inference utilities")
    }

    Rel(bert4rec, transformer_block, "Uses N layers")
    Rel(transformer_block, attention, "Applies attention")
    Rel(transformer_block, ffn, "Applies FFN")
    Rel(bert4rec, embeddings, "Embeds items and positions")
    Rel(bert4rec, metadata_emb, "Embeds metadata features")
    Rel(bert4rec, fusion, "Fuses item + metadata embeddings")
    Rel(fusion, transformer_block, "Feeds fused embeddings")
    Rel(bert4rec, output_layer, "Projects to item space")

    Rel(data_module, train_dataset, "Creates for training")
    Rel(data_module, val_dataset, "Creates for validation")
    Rel(data_module, data_loader, "Wraps datasets")
    Rel(data_module, metadata_loader, "Loads metadata files")

    Rel(bert4rec, metrics, "Calculates Hit@K, NDCG@K")
    Rel(bert4rec, data_loader, "Receives batches from")
```

## Level 4: Code Diagram - BERT4Rec Architecture

```mermaid
classDiagram
    class BERT4Rec {
        +int num_items
        +int hidden_units
        +int num_heads
        +int num_layers
        +int max_len
        +float dropout_rate
        +float mask_prob
        +Embedding item_emb
        +Embedding pos_emb
        +ModuleList blocks
        +bool use_genre_emb
        +bool use_director_emb
        +bool use_writer_emb
        +bool use_title_emb
        +str metadata_fusion
        +Embedding genre_emb
        +Embedding director_emb
        +Embedding writer_emb
        +Linear fusion_layer
        +Linear fusion_gate
        +forward(log_seqs, metadata, return_gate_values)
        +training_step(batch, batch_idx)
        +validation_step(batch, batch_idx)
        +on_validation_epoch_end()
        +predict(user_sequences, topk)
        +configure_optimizers()
    }

    class BERT4RecBlock {
        +MultiHeadAttention attention
        +PositionwiseFeedForward ffn
        +forward(input_enc, mask)
    }

    class MultiHeadAttention {
        +int num_heads
        +int head_dim
        +Linear W_Q, W_K, W_V, W_O
        +ScaledDotProductAttention attention
        +Dropout dropout
        +LayerNorm layerNorm
        +forward(enc, mask)
    }

    class ScaledDotProductAttention {
        +int head_dim
        +Dropout dropout
        +forward(Q, K, V, mask)
    }

    class PositionwiseFeedForward {
        +Linear W_1, W_2
        +Dropout dropout
        +LayerNorm layerNorm
        +forward(x)
    }

    class BERT4RecDataModule {
        +str data_dir
        +int batch_size
        +int max_len
        +Dict user_train
        +Dict user_valid
        +Dict item_genres
        +Dict item_directors
        +Dict item_writers
        +Dict item_title_embs
        +int num_genres
        +int num_directors
        +int num_writers
        +int title_embedding_dim
        +prepare_data()
        +setup(stage)
        +train_dataloader()
        +val_dataloader()
        +load_metadata()
    }

    class BERT4RecDataset {
        +Dict user_sequences
        +int max_len
        +float mask_prob
        +__getitem__(idx)
        -_mask_sequence(seq)
    }

    class BERT4RecValidationDataset {
        +Dict user_sequences
        +Dict user_targets
        +int max_len
        +__getitem__(idx)
    }

    BERT4Rec "1" *-- "N" BERT4RecBlock : contains
    BERT4RecBlock "1" *-- "1" MultiHeadAttention : uses
    BERT4RecBlock "1" *-- "1" PositionwiseFeedForward : uses
    MultiHeadAttention "1" *-- "1" ScaledDotProductAttention : uses
    BERT4RecDataModule "1" --> "1" BERT4RecDataset : creates
    BERT4RecDataModule "1" --> "1" BERT4RecValidationDataset : creates
```

## Training Sequence Diagram

```mermaid
sequenceDiagram
    actor User as ML Engineer
    participant Script as train_bert4rec.py
    participant Hydra as Hydra Config
    participant DM as DataModule
    participant Model as BERT4Rec
    participant Trainer as Lightning Trainer
    participant Logger as TensorBoard

    User->>Script: Execute training
    Script->>Hydra: Load configuration
    Hydra-->>Script: Config object

    Script->>DM: Initialize with config
    DM->>DM: Load CSV data
    DM->>DM: Create user sequences
    DM->>DM: Train/val split
    DM-->>Script: num_items, num_users

    Script->>Model: Initialize(num_items, config)
    Model->>Model: Build transformer layers
    Model->>Model: Initialize embeddings

    Script->>Trainer: Initialize with callbacks
    Script->>Trainer: fit(model, datamodule)

    loop Each Epoch
        Trainer->>DM: Get train batch
        DM-->>Trainer: (masked_seqs, labels)
        Trainer->>Model: training_step(batch)
        Model->>Model: forward(masked_seqs)
        Model->>Model: compute_loss(logits, labels)
        Model-->>Trainer: loss
        Trainer->>Logger: Log train_loss

        Trainer->>DM: Get val batch
        DM-->>Trainer: (seqs_with_mask, target_items)
        Trainer->>Model: validation_step(batch)
        Model->>Model: forward(seqs)
        Model->>Model: compute_metrics(predictions, targets)
        Model-->>Trainer: metrics (Hit@10, NDCG@10)
        Trainer->>Logger: Log validation metrics
    end

    Trainer->>Trainer: Save best checkpoint
    Trainer-->>Script: Training complete
    Script-->>User: Best model path
```

## Inference Sequence Diagram

```mermaid
sequenceDiagram
    actor User as ML Engineer
    participant Script as predict_bert4rec.py
    participant DM as DataModule
    participant Model as BERT4Rec
    participant Recommend as Recommender

    User->>Script: Execute prediction
    Script->>DM: Load data and setup
    DM->>DM: Load all user sequences
    DM-->>Script: user_sequences, mappings

    Script->>Model: Load from checkpoint
    Model-->>Script: Trained model

    Script->>Model: Set to eval mode

    loop For each user
        Script->>Recommend: Generate recommendations
        Recommend->>Model: predict(user_seq, topk=10)
        Model->>Model: Prepare sequence + [MASK]
        Model->>Model: forward(masked_seq)
        Model->>Model: Get logits for last position
        Model->>Model: Mask invalid items
        Model->>Model: Get top-K items
        Model-->>Recommend: top_k_items
        Recommend-->>Script: Recommendations
    end

    Script->>Script: Convert indices to original IDs
    Script->>Script: Save to CSV
    Script-->>User: Recommendations file
```

## Data Flow Diagram

```mermaid
graph TB
    subgraph External["External Resources"]
        HF[HuggingFace Hub<br/>mixedbread-ai/<br/>mxbai-embed-large-v1]
    end

    subgraph Input["Input Data"]
        A[train_ratings.csv]
        B[years.tsv]
        C[bert4rec_v2.yaml]
        D[genres.tsv]
        E[directors.tsv]
        F[writers.tsv]
        G0[titles_raw.tsv<br/>item, title text]
    end

    subgraph Preprocessing["Preprocessing (One-time)"]
        PRE[preprocess_title_embeddings.py]
        SBERT[Sentence-BERT Model<br/>mxbai-embed-large-v1<br/>1024-dim]
        ENCODE[Encode Titles<br/>batch processing]
    end

    subgraph PreprocessOutput["Preprocessing Output"]
        G[title_embeddings.pkl<br/>dict: item_id → 1024-dim vector]
        G2[metadata.pkl<br/>model info, embedding_dim, num_items]
    end

    subgraph Data Processing
        H[Load & Parse CSV]
        I[Create Mappings<br/>item2idx, user2idx]
        J[Build User Sequences]
        K[Load Metadata<br/>Genre/Director/Writer/Title]
        L[Train/Val Split]
        M[BERT Masking]
    end

    subgraph Model Training
        N[Item Embeddings]
        O[Positional Embeddings]
        P[Metadata Embeddings<br/>Genre/Director/Writer/Title]
        Q[Metadata Fusion<br/>Concat/Add/Gate]
        R[Transformer Blocks]
        S[Multi-Head Attention]
        T[Feed-Forward Network]
        U[Output Projection]
    end

    subgraph Output
        V[Loss Computation]
        W[Metrics: Hit@10, NDCG@10]
        X[Gate Values<br/>Feature Importance]
        Y[Model Checkpoints]
        Z[TensorBoard Logs<br/>Metrics + Gate Values + HParams]
    end

    HF -->|Download model| PRE
    G0 --> PRE
    PRE --> SBERT
    SBERT --> ENCODE
    ENCODE --> G
    ENCODE --> G2

    A --> H
    B --> I
    C --> H
    D --> K
    E --> K
    F --> K
    G --> K
    G2 --> K
    H --> I
    I --> J
    J --> L
    K --> P
    L --> M

    M --> N
    M --> O
    M --> P
    N --> Q
    P --> Q
    Q --> R
    O --> R
    R --> S
    S --> T
    T --> U

    U --> V
    U --> W
    Q --> X
    V --> Y
    W --> Z
    X --> Z

    style HF fill:#fff3cd
    style A fill:#e1f5ff
    style B fill:#e1f5ff
    style C fill:#e1f5ff
    style D fill:#e1f5ff
    style E fill:#e1f5ff
    style F fill:#e1f5ff
    style G0 fill:#e1f5ff
    style G fill:#d4edda
    style PRE fill:#d4edda
    style SBERT fill:#d4edda
    style ENCODE fill:#d4edda
    style Y fill:#ffe1e1
    style Z fill:#ffe1e1
```

## Deployment Diagram

```mermaid
C4Deployment
    title Deployment diagram for BERT4Rec System

    Deployment_Node(dev_machine, "Development Machine", "Linux/GPU Server") {
        Deployment_Node(python_env, "Python Environment", "Python 3.10 + venv") {
            Container(training, "Training Process", "PyTorch Lightning", "Trains BERT4Rec models")
            Container(inference, "Inference Process", "PyTorch", "Generates recommendations")
        }

        Deployment_Node(storage, "Local Storage", "File System") {
            ContainerDb(data, "Training Data", "CSV/TSV", "User-item interactions")
            ContainerDb(checkpoints, "Model Checkpoints", "PyTorch", "Trained weights")
            ContainerDb(logs, "Logs", "TensorBoard", "Training metrics")
        }
    }

    Rel(training, data, "Reads from")
    Rel(training, checkpoints, "Writes to")
    Rel(training, logs, "Writes to")
    Rel(inference, data, "Reads from")
    Rel(inference, checkpoints, "Loads from")
```

---

## Metadata Fusion Flow Diagram

```mermaid
graph TB
    subgraph Input["Input Sequence (Batch)"]
        SEQ[Item IDs: i₁, i₂, i₃, ..., iₙ]
        META["Metadata:<br/>- Genres: g₁, g₂, ...<br/>- Directors: d₁, d₂, ...<br/>- Writers: w₁, w₂, ...<br/>- Titles: t₁, t₂, ..."]
    end

    subgraph Embedding["Embedding Layer"]
        ITEM_EMB[Item Embedding<br/>lookup item_emb]
        GENRE_EMB[Genre Embedding<br/>avg pool]
        DIR_EMB[Director Embedding<br/>lookup]
        WRITER_EMB[Writer Embedding<br/>avg pool]
        TITLE_EMB[Title Embedding<br/>pre-computed 1024-dim]
    end

    subgraph Fusion["Metadata Fusion"]
        CONCAT{Fusion Strategy}
        CONCAT_OP["Concat Fusion<br/>concat all → Linear"]
        ADD_OP["Add Fusion<br/>element-wise sum"]
        GATE_OP["Gate Fusion<br/>softmax weighted"]
        GATE_LOG["Gate Logging<br/>TensorBoard"]
    end

    subgraph Transformer["Transformer Processing"]
        POS[+ Position Embedding]
        TRANS[Transformer Blocks<br/>N layers]
    end

    subgraph Output["Output"]
        LOGITS[Item Logits<br/>batch × seq_len × num_items]
        METRICS["Metrics:<br/>Hit@10, NDCG@10"]
        GATE_METRICS["Gate Metrics:<br/>val_gate/item<br/>val_gate/title"]
    end

    SEQ --> ITEM_EMB
    META --> GENRE_EMB
    META --> DIR_EMB
    META --> WRITER_EMB
    META --> TITLE_EMB

    ITEM_EMB --> CONCAT
    GENRE_EMB --> CONCAT
    DIR_EMB --> CONCAT
    WRITER_EMB --> CONCAT
    TITLE_EMB --> CONCAT

    CONCAT -->|concat| CONCAT_OP
    CONCAT -->|add| ADD_OP
    CONCAT -->|gate| GATE_OP

    CONCAT_OP --> POS
    ADD_OP --> POS
    GATE_OP --> POS
    GATE_OP --> GATE_LOG

    POS --> TRANS
    TRANS --> LOGITS
    LOGITS --> METRICS
    GATE_LOG --> GATE_METRICS

    style GATE_OP fill:#90EE90
    style GATE_LOG fill:#FFD700
    style GATE_METRICS fill:#FFD700
```

**Gate Fusion 상세 동작**:
```
1. Collect embeddings: [item_emb, genre_emb, director_emb, writer_emb, title_emb]
2. Concatenate: concat_emb = [e₁; e₂; e₃; e₄; e₅]  # [batch, seq_len, hidden*5]
3. Compute gates: gates = softmax(Linear(concat_emb))  # [batch, seq_len, 5]
4. Weight embeddings: weighted = Σ (gates[i] · embeddings[i])
5. Log gate values: val_gate/{feature_name} → TensorBoard
6. Output: fused_emb [batch, seq_len, hidden_dim]
```

---

## Key Architecture Decisions

### 1. Bidirectional Attention
- Unlike SASRec (unidirectional), BERT4Rec uses bidirectional self-attention
- Allows the model to capture context from both past and future items in the sequence
- Implemented through full attention mask (only padding is masked)

### 2. BERT-Style Training
- **Cloze Task**: Randomly mask 15% of items in sequences
  - 80%: Replace with [MASK] token
  - 10%: Replace with random item
  - 10%: Keep original
- Predicts masked items using bidirectional context

### 3. Embedding Sharing
- Output layer shares weights with item embedding (transpose)
- Reduces parameters and improves efficiency
- Configurable via `share_embeddings` parameter

### 4. Transformer Architecture
- **Layers**: 2 transformer blocks (configurable)
- **Hidden Units**: 256 dimensions
- **Attention Heads**: 4 heads
- **Feed-Forward**: 4x expansion (hidden → 1024 → hidden)
- **Activation**: GELU (more stable than ReLU for transformers)

### 5. Training Strategy
- **Optimizer**: Adam with learning rate 0.001
- **Max Sequence Length**: 200 items
- **Dropout**: 0.2 for regularization
- **Gradient Clipping**: 5.0 to prevent exploding gradients
- **Early Stopping**: Monitor validation NDCG@10 with patience 20

### 6. Validation Strategy
- Last item in sequence held out for validation
- Metrics: Hit@10, NDCG@10
- No negative sampling during evaluation (rank against all items)

### 7. Configuration Management
- Hydra framework for flexible configuration
- Easy hyperparameter tuning via command line
- Automatic experiment tracking and logging

### 8. Metadata Fusion Architecture
- **Multiple Metadata Types**: Support for genre, director, writer, and title embeddings
- **Flexible Fusion Strategies**:
  - **Concatenation**: Simple concat + linear projection
  - **Addition**: Element-wise sum (parameter-free)
  - **Gate Fusion**: Learnable weighted combination with softmax normalization
- **Gate Monitoring**: TensorBoard logging for real-time feature importance tracking
- **Modular Design**: Each metadata type can be independently enabled/disabled
- **Content + Collaborative**: Combines content-based (metadata) and collaborative filtering (item embeddings)

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **Deep Learning Framework** | PyTorch 2.x |
| **Training Framework** | PyTorch Lightning |
| **Configuration** | Hydra + OmegaConf |
| **Logging** | TensorBoard |
| **Data Processing** | Pandas + NumPy |
| **Model Architecture** | Transformer (BERT-style) |
| **Optimizer** | Adam |

---

## File Structure

```
lightning/
├── configs/
│   └── bert4rec_v2.yaml          # Model & training configuration
├── src/
│   ├── models/
│   │   └── bert4rec.py           # BERT4Rec model implementation
│   ├── data/
│   │   └── bert4rec_data.py      # Data loading & preprocessing
│   └── utils/
│       ├── metrics.py            # Evaluation metrics
│       ├── path_utils.py         # Directory management
│       └── recommend.py          # Inference utilities
├── train_bert4rec.py             # Training script
└── predict_bert4rec.py           # Inference script
```
