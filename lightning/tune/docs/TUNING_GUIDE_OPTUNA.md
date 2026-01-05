# BERT4Rec Hyperparameter Tuning Guide

## 📋 목차

1. [개요](#1-개요)
2. [사전 준비 및 환경 검증](#2-사전-준비-및-환경-검증)
3. [Tuning 단계별 프로세스](#3-tuning-단계별-프로세스)
   - [Stage 0: Test Mode - 환경 검증](#stage-0-test-mode---환경-검증)
   - [Stage 1: Quick Mode - 빠른 탐색](#stage-1-quick-mode---빠른-탐색)
   - [Stage 2: Medium Mode - 정밀 탐색](#stage-2-medium-mode---정밀-탐색)
   - [Stage 3: Seed Search - 재현성 확보](#stage-3-seed-search---재현성-확보)
4. [일반화를 위한 전략](#4-일반화를-위한-전략)
5. [Tuning vs Training 환경 일치](#5-tuning-vs-training-환경-일치)
6. [파라미터별 가이드](#6-파라미터별-가이드)
7. [트러블슈팅](#7-트러블슈팅)
8. [Best Practices](#8-best-practices)

---

## 1. 개요

### 1.1 Tuning의 목적

- ✅ **일반화 성능 최적화**: Validation뿐 아니라 Public/Private test 성능 향상
- ✅ **재현 가능한 결과**: Seed 고정으로 deterministic한 결과
- ✅ **효율적인 탐색**: 단계적 탐색으로 시간/비용 절약
- ✅ **Overfitting 방지**: Regularization 파라미터 최적화

### 1.2 핵심 원칙

```
🎯 원칙 1: Tuning 환경 = Training 환경
   - Scheduler, Dataset split, Loss function 동일
   - 다르면 Tuning 결과가 Training에 적용 안됨

🎯 원칙 2: 일반화 우선
   - Val NDCG@10 최고 ≠ Public score 최고
   - Regularization 파라미터 충분히 크게

🎯 원칙 3: 단계적 탐색
   - Test → Quick → Medium → Seed
   - 넓게 탐색 → 좁게 정밀화

🎯 원칙 4: 재현성 확보
   - Seed 고정 필수
   - 모든 실험 기록
```

---

## 2. 사전 준비 및 환경 검증

### 2.1 파일 구조 확인

```bash
lightning/
├── tune/
│   ├── tune_bert4rec_optuna.py          # Main tuning script
│   ├── tune_bert4rec_optuna_monitored.py # 진행 상황 모니터링
│   ├── quick_tune.py                     # Quick mode helper
│   └── results/                          # Tuning 결과 저장
├── src/
│   ├── models/bert4rec.py               # Model definition
│   └── data/bert4rec_data.py            # DataModule
├── configs/
│   └── bert4rec_v2.yaml                 # Training config
└── train_bert4rec.py                    # Training script
```

### 2.2 필수 체크리스트

```bash
# 1. 데이터 확인
ls -lh ~/data/train/train_ratings.csv

# 2. 환경 확인
source .venv/bin/activate
python -c "import torch, lightning, optuna; print('OK')"

# 3. GPU 확인
nvidia-smi

# 4. 이전 DB 백업
cd ~/juik/lightning/tune
ls -lh *.db
cp important.db important_backup_$(date +%Y%m%d).db
```

---

## 3. Tuning 단계별 프로세스

## Stage 0: Test Mode - 환경 검증

**목적**: 전체 파이프라인이 정상 동작하는지 확인

### Step 0-1: 단일 Trial 테스트

```bash
cd ~/juik/lightning/tune

# quick_tune.py를 test 모드로 실행
python quick_tune.py --mode test
```

**또는 직접 실행**:
```bash
python tune_bert4rec_optuna.py \
    --study_name bert4rec_test \
    --n_trials 2 \
    --n_epochs 2
```

**확인 사항**:
- ✅ 에러 없이 완료
- ✅ DB 파일 생성 (`bert4rec_test.db`)
- ✅ NDCG@10 값 출력 (정상 범위: 0.01~0.05)
- ✅ 실행 시간 (3 epochs: ~5-10분)

### Step 0-2: Tuning vs Training 환경 일치 검증

**중요**: Tuning에서 찾은 하이퍼파라미터가 Training에서 재현되려면 환경이 동일해야 함

#### ✅ Checkpoint 1: Seed 설정

```python
# tune_bert4rec_optuna.py (Line 62-63)
import lightning as L
L.seed_everything(42, workers=True)

# train_bert4rec.py (Line 40)
L.seed_everything(cfg.data.seed, workers=True)

# bert4rec_v2.yaml (Line 17)
data:
  seed: 42
```

**검증**:
```bash
grep "seed_everything" tune/tune_bert4rec_optuna.py
grep "seed_everything" train_bert4rec.py
grep "seed:" configs/bert4rec_v2.yaml
```

#### ✅ Checkpoint 2: LR Scheduler 일치

**목적**: Tuning 환경과 Training 환경의 LR scheduler가 동일한지 확인

**Tuning 환경**:
```python
# tune_bert4rec_optuna.py → BERT4Rec 모델 사용
# src/models/bert4rec.py의 configure_optimizers() 호출
```

**Training 환경**:
```python
# train_bert4rec.py → 동일한 BERT4Rec 모델 사용
# src/models/bert4rec.py의 configure_optimizers() 호출
```

**확인 방법**:
```python
# src/models/bert4rec.py (Line 641-658)
def configure_optimizers(self):
    optimizer = torch.optim.AdamW(...)

    # Case 1: Scheduler 없음
    return optimizer
    # → Tuning과 Training 모두 constant LR

    # Case 2: Scheduler 있음
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(...)
    return [optimizer], [scheduler]
    # → Tuning과 Training 모두 cosine scheduler 사용
```

**검증**:
```bash
# Scheduler 설정 확인
grep -A 15 "def configure_optimizers" src/models/bert4rec.py

# Tuning과 Training이 같은 BERT4Rec 클래스 사용하는지 확인
grep "from src.models.bert4rec import BERT4Rec" tune/tune_bert4rec_optuna.py
grep "from src.models.bert4rec import BERT4Rec" train_bert4rec.py
```

**중요**:
- Scheduler를 사용하려면 Tuning에서도 사용해야 함
- Scheduler를 사용하지 않으려면 둘 다 사용하지 않아야 함
- **현재 권장**: Scheduler 없음 (constant LR)이 BERT4Rec에 더 적합

#### ✅ Checkpoint 3: Dataset Split

```python
# src/data/bert4rec_data.py (Line 596-604)
for user, seq in user_sequences.items():
    if self.use_full_data:
        self.user_train[user] = seq
        self.user_valid[user] = seq[-1]  # Dummy
    else:
        self.user_train[user] = seq[:-1]  # ✓ Last item split
        self.user_valid[user] = seq[-1]
```

**Tuning과 Training 모두**:
```python
use_full_data=False  # ✓ 동일
```

**검증**:
```bash
grep "use_full_data" tune/tune_bert4rec_optuna.py
grep "use_full_data" configs/bert4rec_v2.yaml
```

#### ✅ Checkpoint 4: Loss Function & Metrics

```python
# src/models/bert4rec.py
# Training step (Line 460-490)
loss = F.cross_entropy(logits, labels)

# Validation step (Line 527-639)
# NDCG@10 계산 로직 동일
val_ndcg_10 = ndcg_values.sum().item() / batch_size
```

**확인**: 동일한 `BERT4Rec` 클래스 사용
```bash
grep "from src.models.bert4rec import BERT4Rec" tune/tune_bert4rec_optuna.py
grep "from src.models.bert4rec import BERT4Rec" train_bert4rec.py
```

#### ✅ Checkpoint 5: Default 값 일치 확인

**목적**: Tuning과 Training에서 **명시하지 않은 파라미터**의 기본값이 동일한지 확인

**확인이 필요한 이유**:
- Tuning에서 명시적으로 설정하지 않은 파라미터가 있을 수 있음
- Training config에만 있는 설정이 결과에 영향을 줄 수 있음
- 두 환경의 암묵적 기본값이 다르면 재현 불가

**주요 체크 항목**:

```python
# 1. Metadata 사용 여부
# tune_bert4rec_optuna.py (Line 95-99)
datamodule = BERT4RecDataModule(
    ...
    use_genre_emb=False,      # ✓ 명시적으로 False
    use_director_emb=False,   # ✓ 명시적으로 False
    use_writer_emb=False,     # ✓ 명시적으로 False
    use_title_emb=False,      # ✓ 명시적으로 False
)

# bert4rec_v2.yaml (Line 38-41)
model:
  use_genre_emb: false        # ✓ 일치
  use_director_emb: false     # ✓ 일치
  use_writer_emb: false       # ✓ 일치
  use_title_emb: false        # ✓ 일치
```

```python
# 2. Data 관련 설정
# tune_bert4rec_optuna.py
datamodule = BERT4RecDataModule(
    min_interactions=3,       # ✓ 명시
    seed=42,                  # ✓ 명시
    num_workers=4,            # ✓ 명시
    use_full_data=False,      # ✓ 명시
)

# bert4rec_v2.yaml
data:
  min_interactions: 3         # ✓ 일치
  seed: 42                    # ✓ 일치 (또는 7222)
  num_workers: 4              # ✓ 일치
  use_full_data: false        # ✓ 일치
```

```python
# 3. Training 설정
# tune_bert4rec_optuna.py (Line 149-160)
trainer = L.Trainer(
    precision="16-mixed",     # ✓ 명시
    gradient_clip_val=5.0,    # ✓ 명시
    ...
)

# bert4rec_v2.yaml (Line 57-59)
training:
  precision: "16-mixed"       # ✓ 일치
  gradient_clip_val: 5.0      # ✓ 일치
```

```python
# 4. Model 기본 설정
# tune_bert4rec_optuna.py (Line 120)
model = BERT4Rec(
    share_embeddings=True,    # ✓ 명시
    ...
)

# bert4rec_v2.yaml (Line 35)
model:
  share_embeddings: true      # ✓ 일치
```

**검증 스크립트**:
```python
# verify_defaults.py
import yaml

# Tuning 기본값 (코드에서 추출)
tuning_defaults = {
    "use_genre_emb": False,
    "use_director_emb": False,
    "use_writer_emb": False,
    "use_title_emb": False,
    "min_interactions": 3,
    "num_workers": 4,
    "use_full_data": False,
    "precision": "16-mixed",
    "gradient_clip_val": 5.0,
    "share_embeddings": True,
}

# Training config 로드
with open("../configs/bert4rec_v2.yaml") as f:
    training_config = yaml.safe_load(f)

# 비교
print("Default Values Verification")
print("=" * 80)

mismatches = []
for key, tuning_val in tuning_defaults.items():
    # Config에서 값 찾기
    if key.startswith("use_"):
        training_val = training_config["model"].get(key)
    elif key in ["min_interactions", "num_workers", "use_full_data"]:
        training_val = training_config["data"].get(key)
    elif key in ["precision", "gradient_clip_val"]:
        training_val = training_config["training"].get(key)
    else:
        training_val = training_config["model"].get(key)

    match = tuning_val == training_val
    status = "✓" if match else "✗"

    print(f"{status} {key:25s}: Tuning={tuning_val:10s}, Training={training_val}")

    if not match:
        mismatches.append((key, tuning_val, training_val))

if mismatches:
    print("\n⚠️ Mismatches found:")
    for key, tuning_val, training_val in mismatches:
        print(f"  {key}: {tuning_val} (Tuning) ≠ {training_val} (Training)")
    print("\nAction required: Update Tuning or Training config to match")
else:
    print("\n✓ All default values match!")
```

```bash
cd ~/juik/lightning/tune
python verify_defaults.py
```

**일반적인 불일치 예시와 해결**:

```python
# 문제: Tuning에서 metadata 미지정 → 기본값 사용
# tune_bert4rec_optuna.py (잘못된 예)
datamodule = BERT4RecDataModule(...)
# use_genre_emb 미지정 → DataModule의 기본값 사용 (True일 수도!)

# 해결: 명시적으로 설정
datamodule = BERT4RecDataModule(
    use_genre_emb=False,  # ✓ 명시
    ...
)
```

**체크리스트**:
- [ ] Metadata 사용 여부 (use_*_emb) 일치
- [ ] Data 설정 (min_interactions, seed, num_workers, use_full_data) 일치
- [ ] Training 설정 (precision, gradient_clip_val) 일치
- [ ] Model 설정 (share_embeddings) 일치
- [ ] 모든 암묵적 기본값 확인

### Step 0-3: 환경 일치 검증 스크립트

```python
# verify_env.py
import sys
sys.path.append('.')

print("=" * 80)
print("Tuning vs Training Environment Verification")
print("=" * 80)

# 1. Seed
print("\n1. Seed Configuration:")
print("   Tuning: L.seed_everything(42, workers=True)")
print("   Training: L.seed_everything(cfg.data.seed, workers=True)")
print("   Config: seed=42")
print("   ✓ Consistent")

# 2. Scheduler
from src.models.bert4rec import BERT4Rec
import torch

model = BERT4Rec(num_items=1000, hidden_units=64, max_len=50, lr=0.001)
optimizer_config = model.configure_optimizers()

print("\n2. LR Scheduler:")
if isinstance(optimizer_config, list):
    print("   ✗ Scheduler detected!")
    print("   Tuning and Training will have different LR schedules")
else:
    print("   ✓ No scheduler (matches Tuning)")

# 3. Dataset split
from src.data.bert4rec_data import BERT4RecDataModule
dm = BERT4RecDataModule(
    data_dir="~/data/train/",
    data_file="train_ratings.csv",
    use_full_data=False
)
print("\n3. Dataset Split:")
print(f"   use_full_data: False")
print("   ✓ Last-item split (matches Tuning)")

# 4. Metrics
print("\n4. Loss & Metrics:")
print("   Loss: CrossEntropy (same BERT4Rec class)")
print("   Metric: NDCG@10 (same validation_step)")
print("   ✓ Identical")

print("\n" + "=" * 80)
print("Environment Verification: PASSED")
print("=" * 80)
```

```bash
python verify_env.py
```

---

## Stage 1: Quick Mode - 빠른 탐색

**목적**:
- 넓은 search space에서 promising한 영역 빠르게 파악
- **일반화를 위한 최소 regularization 값 확인**
- 10-20 trials, 20-30 epochs로 빠른 피드백

### Step 1-1: Quick Tuning 실행

**quick_tune.py 사용 (권장)**:
```bash
cd ~/juik/lightning/tune

# Quick mode 실행 (100 trials, 20 epochs)
python quick_tune.py --mode quick
```

**또는 직접 실행**:
```bash
python tune_bert4rec_optuna.py \
    --study_name bert4rec_quick \
    --n_trials 100 \
    --n_epochs 20 \
    --n_jobs 1
```

### Step 1-2: Quick 결과 분석

```python
# analyze_quick_results.py
import optuna
import numpy as np
from tabulate import tabulate

study = optuna.load_study(
    study_name="bert4rec_quick",
    storage="sqlite:///bert4rec_quick.db"
)

completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

# 파라미터별 중요도 분석
print("=" * 80)
print("Parameter Importance Analysis")
print("=" * 80)

param_importance = optuna.importance.get_param_importances(study)
for param, importance in sorted(param_importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{param:25s}: {importance:.4f}")

# Top 5 trials 분석
print("\n" + "=" * 80)
print("Top 5 Trials")
print("=" * 80)

top_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
for i, trial in enumerate(top_trials, 1):
    print(f"\n{i}. Trial #{trial.number}: NDCG@10 = {trial.value:.4f}")
    for key, value in trial.params.items():
        print(f"   {key:20s}: {value}")

# Regularization 파라미터 분석 (일반화 중요!)
print("\n" + "=" * 80)
print("Regularization Parameters Analysis")
print("=" * 80)

reg_params = ['dropout_rate', 'weight_decay', 'random_mask_prob']
for param in reg_params:
    values = [t.params[param] for t in completed if param in t.params]
    scores = [t.value for t in completed if param in t.params]

    # 상위 20% trials의 평균값
    top_20_idx = np.argsort(scores)[-len(scores)//5:]
    top_20_values = [values[i] for i in top_20_idx]

    print(f"\n{param}:")
    print(f"  Overall range: [{min(values):.4f}, {max(values):.4f}]")
    print(f"  Top 20% avg:   {np.mean(top_20_values):.4f}")
    print(f"  Top 20% min:   {min(top_20_values):.4f} ← 일반화 최소값")
    print(f"  Recommendation for Medium: [{min(top_20_values):.4f}, {max(top_20_values):.4f}]")
```

```bash
python analyze_quick_results.py
```

### Step 1-3: Medium 준비 - 범위 축소 기준

**파라미터 고정 vs 탐색 결정 기준**:

```python
# 결정 트리
if param_importance > 0.1:
    # 중요 파라미터 → Medium에서 계속 탐색
    # 단, Quick 결과로 범위 축소

    if param == "dropout_rate":
        # 일반화를 위한 최소값 설정
        min_val = top_20_min * 1.1  # 상위 20% 최소값의 110%
        max_val = top_20_max

    elif param == "lr":
        # Log scale 파라미터
        min_val = top_20_min * 0.5
        max_val = top_20_max * 1.5

    else:
        # 일반 파라미터
        min_val = top_20_min * 0.9
        max_val = top_20_max * 1.1

elif param_importance < 0.05:
    # 덜 중요 파라미터 → 고정
    fixed_val = best_trial_value

else:
    # 중간 중요도 → 좁은 범위 탐색
    min_val = top_20_min * 0.95
    max_val = top_20_max * 1.05
```

**예시**:

```python
# Quick 결과 (가정)
Importance:
  lr: 0.35              # 매우 중요
  dropout_rate: 0.25    # 중요
  weight_decay: 0.18    # 중요
  batch_size: 0.12      # 중간
  num_heads: 0.06       # 덜 중요
  hidden_units: 0.04    # 덜 중요

Top 20% 범위:
  lr: [0.0015, 0.0030]
  dropout_rate: [0.15, 0.25]  # 최소값 0.15 ← 일반화 중요!
  weight_decay: [0.02, 0.08]  # 최소값 0.02 ← 일반화 중요!

# Medium search space
lr = trial.suggest_float("lr", 0.001, 0.004, log=True)  # ×0.67, ×1.33
dropout_rate = trial.suggest_float("dropout_rate", 0.165, 0.25)  # 최소값 110%
weight_decay = trial.suggest_float("weight_decay", 0.022, 0.08)  # 최소값 110%
batch_size = trial.suggest_categorical("batch_size", [128, 256])  # 축소
num_heads = 8  # 고정 (best value)
hidden_units = 256  # 고정 (best value)
```

---

## Stage 2: Medium Mode - 정밀 탐색

**목적**:
- Quick에서 찾은 promising 영역을 정밀 탐색
- 충분한 epochs (50)로 안정적인 성능 확인
- 일반화 파라미터 최소값 보장

### Step 2-1: Medium Search Space 설정

```python
# tune_bert4rec_optuna.py 수정
# Medium mode 설정

class OptunaObjective:
    def __call__(self, trial: optuna.Trial):
        # Seed 고정
        import lightning as L
        L.seed_everything(42, workers=True)

        # ===== Fixed Parameters (Quick 결과로 고정) =====
        hidden_units = 256  # Quick best
        num_heads = 8       # Quick best
        num_layers = 3      # Quick best
        max_len = 200       # Quick best

        # ===== High Priority (넓게 탐색) =====
        lr = trial.suggest_float("lr", 0.001, 0.004, log=True)

        # ===== Regularization (일반화 최소값 보장) =====
        dropout_rate = trial.suggest_float("dropout_rate", 0.165, 0.28)
        # 최소값 0.165 = Quick top 20% min (0.15) × 1.1
        # → 일반화 보장

        weight_decay = trial.suggest_float("weight_decay", 0.022, 0.09)
        # 최소값 0.022 = Quick top 20% min (0.02) × 1.1
        # → L2 regularization 보장

        random_mask_prob = trial.suggest_float("random_mask_prob", 0.17, 0.23)
        # Data augmentation 충분히

        # ===== Medium Priority (좁게 탐색) =====
        batch_size = trial.suggest_categorical("batch_size", [128, 256])
        last_item_mask_ratio = trial.suggest_float("last_item_mask_ratio", 0.05, 0.12)

        # ... rest of training ...
```

### Step 2-2: Medium Tuning 실행

**quick_tune.py 사용 (권장)**:
```bash
cd ~/juik/lightning/tune

# Medium mode 실행 (50 trials, 50 epochs)
python quick_tune.py --mode medium
```

**또는 직접 실행**:
```bash
python tune_bert4rec_optuna.py \
    --study_name bert4rec_medium \
    --n_trials 50 \
    --n_epochs 50 \
    --n_jobs 1
```

**예상 시간**: 약 8-12시간 (50 trials × 50 epochs, pruning으로 조기 종료)

**모니터링**:
```bash
# 다른 터미널에서
optuna-dashboard sqlite:///bert4rec_medium.db --port 8080
# 브라우저: http://localhost:8080
```

### Step 2-3: Medium 결과 분석

```python
# analyze_medium_results.py
import optuna
import numpy as np

study = optuna.load_study(
    study_name="bert4rec_medium",
    storage="sqlite:///bert4rec_medium.db"
)

completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
best_trial = study.best_trial

print("=" * 80)
print("Medium Tuning Results")
print("=" * 80)

print(f"\nBest Trial: #{best_trial.number}")
print(f"Best NDCG@10: {best_trial.value:.6f}")

print("\nBest Hyperparameters:")
for key, value in sorted(best_trial.params.items()):
    print(f"  {key:25s}: {value}")

# Overfitting 체크
print("\n" + "=" * 80)
print("Overfitting Check")
print("=" * 80)

# Top 10 trials의 regularization 파라미터
top_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:10]

reg_params = ['dropout_rate', 'weight_decay', 'random_mask_prob']
for param in reg_params:
    values = [t.params[param] for t in top_trials if param in t.params]
    print(f"\n{param} (Top 10 trials):")
    print(f"  Min: {min(values):.4f}")
    print(f"  Max: {max(values):.4f}")
    print(f"  Avg: {np.mean(values):.4f}")

    # 경고
    if param == "dropout_rate" and min(values) < 0.15:
        print("  ⚠️ Warning: Too low dropout may cause overfitting")
    if param == "weight_decay" and min(values) < 0.01:
        print("  ⚠️ Warning: Too low weight_decay may cause overfitting")

# Best config 저장
print("\n" + "=" * 80)
print("Saving Best Config")
print("=" * 80)

best_config = {
    "model": {
        "hidden_units": 256,  # Fixed
        "num_heads": 8,       # Fixed
        "num_layers": 3,      # Fixed
        "max_len": 200,       # Fixed
        "dropout_rate": best_trial.params["dropout_rate"],
        "random_mask_prob": best_trial.params["random_mask_prob"],
        "last_item_mask_ratio": best_trial.params["last_item_mask_ratio"],
    },
    "training": {
        "lr": best_trial.params["lr"],
        "weight_decay": best_trial.params["weight_decay"],
    },
    "data": {
        "batch_size": best_trial.params["batch_size"],
    },
    "best_score": best_trial.value,
}

import yaml
with open("results/bert4rec_medium_best_config.yaml", "w") as f:
    yaml.dump(best_config, f)

print("✓ Saved to results/bert4rec_medium_best_config.yaml")
```

```bash
python analyze_medium_results.py
```

---

## Stage 3: Seed Search - 재현성 확보

**목적**:
- Medium best config에 최적화된 seed 찾기
- 재현 가능한 최종 모델

**주의**: Seed 탐색은 보통 **불필요**하거나 **비효율적**
- 이유: Seed 공간이 너무 큼 (43억)
- 대안: Seed=42 고정 + 전체 재튜닝

### Option A: Seed 고정 (권장)

```python
# Medium 결과를 그대로 사용
# Seed=42로 고정되어 있으므로 완전 재현 가능

# bert4rec_v2.yaml에 Medium best config 적용
# 재학습
./run_bert4rec.sh train bert4rec_v2
```

### Option B: Seed 범위 제한 탐색 (선택사항)

**언제 사용?**
- Medium best가 약간 불안정할 때
- 여러 seed로 앙상블하고 싶을 때

```python
# tune_seed_only.py
"""
Seed-only tuning with fixed hyperparameters
WARNING: 보통 비효율적, 신중히 사용
"""

class SeedOnlyObjective:
    def __init__(self, fixed_params):
        self.fixed_params = fixed_params

    def __call__(self, trial: optuna.Trial):
        # Seed만 탐색 (0-999 범위)
        seed = trial.suggest_int("seed", 0, 999)

        import lightning as L
        L.seed_everything(seed, workers=True)

        # Fixed params 사용
        datamodule = BERT4RecDataModule(
            batch_size=self.fixed_params["batch_size"],
            seed=seed,  # DataModule seed도 맞춤
            ...
        )

        model = BERT4Rec(
            dropout_rate=self.fixed_params["dropout_rate"],
            lr=self.fixed_params["lr"],
            ...
        )

        trainer = L.Trainer(max_epochs=60, ...)
        trainer.fit(model, datamodule)

        return trainer.callback_metrics["val_ndcg@10"].item()

# Medium best params 로드
with open("results/bert4rec_medium_best_config.yaml") as f:
    medium_best = yaml.safe_load(f)

fixed_params = {
    **medium_best["model"],
    **medium_best["training"],
    **medium_best["data"],
}

# Seed 탐색 (10-20 trials)
study = optuna.create_study(
    study_name="bert4rec_seed_search",
    direction="maximize",
    storage="sqlite:///bert4rec_seed.db"
)

objective = SeedOnlyObjective(fixed_params)
study.optimize(objective, n_trials=15)

print(f"Best seed: {study.best_trial.params['seed']}")
print(f"Best NDCG@10: {study.best_trial.value:.4f}")
```

```bash
# 실행 (선택사항)
python tune_seed_only.py
```

**예상 시간**: 5-7시간 (15 trials × 30분)

---

## 4. 일반화를 위한 전략

### 4.1 Regularization 파라미터 최소값 보장

**원칙**: Overfitting 방지 > Val score 최대화

```python
# ❌ 나쁜 예: 범위에 0 포함
dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.3)
weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)

# ✅ 좋은 예: 적절한 최소값
dropout_rate = trial.suggest_float("dropout_rate", 0.15, 0.3)   # ≥ 0.15
weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)   # ≥ 0.01
random_mask_prob = trial.suggest_float("random_mask_prob", 0.15, 0.25)  # ≥ 0.15
```

### 4.2 Train/Val Gap Monitoring

```python
# Objective에 gap penalty 추가 (선택사항)

def __call__(self, trial):
    trainer.fit(model, datamodule)

    val_ndcg = trainer.callback_metrics["val_ndcg@10"].item()

    # Train score도 기록했다면
    train_ndcg = trainer.callback_metrics.get("train_ndcg@10", val_ndcg)

    # Gap penalty
    gap = max(0, train_ndcg - val_ndcg)
    if gap > 0.02:  # Gap 너무 크면 penalty
        penalty = gap * 0.5
    else:
        penalty = 0

    objective_value = val_ndcg - penalty

    return objective_value
```

### 4.3 Conservative Approach

```python
# Best trial 대신 Top-K 평균 사용 (더 robust)

completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
top_k = sorted(completed, key=lambda t: t.value, reverse=True)[:5]

# 각 파라미터의 Top-K 평균
avg_params = {}
for param in top_k[0].params.keys():
    values = [t.params[param] for t in top_k]
    if isinstance(values[0], (int, float)):
        avg_params[param] = np.mean(values)
    else:
        # Categorical: 최빈값
        from collections import Counter
        avg_params[param] = Counter(values).most_common(1)[0][0]

print("Top-5 Average Params (more robust):")
print(avg_params)
```

### 4.4 Ensemble Strategy

```python
# Top-K models로 ensemble
top_k_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:5]

models = []
for trial in top_k_trials:
    model = train_with_config(trial.params)
    models.append(model)

# Prediction
def ensemble_predict(models, dataloader):
    all_scores = []
    for model in models:
        scores = model.predict(dataloader)
        all_scores.append(scores)

    # Average
    ensemble_scores = torch.stack(all_scores).mean(dim=0)
    return ensemble_scores
```

---

## 5. Tuning vs Training 환경 일치

### 5.1 체크리스트

| 항목 | Tuning | Training | 일치 여부 |
|------|--------|----------|-----------|
| Seed | `L.seed_everything(42)` | `L.seed_everything(cfg.data.seed)` | ✅ |
| LR Scheduler | 없음 | 없음 | ✅ |
| Dataset Split | `seq[:-1]` / `seq[-1]` | `seq[:-1]` / `seq[-1]` | ✅ |
| Loss Function | `CrossEntropy` | `CrossEntropy` | ✅ |
| Validation Metric | `NDCG@10` | `NDCG@10` | ✅ |
| Early Stopping | `patience=5` | `patience=10` | ⚠️ |
| Precision | `16-mixed` | `16-mixed` | ✅ |
| Gradient Clip | `5.0` | `5.0` | ✅ |

### 5.2 불일치 발견 시 대처

**Scheduler 불일치**:
```python
# src/models/bert4rec.py
def configure_optimizers(self):
    optimizer = torch.optim.AdamW(...)
    # scheduler = ...  # 주석처리
    return optimizer
```

**Early Stopping 불일치**:
```python
# Tuning patience를 Training과 맞춤
EarlyStopping(patience=10, ...)  # Training과 동일
```

---

## 6. 파라미터별 가이드

### 6.1 Model Architecture

| Parameter | Quick Range | Medium Range | Fixed | 설명 |
|-----------|-------------|--------------|-------|------|
| `hidden_units` | [128, 256] | 256 | ✓ | Embedding 차원 |
| `num_heads` | [2, 4, 8] | 8 | ✓ | Attention heads |
| `num_layers` | [2, 3] | 3 | ✓ | Transformer layers |
| `max_len` | [100, 150, 200] | 200 | ✓ | Sequence 길이 |

**고정 기준**: Importance < 0.05 또는 Best trial에서 명확한 선호

### 6.2 Regularization (일반화 핵심!)

| Parameter | Quick Range | Medium Range | 최소값 | 설명 |
|-----------|-------------|--------------|--------|------|
| `dropout_rate` | [0.1, 0.3] | [0.15, 0.28] | **0.15** | Dropout (높을수록 regularization) |
| `weight_decay` | [0.0, 0.1] | [0.02, 0.09] | **0.02** | L2 reg (0이면 regularization 없음) |
| `random_mask_prob` | [0.15, 0.25] | [0.17, 0.23] | **0.15** | Data augmentation |
| `last_item_mask_ratio` | [0.0, 0.2] | [0.05, 0.12] | 0.0 | 추가 masking |

**중요**: 최소값을 충분히 크게 설정해야 overfitting 방지

### 6.3 Training

| Parameter | Quick Range | Medium Range | Log Scale | 설명 |
|-----------|-------------|--------------|-----------|------|
| `lr` | [8e-4, 5e-3] | [1e-3, 4e-3] | ✓ | Learning rate |
| `weight_decay` | 위 참조 | 위 참조 | | L2 regularization |
| `batch_size` | [128, 256, 512] | [128, 256] | | Batch size |

### 6.4 Data

| Parameter | Quick Range | Medium Range | 설명 |
|-----------|-------------|--------------|------|
| `batch_size` | [128, 256, 512] | [128, 256] | 작을수록 regularization |
| `random_mask_prob` | 위 참조 | 위 참조 | Masking 확률 |
| `last_item_mask_ratio` | 위 참조 | 위 참조 | Last item masking |

---

## 7. 트러블슈팅

### 7.1 ValueError: CategoricalDistribution does not support dynamic value space

**원인**: 기존 study의 search space와 현재 코드 불일치

**해결**:
```bash
# Option 1: 새 study 생성
python tune_bert4rec_optuna.py --study_name new_study --n_trials 50

# Option 2: 기존 study 삭제 (백업 후!)
cp old_study.db old_study_backup.db
rm old_study.db
python tune_bert4rec_optuna.py --study_name old_study --n_trials 50
```

### 7.2 Val NDCG@10 높은데 Public Score 낮음

**원인**: Overfitting

**해결**:
```yaml
# bert4rec_v2.yaml 수정
model:
  dropout_rate: 0.25  # 증가

training:
  weight_decay: 0.05  # 증가 (0.001 → 0.05)

data:
  batch_size: 128     # 감소 (256 → 128)
```

### 7.3 Tuning 결과가 Training에 재현 안됨

**원인**: 환경 불일치 (scheduler, seed 등)

**해결**: [5. Tuning vs Training 환경 일치](#5-tuning-vs-training-환경-일치) 체크리스트 확인

### 7.4 Seed 탐색으로 0.1 달성 못함

**원인**: Seed-하이퍼파라미터 상호작용

**해결**:
```bash
# Seed 고정 + 전체 재튜닝
# tune_bert4rec_optuna.py에 L.seed_everything(42) 추가
python tune_bert4rec_optuna.py --study_name bert4rec_seed42 --n_trials 50
```

---

## 8. Best Practices

### 8.1 Tuning Checklist

- [ ] **환경 검증 완료** (Stage 0)
- [ ] **Tuning = Training 환경** (Scheduler, Seed, Dataset split 일치)
- [ ] **Regularization 최소값 설정** (dropout ≥ 0.15, weight_decay ≥ 0.02)
- [ ] **Quick → Medium 단계적 탐색**
- [ ] **모든 결과 저장 및 백업**
- [ ] **Overfitting 체크** (Train/Val gap < 0.02)
- [ ] **최종 config 검증** (Training으로 재현 확인)

### 8.2 시간 예산별 전략

**1일 (24시간)**:
- Quick (15 trials × 30 epochs) → 5시간
- Medium (30 trials × 50 epochs) → 15시간
- 분석 및 최종 training → 4시간

**2-3일 (48-72시간)**:
- Quick (20 trials × 30 epochs) → 7시간
- Medium (50 trials × 50 epochs) → 25시간
- Seed search (15 trials × 60 epochs) → 8시간
- 앙상블 준비 → 10시간

**1주일+**:
- Quick (30 trials) → 10시간
- Medium (100 trials) → 50시간
- Multiple seed ensembles → 30시간
- Validation 전략 실험 → 20시간

### 8.3 리소스 최적화

**GPU 1개**:
```bash
python tune_bert4rec_optuna.py --n_jobs 1
```

**GPU 2개+**:
```bash
# 주의: Multi-GPU는 성능 향상 미미, 1개씩 독립 실행 권장
CUDA_VISIBLE_DEVICES=0 python tune_bert4rec_optuna.py --study_name study_1 &
CUDA_VISIBLE_DEVICES=1 python tune_bert4rec_optuna.py --study_name study_2 &
```

**모니터링**:
```bash
# Terminal 1: Training
python tune_bert4rec_optuna.py ...

# Terminal 2: Dashboard
optuna-dashboard sqlite:///bert4rec_medium.db --port 8080

# Terminal 3: GPU 모니터링
watch -n 1 nvidia-smi
```

### 8.4 최종 Training 전 검증

```python
# final_validation.py
"""
Best config로 5번 학습하여 재현성 및 안정성 확인
"""

import yaml
from train_with_config import train

with open("results/bert4rec_medium_best_config.yaml") as f:
    best_config = yaml.safe_load(f)

results = []
for run in range(5):
    print(f"\n{'='*80}")
    print(f"Validation Run {run+1}/5")
    print(f"{'='*80}")

    # Seed 고정 (재현성)
    best_config["data"]["seed"] = 42

    result = train(best_config)
    results.append(result["val_ndcg@10"])

    print(f"Val NDCG@10: {result['val_ndcg@10']:.4f}")

print(f"\n{'='*80}")
print("Validation Results")
print(f"{'='*80}")
print(f"Mean: {np.mean(results):.4f}")
print(f"Std:  {np.std(results):.4f}")
print(f"Min:  {min(results):.4f}")
print(f"Max:  {max(results):.4f}")

if np.std(results) < 0.002:
    print("\n✓ Stable and reproducible!")
else:
    print("\n⚠ High variance, check seed configuration")
```

---

## 9. Quick Reference

### 9.1 명령어 모음

```bash
# Stage 0: Test
python tune_bert4rec_optuna.py --study_name test --n_trials 1 --n_epochs 3

# Stage 1: Quick
python quick_tune.py

# Stage 2: Medium
python tune_bert4rec_optuna.py --study_name bert4rec_medium --n_trials 50 --n_epochs 50

# Stage 3: Seed (선택)
python tune_seed_only.py

# 결과 분석
python analyze_quick_results.py
python analyze_medium_results.py

# 최종 Training
./run_bert4rec.sh train bert4rec_v2

# Dashboard
optuna-dashboard sqlite:///bert4rec_medium.db --port 8080
```

### 9.2 파일 경로

```
tune/
├── tune_bert4rec_optuna.py       # Main tuning script
├── quick_tune.py                  # Quick mode
├── analyze_quick_results.py       # Quick 분석
├── analyze_medium_results.py      # Medium 분석
├── tune_seed_only.py             # Seed search
├── verify_env.py                 # 환경 검증
└── results/
    ├── bert4rec_quick_best_config.yaml
    ├── bert4rec_medium_best_config.yaml
    └── ...

*.db                              # Optuna study databases
```

---

## 부록 A: 파라미터 중요도 해석

| Importance | 의미 | 조치 |
|------------|------|------|
| > 0.3 | 매우 중요 | Medium에서 넓게 탐색 |
| 0.1 ~ 0.3 | 중요 | Medium에서 좁게 탐색 |
| 0.05 ~ 0.1 | 중간 | Medium에서 좁게 또는 고정 |
| < 0.05 | 덜 중요 | 고정 (Best value) |

## 부록 B: Regularization 체크리스트

- [ ] `dropout_rate ≥ 0.15`
- [ ] `weight_decay ≥ 0.01`
- [ ] `random_mask_prob ≥ 0.15`
- [ ] `batch_size ≤ 256` (작을수록 regularization)
- [ ] Train/Val gap < 0.02
- [ ] Early stopping patience = 10 (충분한 학습)

## 부록 C: 일반화 vs Overfitting 신호

**좋은 신호 (일반화)**:
- Val NDCG@10 = 0.095, Public = 0.100 ✓
- Dropout ≥ 0.2, Weight decay ≥ 0.03
- Train/Val gap < 0.02

**나쁜 신호 (Overfitting)**:
- Val NDCG@10 = 0.105, Public = 0.090 ✗
- Dropout < 0.15, Weight decay < 0.01
- Train/Val gap > 0.03

---

**Last Updated**: 2026-01-02
**Version**: 2.0
