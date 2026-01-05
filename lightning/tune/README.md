# BERT4Rec Hyperparameter Tuning with Optuna

Optuna 기반의 BERT4Rec 하이퍼파라미터 자동 튜닝 도구입니다.

## 디렉토리 구조

```
tune/
├── README.md                          # 이 파일
├── quick_tune.py                      # 빠른 튜닝 스크립트 (권장)
├── tune_bert4rec_optuna.py            # 기본 튜닝 스크립트
├── tune_bert4rec_optuna_monitored.py  # 모니터링 강화 버전
├── bert4rec_*.db                      # Optuna study 데이터베이스
├── docs/                              # 문서
│   ├── README_optuna.md               # Optuna 튜닝 상세 가이드
│   └── MONITORING_GUIDE_optuna.md     # 모니터링 가이드
└── results/                           # 튜닝 결과
    ├── bert4rec_*_best_config.yaml    # 최적 하이퍼파라미터
    ├── bert4rec_*_history.html        # 최적화 히스토리
    ├── bert4rec_*_importance.html     # 파라미터 중요도
    └── bert4rec_*_parallel.html       # 병렬 좌표 플롯
```

## 빠른 시작

### 1. 기본 사용법 (권장)

```bash
cd tune

# Test 모드 (2 trials, 2 epochs) - 스크립트 테스트용
python quick_tune.py --mode test

# Quick 모드 (10 trials, 20 epochs) - 약 2-3시간
python quick_tune.py --mode quick

# Medium 모드 (30 trials, 50 epochs) - 약 8-12시간
python quick_tune.py --mode medium

# Full 모드 (100 trials, 100 epochs) - 약 1-2일
python quick_tune.py --mode full
```

### 2. Study 재개 (중단 후 이어서 실행)

Ctrl+C로 중단했거나 에러로 중단된 경우, `--resume` 플래그로 이어서 실행:

```bash
cd tune

# quick_tune.py로 재개
python quick_tune.py --mode medium --resume

# 직접 실행으로 재개
python tune_bert4rec_optuna.py \
    --study_name bert4rec_medium \
    --n_trials 20 \
    --resume
```

**자동 처리 기능**:
- ✅ Stuck RUNNING trials 자동 FAILED 처리
- ✅ 기존 trial 현황 출력 (완료/pruned/전체)
- ✅ 추가로 실행할 trial 수만큼 이어서 진행

### 3. 직접 실행 (고급)

```bash
cd tune

# 기본 튜닝
python tune_bert4rec_optuna.py --n_trials 50 --n_epochs 50

# 모니터링 강화 버전
python tune_bert4rec_optuna_monitored.py --n_trials 50
```

## 병렬 실행 (Multi-GPU)

GPU가 여러 개 있는 경우:

```bash
cd tune

# GPU 2개 사용
python tune_bert4rec_optuna.py --n_trials 50 --n_jobs 2

# GPU 4개 사용
python quick_tune.py --mode medium --n_jobs 4
```

**주의**: `n_jobs` 값만큼 GPU가 필요합니다.

## 실시간 모니터링

터미널을 하나 더 열어서:

```bash
cd tune

# Optuna Dashboard 실행
optuna-dashboard sqlite:///bert4rec_medium.db
```

브라우저에서 http://127.0.0.1:8080 열기

## 결과 확인

### 1. 최적 하이퍼파라미터

```bash
cd tune

cat results/bert4rec_medium_best_config.yaml
```

### 2. 시각화 (HTML)

```bash
cd tune/results

# 브라우저로 열기
open bert4rec_medium_history.html      # 최적화 히스토리
open bert4rec_medium_importance.html   # 파라미터 중요도
open bert4rec_medium_parallel.html     # 병렬 좌표 플롯
```

### 3. SQLite 직접 조회

```bash
cd tune

# 상위 5개 trial 확인
sqlite3 bert4rec_medium.db \
  'SELECT number, value FROM trials ORDER BY value DESC LIMIT 5;'
```

## 주요 파라미터

### quick_tune.py

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--mode` | 튜닝 모드 (test/quick/medium/full) | test |
| `--data_dir` | 데이터 디렉토리 | ~/data/train/ |
| `--resume` | Study 재개 (중단 후 이어서 실행) | False |

**모드별 설정**:
| 모드 | Trial 수 | Epoch 수 | 예상 시간 |
|------|----------|----------|----------|
| test | 2 | 2 | ~2-3분 |
| quick | 10 | 20 | ~2-3시간 |
| medium | 30 | 50 | ~8-12시간 |
| full | 100 | 100 | ~1-2일 |

### tune_bert4rec_optuna.py

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--n_trials` | Trial 개수 | 50 |
| `--n_epochs` | Trial당 최대 epoch | 50 |
| `--study_name` | Study 이름 | bert4rec_study |
| `--n_jobs` | 병렬 실행 개수 (GPU 수) | 1 |
| `--no_pruning` | Pruning 비활성화 | False |
| `--resume` | Study 재개 | False |

## 튜닝 대상 하이퍼파라미터

1. **모델 아키텍처**
   - `hidden_units`: [64, 128, 256]
   - `num_heads`: [2, 4, 8]
   - `num_layers`: [1, 2, 3]
   - `max_len`: [50, 100, 150, 200]
   - `dropout_rate`: [0.1 ~ 0.5]

2. **학습 파라미터**
   - `lr`: [1e-4 ~ 1e-2] (log scale)
   - `weight_decay`: [0.0 ~ 0.1]
   - `batch_size`: [128, 256, 512]

3. **마스킹 전략**
   - `random_mask_prob`: [0.1 ~ 0.3]
   - `last_item_mask_ratio`: [0.0 ~ 0.5]

## Trial 상태

- **Complete**: 정상 완료
- **Pruned**: 성능이 낮아 조기 종료 (정상, 에러 아님)
- **Running**: 실행 중
- **Failed**: 에러 발생 또는 중단된 trial

## 유용한 팁

### 1. 중단 후 재개 (Resume)

언제든지 Ctrl+C로 중단하고, `--resume`으로 이어서 실행 가능:
```bash
# 중단 (Ctrl+C)
# ...

# 이어서 실행
python quick_tune.py --mode medium --resume
```

### 2. Study User Attributes

Optuna Dashboard에서 "Study User Attributes" 섹션을 확인하면:
- 실험 시작 시간 (`created_at`)
- 데이터 디렉토리 (`data_dir`)
- Python/PyTorch 버전
- GPU 정보
- 튜닝 설정값

등의 메타데이터를 볼 수 있습니다.

### 3. 최적 파라미터로 학습하기

튜닝 완료 후:
```bash
# 1. 최적 설정 확인
cat tune/results/bert4rec_medium_best_config.yaml

# 2. 해당 파라미터를 configs/bert4rec_v2.yaml에 반영

# 3. 전체 데이터로 학습
python train_bert4rec.py
```

### 4. 여러 Study 비교

```bash
# Dashboard에서 여러 study 동시 확인
optuna-dashboard sqlite:///bert4rec_quick.db sqlite:///bert4rec_medium.db
```

## 더 알아보기

- **하이터파라미터 튜닝 가이드**: [docs/HYPERPARAMETER_TUNING_GUIDE.md](docs/HYPERPARAMETER_TUNING_GUIDE.md)
- **모니터링 가이드**: [docs/MONITORING_GUIDE_optuna.md](docs/MONITORING_GUIDE_optuna.md)

## 문제 해결

### Stuck RUNNING trials (중단 후 trial이 running 상태로 남아있음)

**증상**: Ctrl+C로 중단한 후, Dashboard에서 여러 trial이 RUNNING 상태로 표시됨

**해결**: `--resume` 플래그로 다시 실행하면 **자동으로 FAILED 처리**
```bash
python quick_tune.py --mode medium --resume
```

출력 예시:
```
Resuming study: bert4rec_medium
Loaded existing study with 45 trials
⚠️  Found 2 stuck RUNNING trials from previous interrupted run
  ✓ Marked trial #41 as FAILED
  ✓ Marked trial #44 as FAILED
Resume mode: 45 trials already exist (43 completed, 2 pruned)
Will run 30 additional trials...
```

### Study가 없다는 에러

```bash
# resume=False로 변경하거나
python tune_bert4rec_optuna.py --study_name new_study

# 또는 기존 study 삭제 후 재시작
rm bert4rec_*.db
```

### .db.db 파일이 생성됨

이전 버전에서 생성된 경우:
```bash
cd tune
rm *.db.db  # 안전하게 삭제 가능
```
현재 버전에서는 자동으로 방지됨 ✅

### GPU 메모리 부족

```bash
# batch_size 범위 축소 (tune_bert4rec_optuna.py 수정)
batch_size = trial.suggest_categorical('batch_size', [128, 256])  # 512 제거
```

### Pruning 너무 공격적

```bash
# Pruning 비활성화
python tune_bert4rec_optuna.py --no_pruning
```

### Metadata 로딩이 느림

Metadata(genre, director, writer, title embeddings)를 사용하지 않는 경우:
- ✅ `quick_tune.py`: 자동으로 metadata 로딩 비활성화 (속도 최적화)
- ✅ 일반 학습: `configs/bert4rec_v2.yaml`에서 `use_*_emb: false` 설정
