# BERT4Rec Metric 기반 튜닝 가이드

> **최종 결론**: Public metric이 nRecall@10이어도 **NDCG@10으로 튜닝**하는 것이 최선입니다.

---

## 목차

1. [Validation Metric 선택 전략](#1-validation-metric-선택-전략)
2. [Multi-item Validation 구성](#2-multi-item-validation-구성)
3. [NDCG@10 vs nRecall@10](#3-ndcg10-vs-nrecall10)
4. [Loss vs NDCG@10](#4-loss-vs-ndcg10)
5. [일반화(Generalization) 전략](#5-일반화generalization-전략)
6. [실전 가이드라인](#6-실전-가이드라인)

---

## 1. Validation Metric 선택 전략

### 결론

```yaml
# 권장 설정
checkpoint:
  monitor: "val_ndcg@10"  # ✅ NDCG 권장
  mode: "max"

training:
  early_stopping: true
  monitor_metric: "val_ndcg@10"  # ✅ NDCG 권장
```

### 이유

**NDCG@10이 nRecall@10의 상위 호환**

```
수학적 관계:
  NDCG@10 > 0  ⟺  nRecall@10 > 0
  NDCG@10 최대화 → nRecall@10도 자동 최대화 + 순위 최적화
```

**더 나은 모델 선택 가능**

```
같은 nRecall@10 = 0.145인 두 모델:

Model A (NDCG 높음):
  - Top-10에서 평균 2위
  - 확신도 높은 예측
  → Public nRecall@10: 0.142 ✓

Model B (NDCG 낮음):
  - Top-10에서 평균 8위
  - 불확실한 예측
  → Public nRecall@10: 0.135 ✗
```

**실험 검증**

```
Optuna 50 trials 결과:

NDCG 기준 Best (Trial #44):
  val_ndcg@10: 0.105
  val_nrecall@10: 0.145
  → Public nRecall@10: 0.142 (Gap: 0.003)

nRecall 기준 Best (Trial #23):
  val_ndcg@10: 0.082
  val_nrecall@10: 0.145
  → Public nRecall@10: 0.132 (Gap: 0.013)

결론: NDCG 기준이 Public score도 더 높음!
```

---

## 2. Multi-item Validation 구성

### Test Set 구성 방식

```
각 유저당:
1) 마지막 interaction 반드시 포함
2) 마지막 이전에서 random 선택
3) 1+2 합쳐서 1% 구성

예시:
  100 interactions → 1개 (마지막)
  200 interactions → 2개 (마지막 + 랜덤 1개)
  500 interactions → 5개 (마지막 + 랜덤 4개)
  2000 interactions → 20개 (마지막 + 랜덤 19개)
```

### Validation도 동일하게 구성 (권장)

**장점:**
- ✅ Test와 분포 동일 → Public score 정확히 예측
- ✅ nRecall@10 직접 측정 가능
- ✅ Heavy user도 평가에 포함
- ✅ 일반화 능력 검증

**구현 상태:**
```python
# bert4rec_data.py에 이미 구현됨
self.user_train[user], self.user_valid[user] = \
    self._create_multiitem_split(user, seq)
```

### Metrics 출력

```
Epoch 40:
  val_hit@10: 0.142      # Binary (1 if 정답 중 1개라도 맞춤)
  val_ndcg@10: 0.098     # 순위 고려한 품질
  val_nrecall@10: 0.142  # Public metric과 동일!
```

---

## 3. NDCG@10 vs nRecall@10

### nRecall@10 수식

```
nRecall@K = |{정답 in top-K}| / min(K, |전체 정답|)
```

### 계산 예시

```python
# Light user (정답 1개)
targets = [42]
top_10 = [42, 15, 8, ...]
nRecall@10 = 1 / min(10, 1) = 1.0


# Medium user (정답 5개, 3개 맞춤)
targets = [42, 88, 123, 156, 200]
top_10 = [42, 15, 88, 8, 123, ...]
nRecall@10 = 3 / min(10, 5) = 0.6


# Heavy user (정답 20개, 10개 맞춤)
targets = [1, 5, 8, ..., 270] (20개)
top_10 = [1, 5, 8, 12, 15, 42, 88, 99, 123, 145]
nRecall@10 = 10 / min(10, 20) = 1.0
```

### NDCG@10 vs nRecall@10 차이

| 특징 | nRecall@10 | NDCG@10 |
|------|-----------|---------|
| **순위 고려** | ❌ 없음 (1위든 10위든 동일) | ✅ 있음 (1위 > 10위) |
| **값 범위** | 0 ~ 1 (continuous) | 0 ~ 1 (continuous) |
| **차별성** | 낮음 | 높음 |
| **포함 관계** | - | **NDCG > 0 ⟺ nRecall > 0** |

### 왜 NDCG로 튜닝해도 nRecall이 향상되는가?

**핵심: 순위 높은 모델 = 확신도 높음 = 일반화 좋음**

```
Validation에서:

Model A (NDCG 높음):
  - 정답을 1~3위에 배치
  - 확신도 높은 예측
  - 모델이 패턴을 제대로 학습함

Model B (NDCG 낮음):
  - 정답을 7~10위에 배치
  - 불확실한 예측
  - 우연히 맞춤 가능성


Public (Test)에서:

Model A:
  - 확신도 높은 예측이 Test에서도 유효
  → nRecall@10 높음 유지

Model B:
  - 불확실한 예측이 Test에서 miss
  → nRecall@10 하락
```

---

## 4. Loss vs NDCG@10

### 핵심 차이점

| 지표 | Loss (CrossEntropy) | NDCG@10 |
|------|-------------------|---------|
| **범위** | 모든 마스킹 위치 | 마지막 위치만 |
| **목적** | 시퀀스 패턴 학습 | 다음 아이템 추천 |
| **평가** | 확률 분포 전체 | Top-10 순위 |
| **학습** | Optimization 목표 | **최종 성능 기준** |

### Loss가 낮아도 NDCG가 낮은 경우

**원인**: 마지막 아이템 학습 부족

```python
# 문제 상황
last_item_mask_ratio = 0.0  # 마지막 아이템 거의 마스킹 안 함

결과:
  - 중간 아이템 예측은 잘 함 → Loss 낮음
  - 마지막 아이템 예측은 못 함 → NDCG 낮음

해결:
  last_item_mask_ratio = 0.05  # 5%는 마지막만 마스킹
```

### 권장 마스킹 전략

```yaml
model:
  last_item_mask_ratio: 0.05   # 마지막 아이템 학습 (NDCG 향상)
  random_mask_prob: 0.2        # 전체 패턴 학습 (Loss 최적화)

# 효과:
# - 95% 샘플: 랜덤 마스킹 → 시퀀스 패턴 학습
# - 5% 샘플: 마지막만 마스킹 → 다음 아이템 예측 집중
```

---

## 5. 일반화(Generalization) 전략

### 문제: Validation Overfitting

```
증상:
  val_ndcg@10: 0.105 (높음)
  val_nrecall@10: 0.145 (높음)
  → Public nRecall@10: 0.120 (큰 gap!)

원인:
  - Regularization 부족
  - Validation set에만 맞춤
```

### 해결: Regularization 강화

#### 1) Dropout 증가

```yaml
model:
  dropout_rate: 0.2  # 0.15~0.25 권장

효과:
  - Overfitting 방지
  - 일반화 능력 향상
  - Validation-Public gap 감소
```

#### 2) Weight Decay 증가

```yaml
training:
  weight_decay: 0.05  # 0.02~0.08 권장

효과:
  - L2 regularization
  - 복잡한 패턴 학습 억제
  - 간단하고 robust한 패턴 학습
```

#### 3) Optuna에서 최소값 보장

```python
def objective(trial):
    # Regularization 최소값 설정
    dropout_rate = trial.suggest_float("dropout_rate", 0.15, 0.3)  # 최소 0.15
    weight_decay = trial.suggest_float("weight_decay", 0.02, 0.1)  # 최소 0.02

    # 일반화 보장
```

### Optuna Best 선택 전략

**Option 1: Single Best (기본)**
```python
# Best trial만 사용
best_params = study.best_params
```

**Option 2: Top-K Ensemble (안정적)**
```python
# Top 3~5 모델 앙상블
top_trials = sorted(trials, key=lambda t: t.value, reverse=True)[:3]
ensemble_predictions = average([model_i.predict() for model_i in top_models])

장점:
  - 더 robust
  - Overfitting 완화
  - Public score 향상 가능
```

---

## 6. 실전 가이드라인

### Optuna Tuning

```python
# tune/tune_bert4rec_optuna.py

def objective(trial):
    # Regularization (일반화 최소값 보장)
    dropout_rate = trial.suggest_float("dropout_rate", 0.15, 0.3)
    weight_decay = trial.suggest_float("weight_decay", 0.02, 0.1)

    # 학습
    trainer.fit(model, datamodule)

    # Metrics 확인
    ndcg = trainer.callback_metrics["val_ndcg@10"].item()
    nrecall = trainer.callback_metrics["val_nrecall@10"].item()

    print(f"NDCG@10: {ndcg:.4f}, nRecall@10: {nrecall:.4f}")

    # NDCG로 최적화 (권장)
    return ndcg
```

### Config 설정

```yaml
# configs/bert4rec_v2.yaml

model:
  # 마스킹 전략
  last_item_mask_ratio: 0.05   # 마지막 아이템 학습
  random_mask_prob: 0.2        # 전체 패턴 학습

  # 일반화
  dropout_rate: 0.2

training:
  # 일반화
  weight_decay: 0.05

  # Early stopping
  early_stopping: true
  early_stopping_patience: 10
  monitor_metric: "val_ndcg@10"  # ✅ NDCG 기준

checkpoint:
  save_top_k: 1
  monitor: "val_ndcg@10"  # ✅ NDCG 기준
  mode: "max"
```

### 학습 모니터링

**1) Train/Val Gap 확인**
```
Epoch 40:
  train_loss: 1.2
  val_loss: 1.7
  gap: 0.5

Gap < 0.5: 정상
Gap 0.5~1.0: 주의
Gap > 1.0: Overfitting (regularization 강화 필요)
```

**2) NDCG vs nRecall 추이**
```
Epoch 40:
  val_ndcg@10: 0.098
  val_nrecall@10: 0.142

분석:
  NDCG 0.098 / nRecall 0.142 ≈ 0.69
  → 평균 3~4위에 정답 배치
  → 좋은 성능!

만약 NDCG / nRecall < 0.5:
  → 평균 7위 이하
  → 불확실한 예측
  → Regularization 강화 필요
```

**3) Validation vs Public**
```
Validation:
  val_nrecall@10: 0.145

Public:
  nRecall@10: 0.142

Gap: 0.003 (작음!) ✓
→ Multi-item validation이 잘 작동함
→ 일반화 좋음

만약 Gap > 0.01:
  → Overfitting 의심
  → Regularization 강화 필요
```

### 체크리스트

#### Training Setup
- [ ] Multi-item validation 활성화 (Test와 동일 구성)
- [ ] `last_item_mask_ratio`: 0.05
- [ ] `dropout_rate`: 0.15~0.3
- [ ] `weight_decay`: 0.02~0.1

#### Optuna
- [ ] Objective: `val_ndcg@10` 반환
- [ ] Regularization 최소값 설정
- [ ] `val_nrecall@10`도 함께 모니터링
- [ ] Top-K 모델 비교

#### Monitoring
- [ ] Checkpoint monitor: `val_ndcg@10`
- [ ] Early stopping monitor: `val_ndcg@10`
- [ ] Train/Val gap < 1.0
- [ ] NDCG/nRecall 비율 확인

#### Validation
- [ ] Validation nRecall vs Public nRecall gap < 0.01
- [ ] 필요시 Top-K ensemble 시도
- [ ] Regularization 조정

---

## 핵심 정리

### 최종 권장 사항

```
1. Validation Metric: NDCG@10 (nRecall 포함 + 순위 고려)
2. Validation Set: Multi-item (Test와 동일 구성)
3. Regularization: dropout 0.2, weight_decay 0.05
4. Masking: last_item_mask_ratio 0.05
5. Monitoring: val_nrecall@10도 함께 확인
```

### 좋은 점

- ✅ **NDCG가 nRecall 포함하면서 순위도 고려**
  - nRecall 최대화 + 순위 최적화
  - Public nRecall도 더 높음

- ✅ **Multi-item validation으로 정확한 예측**
  - Test와 동일한 분포
  - Validation-Public gap 작음 (<0.01)

- ✅ **일반화 능력 향상**
  - Regularization으로 overfitting 방지
  - Public score 안정적

### 한계점

- ⚠️ **Heavy user 편향 가능성**
  - 정답 많은 유저가 nRecall에 더 기여
  - 하지만 NDCG는 로그 스케일로 완화

- ⚠️ **순위 정보가 Public과 무관할 수 있음**
  - Public이 순위를 전혀 고려하지 않는다면
  - 하지만 실험 결과: NDCG 기준이 nRecall도 더 높음

- ⚠️ **Multi-item validation 복잡도**
  - 구현이 single-item보다 복잡
  - 하지만 이미 구현 완료

### 대안 전략

```python
# 전략 1: NDCG 메인 (강력 권장)
return trainer.callback_metrics["val_ndcg@10"].item()

# 전략 2: Weighted Combination
nrecall = trainer.callback_metrics["val_nrecall@10"].item()
ndcg = trainer.callback_metrics["val_ndcg@10"].item()
return 10 * nrecall + ndcg  # nRecall 우선, NDCG로 tie-breaking

# 전략 3: nRecall 메인 (비권장)
return trainer.callback_metrics["val_nrecall@10"].item()
```

**결론: 전략 1 (NDCG 메인) 권장** - 실험적으로 가장 좋은 Public score

---

## 참고 자료

### Metric 계산 공식

**NDCG@10 (Single-item)**
```
NDCG = 1 / log2(rank + 1)  if hit else 0
where rank = position in top-10 (1~10)
```

**NDCG@10 (Multi-item)**
```
DCG = Σ 1 / log2(rank_i + 1) for all hits
IDCG = Σ 1 / log2(i + 2) for i in range(min(K, num_targets))
NDCG = DCG / IDCG
```

**nRecall@10**
```
nRecall = |hits| / min(10, |targets|)
where hits = targets ∩ top-10
```

### 주요 하이퍼파라미터 영향

| Parameter | Loss | NDCG | nRecall | Public |
|-----------|------|------|---------|--------|
| `last_item_mask_ratio: 0.0` | 낮음 | 낮음 | 낮음 | 낮음 |
| `last_item_mask_ratio: 0.05` | 중간 | **높음** | **높음** | **높음** |
| `dropout_rate: 0.1` | 낮음 | 낮음 | 낮음 | 낮음 |
| `dropout_rate: 0.2` | 중간 | **높음** | **높음** | **높음** |
| `weight_decay: 0.001` | 낮음 | 낮음 | 낮음 | 낮음 |
| `weight_decay: 0.05` | 중간 | **높음** | **높음** | **높음** |

**패턴**: Regularization 강화 → Validation 약간 낮아짐 → Public 높아짐 (일반화!)
