# Preprocessing Scripts

## Title + Genre Embeddings (권장)

영화 제목과 장르를 결합하여 `mxbai-embed-large-v1` 모델로 임베딩을 생성합니다.

### Quick Start

```bash
# 1. 기본 사용 (가장 간단)
python scripts/preprocess_title_genre_embeddings.py

# 2. 안전하게 백업 생성 후 실행
python scripts/preprocess_title_genre_embeddings.py --backup-original

# 3. GPU 메모리 부족 시 배치 크기 조정
python scripts/preprocess_title_genre_embeddings.py --batch-size 16
```

### 상세 옵션

```bash
python scripts/preprocess_title_genre_embeddings.py \
    --data-dir ~/data/train \              # 데이터 디렉토리
    --output-path ~/data/train/title_embeddings/titles.tsv \ # 출력 파일 경로
    --model-name mixedbread-ai/mxbai-embed-large-v1 \  # 모델 선택
    --batch-size 32 \                      # 배치 크기
    --max-length 128 \                     # 최대 토큰 길이
    --device cuda \                        # GPU/CPU 선택
    --backup-original                      # 원본 백업 (원본 덮어쓸 때만)
```

### 지원 모델

| 모델 | 차원 | 특징 |
|------|------|------|
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | MTEB 최상위, 추천 시스템 최적화 (권장) |
| `sentence-transformers/all-mpnet-base-v2` | 768 | 고품질, 빠른 추론 |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 경량화, 베이스라인 |

### 출력 형식

**기본 출력 위치**: `~/data/train/title_embeddings/titles.tsv`

```tsv
item	title
318	0.123 -0.456 0.789 ... (1024개 값)
2571	-0.234 0.567 -0.890 ...
```

**참고**: 원본 `titles.tsv` 파일은 그대로 유지되며, 생성된 임베딩은 별도의 `title_embeddings/` 디렉토리에 저장됩니다.

### 예상 실행 시간

- **GPU (T4)**: ~2-3분 (6807 items)
- **CPU**: ~15-20분

### 트러블슈팅

**문제**: `ModuleNotFoundError: No module named 'transformers'`
```bash
pip install transformers torch
```

**문제**: GPU 메모리 부족
```bash
python scripts/preprocess_title_genre_embeddings.py --batch-size 8
```

**문제**: 모델 다운로드 실패
```bash
# 프록시 설정 또는 미러 사이트 사용
export HF_ENDPOINT=https://hf-mirror.com
python scripts/preprocess_title_genre_embeddings.py
```

## Legacy Scripts

### preprocess_title_embeddings.py

제목만 사용하는 이전 버전 스크립트입니다. 새로운 프로젝트는 `preprocess_title_genre_embeddings.py` 사용을 권장합니다.

```bash
python scripts/preprocess_title_embeddings.py \
    --model-name sentence-transformers/all-MiniLM-L6-v2
```
