"""
Title + Genre Text Embedding 전처리 스크립트

영화 제목과 장르를 결합한 텍스트를 mxbai-embed-large-v1 모델로 인코딩합니다.
생성된 embedding은 BERT4Rec 모델에서 item representation 강화에 사용됩니다.

Usage:
    # 기본 사용 (mxbai-embed-large-v1 모델 사용)
    python scripts/preprocess_title_genre_embeddings.py

    # 다른 모델 사용
    python scripts/preprocess_title_genre_embeddings.py --model-name sentence-transformers/all-mpnet-base-v2

    # 다른 경로 지정
    python scripts/preprocess_title_genre_embeddings.py --data-dir ~/data/train

Output:
    {data_dir}/title_embeddings/titles.tsv - 각 아이템별 임베딩 벡터 (TSV 형식)
    형식: item\ttitle
          1\t0.123 -0.456 0.789 ... (1024개 값)
"""

import os
import sys
import argparse
import logging
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def mean_pooling(model_output, attention_mask):
    """
    Mean pooling - attention mask를 고려한 평균 풀링

    Args:
        model_output: BERT 모델의 출력 (last_hidden_state)
        attention_mask: Attention mask

    Returns:
        torch.Tensor: Pooled sentence embedding
    """
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def load_genres(genres_path):
    """
    장르 데이터 로드 및 item별 집계

    Args:
        genres_path: genres.tsv 파일 경로

    Returns:
        Dict[int, List[str]]: {item_id: [genre1, genre2, ...]}
    """
    log.info(f"Loading genres from: {genres_path}")
    if not os.path.exists(genres_path):
        log.warning(f"genres.tsv not found at {genres_path}, proceeding without genres")
        return {}

    genres_df = pd.read_csv(genres_path, sep="\t")
    log.info(f"Loaded {len(genres_df)} genre entries")

    # item별 장르 리스트 생성
    item_genres = defaultdict(list)
    for _, row in genres_df.iterrows():
        item_genres[row["item"]].append(row["genre"])

    log.info(f"Aggregated genres for {len(item_genres)} items")
    return dict(item_genres)


def create_combined_text(title, genres):
    """
    타이틀과 장르를 결합한 텍스트 생성

    Args:
        title: 영화 제목
        genres: 장르 리스트

    Returns:
        str: 결합된 텍스트

    Examples:
        >>> create_combined_text("The Matrix", ["Action", "Sci-Fi"])
        "The Matrix. Genres: Action, Sci-Fi"

        >>> create_combined_text("Inception", [])
        "Inception"
    """
    if not genres:
        return title

    genre_str = ", ".join(genres)
    return f"{title}. Genres: {genre_str}"


def generate_title_genre_embeddings(
    titles_path,
    genres_path,
    output_path,
    model_name="mixedbread-ai/mxbai-embed-large-v1",
    batch_size=32,
    max_length=128,
    device=None,
):
    """
    영화 제목 + 장르를 결합하여 BERT embedding 생성

    Args:
        titles_path: titles.tsv 파일 경로
        genres_path: genres.tsv 파일 경로
        output_path: 저장할 파일 경로 (titles.tsv)
        model_name: Hugging Face 모델명
            - "mixedbread-ai/mxbai-embed-large-v1": 1024-dim, MTEB 최상위 (추천)
            - "sentence-transformers/all-mpnet-base-v2": 768-dim, 고품질
            - "sentence-transformers/all-MiniLM-L6-v2": 384-dim, 빠름
        batch_size: 배치 크기 (GPU 메모리에 따라 조정)
        max_length: 최대 토큰 길이
        device: 'cuda', 'cpu', 또는 None (자동 선택)

    Returns:
        tuple: (embedding_dim, num_items)
    """
    # Device 설정
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    # 모델 로드
    log.info(f"Loading model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        log.info("Model loaded successfully")
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        log.info("Make sure the model exists on HuggingFace Hub")
        raise

    # titles.tsv 로드
    log.info(f"Loading titles from: {titles_path}")
    if not os.path.exists(titles_path):
        raise FileNotFoundError(f"titles.tsv not found at {titles_path}")

    titles_df = pd.read_csv(titles_path, sep="\t")
    log.info(f"Loaded {len(titles_df)} titles")

    # genres.tsv 로드
    item_genres = load_genres(genres_path)

    # 결합 텍스트 생성
    log.info("Creating combined title + genre texts...")
    combined_texts = []
    item_ids = []

    for _, row in titles_df.iterrows():
        item_id = row["item"]
        title = row["title"]
        genres = item_genres.get(item_id, [])

        combined_text = create_combined_text(title, genres)
        combined_texts.append(combined_text)
        item_ids.append(item_id)

    # 샘플 출력
    log.info("\nExample combined texts:")
    for i in range(min(5, len(combined_texts))):
        log.info(f"  [{item_ids[i]}] {combined_texts[i]}")
    log.info("")

    # Embedding 생성
    log.info("Generating embeddings...")
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(combined_texts), batch_size), desc="Processing batches"):
            batch_texts = combined_texts[i : i + batch_size]

            # Tokenize
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            # Forward pass
            outputs = model(**encoded)

            # Mean pooling
            batch_embeddings = mean_pooling(outputs, encoded["attention_mask"])

            # L2 Normalize (recommended for similarity tasks)
            batch_embeddings = torch.nn.functional.normalize(
                batch_embeddings, p=2, dim=1
            )

            # Convert to numpy
            batch_embeddings = batch_embeddings.cpu().numpy()
            all_embeddings.append(batch_embeddings)

    # Concatenate all embeddings
    embeddings = np.vstack(all_embeddings)
    embedding_dim = embeddings.shape[1]

    log.info(f"Generated embeddings shape: {embeddings.shape}")
    log.info(f"Embedding dimension: {embedding_dim}")

    # TSV 파일로 저장
    log.info(f"Saving embeddings to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        # Header
        f.write("item\ttitle\n")

        # Write each item
        for item_id, emb in zip(item_ids, embeddings):
            emb_str = " ".join(map(str, emb))
            f.write(f"{item_id}\t{emb_str}\n")

    log.info("Saved successfully!")

    # Summary
    log.info("=" * 80)
    log.info("Title + Genre Embedding Generation Complete!")
    log.info(f"  Model: {model_name}")
    log.info(f"  Total items: {len(item_ids)}")
    log.info(f"  Embedding dimension: {embedding_dim}")
    log.info(f"  Items with genres: {sum(1 for iid in item_ids if iid in item_genres)}")
    log.info(f"  Items without genres: {sum(1 for iid in item_ids if iid not in item_genres)}")
    log.info(f"  Output file: {output_path}")
    log.info("=" * 80)

    return embedding_dim, len(item_ids)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate title + genre embeddings for BERT4Rec using mxbai-embed-large-v1"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="~/data/train",
        help="Directory containing titles.tsv and genres.tsv (default: ~/data/train)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output file path (default: {data_dir}/title_embeddings/titles.tsv)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="mixedbread-ai/mxbai-embed-large-v1",
        help="Hugging Face model name (default: mixedbread-ai/mxbai-embed-large-v1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum token length (default: 128)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--backup-original",
        action="store_true",
        help="Backup original titles.tsv before overwriting",
    )

    args = parser.parse_args()

    # Expand paths
    data_dir = os.path.expanduser(args.data_dir)
    titles_path = os.path.join(data_dir, "titles.tsv")
    genres_path = os.path.join(data_dir, "genres.tsv")

    if args.output_path is None:
        # Default: save to title_embeddings subdirectory
        output_dir = os.path.join(data_dir, "title_embeddings")
        output_path = os.path.join(output_dir, "titles.tsv")
    else:
        output_path = os.path.expanduser(args.output_path)

    # Backup original if requested (only when overwriting)
    if args.backup_original and output_path == titles_path:
        backup_path = titles_path + ".backup"
        log.info(f"Backing up original to: {backup_path}")
        import shutil
        shutil.copy2(titles_path, backup_path)

    # Generate embeddings
    try:
        generate_title_genre_embeddings(
            titles_path=titles_path,
            genres_path=genres_path,
            output_path=output_path,
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )
    except Exception as e:
        log.error(f"Error during embedding generation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
