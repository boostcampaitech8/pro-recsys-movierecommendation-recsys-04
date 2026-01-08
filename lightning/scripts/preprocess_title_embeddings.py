"""
Title Text Embedding 전처리 스크립트

영화 제목을 사전 학습된 Sentence-BERT 모델로 인코딩하여 embedding을 생성합니다.
생성된 embedding은 BERT4Rec 모델에서 item representation 강화에 사용됩니다.

Usage:
    python scripts/preprocess_title_embeddings.py

    # 다른 모델 사용
    python scripts/preprocess_title_embeddings.py --model-name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

    # 다른 경로 지정
    python scripts/preprocess_title_embeddings.py --data-dir ~/data/train --output-dir ~/data/train/title_embeddings
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
import pickle
from pathlib import Path

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


def generate_title_embeddings(
    titles_path,
    output_dir,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    batch_size=32,
    max_length=64,
    device=None,
):
    """
    영화 제목을 BERT embedding으로 변환

    Args:
        titles_path: titles.tsv 파일 경로
        output_dir: 저장할 디렉토리
        model_name: Hugging Face 모델명
            - "sentence-transformers/all-MiniLM-L6-v2": 384-dim, 빠름, 영어 (추천)
            - "mixedbread-ai/mxbai-embed-large-v1": 1024-dim, MTEB 최상위
            - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384-dim, 다국어
            - "bert-base-uncased": 768-dim, 표준 BERT
        batch_size: 배치 크기 (GPU 메모리에 따라 조정)
        max_length: 최대 토큰 길이
        device: 'cuda', 'cpu', 또는 None (자동 선택)
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
        log.info("Trying to install sentence-transformers...")
        os.system("pip install sentence-transformers -q")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()

    # titles.tsv 로드
    log.info(f"Loading titles from: {titles_path}")
    if not os.path.exists(titles_path):
        raise FileNotFoundError(f"titles.tsv not found at {titles_path}")

    titles_df = pd.read_csv(titles_path, sep="\t")
    log.info(f"Loaded {len(titles_df)} titles")

    # Embedding 생성
    log.info("Generating embeddings...")
    item_ids = []
    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(titles_df), batch_size), desc="Processing batches"):
            batch_df = titles_df.iloc[i : i + batch_size]
            batch_item_ids = batch_df["item"].tolist()
            batch_titles = batch_df["title"].tolist()

            # Tokenize
            encoded = tokenizer(
                batch_titles,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            # Forward pass
            outputs = model(**encoded)

            # Mean pooling
            batch_embeddings = mean_pooling(outputs, encoded["attention_mask"])

            # Normalize (optional, but recommended for similarity tasks)
            batch_embeddings = torch.nn.functional.normalize(
                batch_embeddings, p=2, dim=1
            )

            # Convert to numpy and store
            batch_embeddings = batch_embeddings.cpu().numpy()

            item_ids.extend(batch_item_ids)
            embeddings.extend(batch_embeddings)

    # Dictionary로 변환 {item_id: embedding_vector}
    title_emb_dict = {item_id: emb for item_id, emb in zip(item_ids, embeddings)}

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # Save embeddings
    embeddings_path = os.path.join(output_dir, "title_embeddings.pkl")
    with open(embeddings_path, "wb") as f:
        pickle.dump(title_emb_dict, f)
    log.info(f"Saved embeddings to: {embeddings_path}")

    # Save metadata
    embedding_dim = embeddings[0].shape[0]
    metadata = {
        "model_name": model_name,
        "embedding_dim": embedding_dim,
        "num_items": len(item_ids),
        "max_length": max_length,
        "normalized": True,
    }
    metadata_path = os.path.join(output_dir, "metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    log.info(f"Saved metadata to: {metadata_path}")

    # Summary
    log.info("=" * 60)
    log.info("Title Embedding Generation Complete!")
    log.info(f"  Total items: {len(item_ids)}")
    log.info(f"  Embedding dimension: {embedding_dim}")
    log.info(f"  Model: {model_name}")
    log.info(f"  Output directory: {output_dir}")
    log.info("=" * 60)

    return title_emb_dict, metadata


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate title embeddings for BERT4Rec"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="~/data/train",
        help="Directory containing titles.tsv (default: ~/data/train)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for embeddings (default: {data_dir}/title_embeddings)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face model name (default: sentence-transformers/all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for encoding (default: 64)",
    )
    parser.add_argument(
        "--max-length", type=int, default=64, help="Maximum token length (default: 64)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)",
    )

    args = parser.parse_args()

    # Expand paths
    data_dir = os.path.expanduser(args.data_dir)
    titles_path = os.path.join(data_dir, "titles.tsv")

    if args.output_dir is None:
        output_dir = os.path.join(data_dir, "title_embeddings")
    else:
        output_dir = os.path.expanduser(args.output_dir)

    # Generate embeddings
    try:
        generate_title_embeddings(
            titles_path=titles_path,
            output_dir=output_dir,
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
