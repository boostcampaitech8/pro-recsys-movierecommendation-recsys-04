# src/models/content_recommender.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class TfidfContentRecommender:
    def __init__(self, min_df=3, max_df=0.9):
        self.min_df = min_df
        self.max_df = max_df

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words="english",
        )

        self.tfidf = None
        self.item_ids = None
        self.item_id_to_idx = None

    def fit(self, metadata_path: str):
        """
        metadata_path: item_content.csv (item, text)
        """
        df = pd.read_csv(metadata_path)

        assert "item" in df.columns
        assert "text" in df.columns

        self.item_ids = df["item"].values
        self.item_id_to_idx = {item: i for i, item in enumerate(self.item_ids)}

        self.tfidf = self.vectorizer.fit_transform(df["text"])

    def recommend(self, user_items, topk=20, exclude=None):
        """
        user_items: 사용자가 본 item list
        exclude: 제외할 item set
        """
        if self.tfidf is None:
            raise RuntimeError("Content model is not fitted. Call fit() first.")

        idxs = [self.item_id_to_idx[i] for i in user_items if i in self.item_id_to_idx]

        if not idxs:
            return []

        user_vec = np.asarray(self.tfidf[idxs].mean(axis=0))
        sims = cosine_similarity(user_vec, self.tfidf).flatten()

        ranked = sims.argsort()[::-1]

        results = []
        for idx in ranked:
            item = self.item_ids[idx]
            if exclude and item in exclude:
                continue
            if item not in user_items:
                results.append(item)
            if len(results) >= topk:
                break

        return results
