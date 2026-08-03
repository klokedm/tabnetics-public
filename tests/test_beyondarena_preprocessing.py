from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tabnetics.datasets.beyondarena import (
    BeyondArenaPreprocessingProfile,
    apply_beyondarena_preprocessing,
    load_beyondarena_spec,
    load_beyondarena_splits,
)


FIXTURES = Path(__file__).parent / "fixtures" / "beyondarena"


def test_group_per_group_preprocessing_hashes_without_target_or_group_index() -> None:
    artifact = FIXTURES / "grouped_small" / "v1"
    spec = load_beyondarena_spec(artifact)
    split = load_beyondarena_splits(artifact).splits[0]
    frame = pd.DataFrame(
        {
            "group_id": ["A", "A", "B", "B", "C", "C"],
            "event_date": pd.date_range("2026-01-01", periods=6),
            "x_num": range(6),
            "target": [0, 1, 0, 1, 0, 1],
        }
    )

    processed = apply_beyondarena_preprocessing(frame, spec, split=split)

    assert "target" not in processed.X.columns
    assert "group_id" not in processed.X.columns
    assert processed.metadata["group_handling"] == "hash50"
    assert processed.metadata["group_hash_buckets"] == 50
    assert len([col for col in processed.X.columns if "__group_hash_" in col]) == 50
    assert "event_date__ordinal" in processed.X.columns
    assert "event_date" not in processed.X.columns


def test_text_and_high_cardinality_fallback_uses_train_only_categories() -> None:
    artifact = FIXTURES / "text_high_cardinality" / "v1"
    spec = load_beyondarena_spec(artifact)
    split = load_beyondarena_splits(artifact).splits[0]
    frame = pd.DataFrame(
        {
            "review_text": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"],
            "merchant_id": ["m1", "m2", "m3", "m4", "m5", "m6"],
            "x_num": [1, 2, 3, 4, 5, 6],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    profile = BeyondArenaPreprocessingProfile(high_cardinality_threshold=3)

    processed = apply_beyondarena_preprocessing(frame, spec, split=split, profile=profile)

    assert "review_text" not in processed.X.columns
    assert "review_text__text_len" in processed.X.columns
    assert "review_text__text_hash" in processed.X.columns
    tfidf_cols = [col for col in processed.X.columns if col.startswith("review_text__tfidf_hash_")]
    assert len(tfidf_cols) == 8
    assert processed.metadata["text_fallback"] == "tfidf_hash"
    assert processed.metadata["text_tfidf_hash_buckets"] == 8
    assert "target" not in processed.X.columns
    assert processed.metadata["high_cardinality_columns"] == ("merchant_id",)
    assert processed.metadata["max_categorical_cardinality"] == 4
    assert processed.X.loc[4, "merchant_id"] == -1.0
    assert processed.X.loc[5, "merchant_id"] == -1.0


def test_text_fallback_can_use_length_hash_ablation() -> None:
    artifact = FIXTURES / "text_high_cardinality" / "v1"
    spec = load_beyondarena_spec(artifact)
    split = load_beyondarena_splits(artifact).splits[0]
    frame = pd.DataFrame(
        {
            "review_text": ["alpha beta", "beta gamma", "gamma delta", "delta alpha", "epsilon", "zeta"],
            "merchant_id": ["m1", "m2", "m3", "m4", "m5", "m6"],
            "x_num": [1, 2, 3, 4, 5, 6],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    profile = BeyondArenaPreprocessingProfile(text_fallback="length_hash")

    processed = apply_beyondarena_preprocessing(frame, spec, split=split, profile=profile)

    assert "review_text__text_len" in processed.X.columns
    assert "review_text__text_hash" in processed.X.columns
    assert not any(col.startswith("review_text__tfidf_hash_") for col in processed.X.columns)
    assert processed.metadata["text_fallback"] == "length_hash"


def test_text_cache_appends_embeddings_and_rejects_target_leakage(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    artifact = FIXTURES / "text_high_cardinality" / "v1"
    spec = load_beyondarena_spec(artifact)
    frame = pd.DataFrame(
        {
            "review_text": ["alpha", "beta"],
            "merchant_id": ["m1", "m2"],
            "x_num": [1, 2],
            "target": [0, 1],
        }
    )
    cache = tmp_path / "text_cache.parquet"
    pd.DataFrame({"emb_0": [0.1, 0.2], "emb_1": [0.3, 0.4]}).to_parquet(cache)

    processed = apply_beyondarena_preprocessing(frame, spec, text_cache_path=cache)

    assert processed.metadata["text_cache_used"] is True
    assert "textcache__emb_0" in processed.X.columns
    leaky = tmp_path / "leaky_cache.parquet"
    pd.DataFrame({"target": [0, 1]}).to_parquet(leaky)
    with pytest.raises(ValueError, match="text cache contains forbidden target"):
        apply_beyondarena_preprocessing(frame, spec, text_cache_path=leaky)
