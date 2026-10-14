import numpy as np
import pytest

from fake_news.generation import (
    VOCABULARY,
    generate,
    one_hot_sentence,
    sample_index,
    simplify_text,
    train,
)


def test_one_hot_and_filtering():
    values = one_hot_sentence("abc")
    assert values.shape == (3, len(VOCABULARY))
    assert np.all(values.sum(axis=1) == 1)
    assert simplify_text("ABC🙂") == "abc"


def test_stable_temperature_sampling():
    rng = np.random.default_rng(42)
    assert sample_index([0, 1, 0], 0.001, rng) == 1
    with pytest.raises(ValueError):
        sample_index([0.5, 0.5], 0, rng)
    with pytest.raises(ValueError):
        sample_index([0, 0], 1, rng)


def test_lstm_train_save_reload_generate(tmp_path):
    pytest.importorskip("tensorflow")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the small robot moves its arm. " * 12)
    path = tmp_path / "generator.keras"
    report = train(
        corpus, path, chunk_length=8, step=8, epochs=1, max_characters=300, batch_size=16, units=8
    )
    assert report["training_chunks"] > 0
    assert np.isfinite(report["loss"]).all()
    output = generate(path, seed_text="The Robot🙂", length=20)
    assert output.startswith("the robot")
    assert len(output) == len("the robot") + 20
