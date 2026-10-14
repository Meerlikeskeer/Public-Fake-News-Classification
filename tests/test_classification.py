import numpy as np
import pytest

from fake_news.analysis import analyze
from fake_news.data import Website, hostname, load_records, validate_splits, write_records
from fake_news.features import WebsiteFeatures, description_from_html, domain_features
from fake_news.model import Classifier, metrics


@pytest.fixture
def records():
    return [
        Website(
            f"site{i}.{'com' if i % 2 == 0 else 'news'}",
            '<meta name="description" content="'
            + ("sports finance report" if i % 2 == 0 else "scandal exposed rumor")
            + '">',
            i % 2,
        )
        for i in range(12)
    ]


def test_hostname_handles_paths_ports_and_case():
    assert hostname("HTTPS://WWW.Example.COM:443/story?q=1") == "example.com"
    assert domain_features("https://example.com/story")[0] == 1
    with pytest.raises(ValueError):
        hostname("file:///etc/hosts")
    with pytest.raises(ValueError, match="embedded credentials"):
        hostname("https://username:password@example.com")


@pytest.mark.parametrize("row", ['{"url": "example.com"}', "[]", "null"])
def test_invalid_dataset_schema_reports_validation_error(tmp_path, row):
    path = tmp_path / "bad.jsonl"
    path.write_text(row + "\n")
    with pytest.raises(ValueError, match="exactly url, html, and label"):
        load_records(path)


def test_description_handles_actual_open_graph_and_missing_content():
    assert description_from_html('<meta property="og:description" content="OG">') == "OG"
    assert (
        description_from_html(
            '<meta name="description"><meta name="description" content="fallback">'
        )
        == "fallback"
    )
    assert description_from_html("<p>No metadata</p>") == ""


@pytest.mark.parametrize("mode", ["domain", "keyword", "bow", "combined"])
def test_save_load_preserves_predictions_and_unknown_validation_tokens(records, tmp_path, mode):
    model = Classifier(mode).fit(records)
    unseen = [Website("unseen.org", '<meta name="description" content="alienwords">', 0)]
    before = model.predict(unseen)
    path = tmp_path / "model.joblib"
    model.save(path)
    after = Classifier.load(path).predict(unseen)
    assert before == after
    assert 0 <= after[0]["fake_probability"] <= 1
    if model.features.vectorizer_ is not None:
        assert "alienwords" not in model.features.vectorizer_.vocabulary_


@pytest.mark.parametrize("mode", ["glove", "combined"])
def test_embeddings_average_known_tokens_and_zero_unknown(tmp_path, mode):
    file = tmp_path / "vectors.txt"
    file.write_text("good 1 2\ngreat 3 4\nunseen 9 9\n")
    data = [Website("a.com", '<meta name="description" content="good great">', 0)]
    features = WebsiteFeatures(mode, embedding_path=file, dimension=2).fit(data)
    assert np.allclose(features.transform(data).toarray()[0, -2:], [2, 3])
    unknown = [Website("b.com", '<meta name="description" content="unseen">', 1)]
    assert np.allclose(features.transform(unknown).toarray()[0, -2:], [0, 0])
    assert "unseen" not in features.vectors_


@pytest.mark.parametrize("mode", ["glove", "combined"])
def test_saved_embedding_model_predicts_without_source_vectors(records, tmp_path, mode):
    vectors = tmp_path / "vectors.txt"
    vectors.write_text("sports 1 0\nfinance 1 1\nscandal -1 0\nexposed -1 -1\n")
    model = Classifier(mode, embedding_path=vectors, dimension=2).fit(records)
    expected = model.predict(records)
    path = tmp_path / "classifier.joblib"
    model.save(path)
    vectors.unlink()
    assert Classifier.load(path).predict(records) == expected


@pytest.mark.parametrize("mode", ["bow", "combined"])
def test_empty_descriptions_are_valid(mode):
    records = [Website("a.com", "", 0), Website("b.news", "", 1)]
    model = Classifier(mode).fit(records)
    assert len(model.predict(records)) == 2
    assert model.features.vectorizer_ is None


def test_overlap_and_bad_label_rejected(records):
    with pytest.raises(ValueError, match="hostname overlap"):
        validate_splits(
            records, [Website("https://www.site0.com/path", "", 0), Website("unique.news", "", 1)]
        )
    with pytest.raises(ValueError, match="Labels"):
        Website("a.com", "", "1")


def test_jsonl_roundtrip(records, tmp_path):
    file = tmp_path / "records.jsonl"
    write_records(records, file)
    assert load_records(file) == records


def test_fraction_undefined_zero_and_infinity():
    records = [Website("a.com", "", 0), Website("b.news", "", 1)]
    assert analyze(records, extension=".news")["fake_to_real_ratio"] == "infinity"
    assert analyze(records, extension=".com")["fake_to_real_ratio"] == 0
    assert analyze(records, extension=".edu")["fake_to_real_ratio"] is None


def test_metrics_use_fake_as_positive_and_matrix_label_order():
    result = metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["precision_fake"] == pytest.approx(2 / 3)
    assert result["recall_fake"] == 1
