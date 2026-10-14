"""Optional character-level LSTM for text generation experiments."""

import argparse
import json
import string
from pathlib import Path

import numpy as np

VOCABULARY = string.ascii_lowercase + string.punctuation + string.digits + " \n"


def simplify_text(text, vocabulary=VOCABULARY):
    allowed = set(vocabulary)
    return "".join(character for character in text.lower() if character in allowed)


def one_hot_sentence(sentence, vocabulary=VOCABULARY):
    mapping = {character: index for index, character in enumerate(vocabulary)}
    encoded = np.zeros((len(sentence), len(vocabulary)), dtype=np.float32)
    for index, character in enumerate(sentence):
        if character not in mapping:
            raise ValueError(f"Character outside vocabulary: {character!r}")
        encoded[index, mapping[character]] = 1
    return encoded


def sample_index(probabilities, temperature, rng):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or probabilities.sum() <= 0
    ):
        raise ValueError("Expected finite, nonnegative probabilities with positive total")
    logits = np.log(np.maximum(probabilities, 1e-12)) / temperature
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    return int(rng.choice(len(weights), p=weights))


def keras_module():
    try:
        import tensorflow as tf
    except ImportError as error:
        raise ValueError("Install the generation extra: pip install -e '.[generation]'") from error
    return tf


def build_model(chunk_length, learning_rate=0.0005, units=100):
    tf = keras_module()
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(chunk_length, len(VOCABULARY))),
            tf.keras.layers.LSTM(units),
            tf.keras.layers.Dense(len(VOCABULARY), activation="softmax"),
        ]
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
    )
    return model


def train(
    corpus,
    output,
    chunk_length=40,
    step=3,
    epochs=3,
    batch_size=64,
    max_characters=200000,
    seed=42,
    units=100,
):
    if min(chunk_length, step, epochs, batch_size, max_characters, units) <= 0:
        raise ValueError("Training sizes, step, and epochs must be positive")
    tf = keras_module()
    tf.keras.utils.set_random_seed(seed)
    text = simplify_text(Path(corpus).read_text(encoding="utf-8")[:max_characters])
    if len(text) <= chunk_length:
        raise ValueError("Corpus must exceed the context length after filtering")
    mapping = {character: index for index, character in enumerate(VOCABULARY)}
    indices = np.array([mapping[c] for c in text], dtype=np.int32)
    starts = np.arange(0, len(indices) - chunk_length, step)

    def examples():
        for start in starts:
            yield indices[start : start + chunk_length], indices[start + chunk_length]

    dataset = tf.data.Dataset.from_generator(
        examples,
        output_signature=(tf.TensorSpec((chunk_length,), tf.int32), tf.TensorSpec((), tf.int32)),
    ).map(lambda x, y: (tf.one_hot(x, len(VOCABULARY)), y))
    # Encode only a batch at a time instead of allocating the full one-hot corpus.
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(len(starts)))
    dataset = dataset.shuffle(min(len(starts), 10000), seed=seed).batch(batch_size).prefetch(1)
    model = build_model(chunk_length, units=units)
    history = model.fit(dataset, epochs=epochs, shuffle=False, verbose=0)
    output = Path(output)
    if output.suffix != ".keras":
        raise ValueError("Generator output must end in .keras")
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    metadata = {
        "vocabulary": VOCABULARY,
        "chunk_length": chunk_length,
        "step": step,
        "seed": seed,
        "epochs": epochs,
        "corpus_characters": len(text),
        "training_chunks": len(starts),
        "loss": history.history["loss"],
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def generate(path, seed_text="the government", length=400, temperature=0.5, random_seed=42):
    if length < 0:
        raise ValueError("Generation length must be nonnegative")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    tf = keras_module()
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata["vocabulary"] != VOCABULARY:
        raise ValueError("Generator vocabulary does not match this release")
    model = tf.keras.models.load_model(path, compile=False)
    chunk = metadata["chunk_length"]
    seed_text = simplify_text(seed_text)
    context = seed_text[-chunk:].rjust(chunk)
    output = seed_text
    rng = np.random.default_rng(random_seed)
    for _ in range(length):
        probabilities = model(one_hot_sentence(context)[None, ...], training=False).numpy()[0]
        character = VOCABULARY[sample_index(probabilities, temperature, rng)]
        output += character
        context = context[1:] + character
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Character language model")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--corpus", type=Path, required=True)
    training.add_argument("--output", type=Path, default=Path("models/generator.keras"))
    for flag, default in (
        ("chunk-length", 40),
        ("step", 3),
        ("epochs", 3),
        ("batch-size", 64),
        ("max-characters", 200000),
        ("seed", 42),
    ):
        training.add_argument("--" + flag, type=int, default=default)
    sampling = commands.add_parser("sample")
    sampling.add_argument("--model", type=Path, required=True)
    sampling.add_argument("--seed-text", default="the government")
    sampling.add_argument("--length", type=int, default=400)
    sampling.add_argument("--temperature", type=float, default=0.5)
    sampling.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            values = vars(args).copy()
            values.pop("command")
            print(json.dumps(train(**values), indent=2))
        else:
            print("SYNTHETIC TEXT — character-model experiment; not factual news")
            print(
                generate(
                    args.model, args.seed_text, args.length, args.temperature, args.random_seed
                )
            )
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
