"""Text-pair neural network with four hidden layers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np


FEATURES = 2048
HIDDEN_SIZES = (8192, 8192, 4096, 619)
DEFAULT_PARAMETER_COUNT = 120_000_000
MODEL_FORMAT = 3
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def count_parameters(features: int, hidden_sizes: tuple[int, ...], outputs: int) -> int:
    layer_sizes = (features, *hidden_sizes, outputs)
    return sum(left * right + right for left, right in zip(layer_sizes, layer_sizes[1:]))


def architecture_for_parameters(
    target_parameters: int, outputs: int
) -> tuple[int, tuple[int, ...], int]:
    """Scale the default four-hidden-layer shape to fit a parameter budget."""
    if target_parameters < 1 or outputs < 1:
        raise ValueError("Parameter count and output count must be positive.")

    minimum_hidden = (1,) * len(HIDDEN_SIZES)
    minimum = count_parameters(1, minimum_hidden, outputs)
    if target_parameters < minimum:
        raise ValueError(
            f"p is too small for {outputs} response classes; at least {minimum} parameters are needed."
        )

    base_count = count_parameters(FEATURES, HIDDEN_SIZES, outputs)
    high = max(1.0, math.sqrt(target_parameters / base_count) * 2)
    low = 0.0
    best_features = 1
    best_hidden = minimum_hidden
    best_count = minimum

    for _ in range(80):
        scale = (low + high) / 2
        features = max(1, round(FEATURES * scale))
        hidden = tuple(max(1, round(size * scale)) for size in HIDDEN_SIZES)
        actual = count_parameters(features, hidden, outputs)
        if actual <= target_parameters:
            low = scale
            best_features, best_hidden, best_count = features, hidden, actual
        else:
            high = scale

    return best_features, best_hidden, best_count


def _feature_index(name: str, feature_count: int) -> int:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % feature_count


def vectorize(text: str, feature_count: int = FEATURES) -> np.ndarray:
    """Build a normalized hashed bag of word, phrase, and character features."""
    if feature_count < 1:
        raise ValueError("Feature count must be positive.")
    tokens = TOKEN_RE.findall(text.casefold())
    normalized = " ".join(tokens)
    counts: dict[int, float] = {}

    def add(feature: str, weight: float = 1.0) -> None:
        index = _feature_index(feature, feature_count)
        counts[index] = counts.get(index, 0.0) + weight

    for token in tokens:
        add("word:" + token)
    for left, right in zip(tokens, tokens[1:]):
        add("pair:" + left + " " + right, 1.4)

    padded = "^" + normalized + "$"
    for size in (3, 4):
        for start in range(max(0, len(padded) - size + 1)):
            add(f"char{size}:" + padded[start : start + size], 0.35)

    result = np.zeros(feature_count, dtype=np.float32)
    for index, count in counts.items():
        result[index] = count
    norm = float(np.linalg.norm(result))
    if norm:
        result /= norm
    return result


def _forward(
    x: np.ndarray, weights: list[np.ndarray], biases: list[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    activations = [x]
    preactivations = []
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        z = activations[-1] @ weight + bias
        preactivations.append(z)
        activations.append(np.maximum(z, 0) if index < len(weights) - 1 else z)
    return activations, preactivations


def _probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def train(
    examples: list[dict[str, str]],
    epochs: int,
    progress: Callable[[int, int], None] | None = None,
    target_parameters: int = DEFAULT_PARAMETER_COUNT,
) -> tuple[dict[str, object], float, float]:
    """Train a response-class network sized to the requested parameter budget."""
    if not examples:
        raise ValueError("No input/output examples were found in dataset.txt.")
    if epochs < 1:
        raise ValueError("Epoch count must be a positive integer.")

    outputs = list(dict.fromkeys(row["output"] for row in examples))
    output_ids = {output: index for index, output in enumerate(outputs)}
    features, hidden_sizes, parameter_count = architecture_for_parameters(
        target_parameters, len(outputs)
    )
    x = np.stack([vectorize(row["input"], features) for row in examples])
    y = np.asarray([output_ids[row["output"]] for row in examples], dtype=np.int64)
    n_samples = len(examples)

    rng = np.random.default_rng(2026)
    layer_sizes = (features, *hidden_sizes, len(outputs))
    weights = [
        (rng.standard_normal((left, right)) * np.sqrt(2 / left)).astype(np.float32)
        for left, right in zip(layer_sizes, layer_sizes[1:])
    ]
    biases = [np.zeros(size, dtype=np.float32) for size in layer_sizes[1:]]
    params = [*weights, *biases]
    moments = [np.zeros_like(param) for param in params]
    variances = [np.zeros_like(param) for param in params]
    beta1, beta2, learning_rate = 0.9, 0.999, 0.01

    for step in range(1, epochs + 1):
        activations, preactivations = _forward(x, weights, biases)
        probabilities = _probabilities(activations[-1])
        delta = probabilities
        delta[np.arange(n_samples), y] -= 1
        delta /= n_samples

        grad_weights = [np.empty_like(weight) for weight in weights]
        grad_biases = [np.empty_like(bias) for bias in biases]
        for layer_index in range(len(weights) - 1, -1, -1):
            grad_weights[layer_index] = activations[layer_index].T @ delta
            grad_biases[layer_index] = delta.sum(axis=0)
            if layer_index:
                delta = (delta @ weights[layer_index].T) * (preactivations[layer_index - 1] > 0)

        gradients = [*grad_weights, *grad_biases]
        for index, (param, grad) in enumerate(zip(params, gradients)):
            moments[index] = beta1 * moments[index] + (1 - beta1) * grad
            variances[index] = beta2 * variances[index] + (1 - beta2) * (grad * grad)
            moment_hat = moments[index] / (1 - beta1**step)
            variance_hat = variances[index] / (1 - beta2**step)
            param -= learning_rate * moment_hat / (np.sqrt(variance_hat) + 1e-8)
        if progress is not None:
            progress(step, epochs)

    activations, _ = _forward(x, weights, biases)
    probabilities = _probabilities(activations[-1])
    loss = float(-np.log(np.maximum(probabilities[np.arange(n_samples), y], 1e-12)).mean())
    accuracy = float(np.mean(np.argmax(probabilities, axis=1) == y))

    model: dict[str, object] = {
        "outputs": outputs,
        "weights": weights,
        "biases": biases,
        "features": features,
        "hidden_sizes": list(hidden_sizes),
        "parameter_count": parameter_count,
        "epochs": epochs,
    }
    return model, loss, accuracy


def save_model(model: dict[str, object], model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    weights = model["weights"]
    biases = model["biases"]
    arrays = {
        name: array
        for index, (weight, bias) in enumerate(zip(weights, biases))  # type: ignore[arg-type]
        for name, array in ((f"w{index}", weight), (f"b{index}", bias))
    }
    weights_path = model_dir / "model.npz"
    weights_tmp = model_dir / "model.npz.tmp"
    with weights_tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(weights_tmp, weights_path)

    metadata = {
        "format": MODEL_FORMAT,
        "features": model["features"],
        "hidden_sizes": model["hidden_sizes"],
        "parameter_count": model["parameter_count"],
        "outputs": model["outputs"],
        "epochs": model["epochs"],
    }
    metadata_tmp = model_dir / "model.json.tmp"
    metadata_tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(metadata_tmp, model_dir / "model.json")


def load_model(model_dir: Path) -> dict[str, object]:
    metadata_path = model_dir / "model.json"
    weights_path = model_dir / "model.npz"
    if not metadata_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError("No trained model found. Run trainer.py first.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != MODEL_FORMAT:
        raise ValueError("The saved model format is old. Run trainer.py to train the larger network.")

    layer_count = len(metadata["hidden_sizes"]) + 1
    with np.load(weights_path, allow_pickle=False) as saved:
        weights = [saved[f"w{index}"] for index in range(layer_count)]
        biases = [saved[f"b{index}"] for index in range(layer_count)]
    return {**metadata, "weights": weights, "biases": biases}


def predict(model: dict[str, object], text: str) -> tuple[str, float]:
    weights = model["weights"]
    biases = model["biases"]
    features = int(model.get("features", FEATURES))
    activations, _ = _forward(  # type: ignore[arg-type]
        vectorize(text, features)[None, :], weights, biases  # type: ignore[arg-type]
    )
    probabilities = _probabilities(activations[-1])[0]
    index = int(np.argmax(probabilities))
    return model["outputs"][index], float(probabilities[index])  # type: ignore[index]
