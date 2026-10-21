#!/usr/bin/env python3
"""Read dataset.txt and train the local text neural network."""

from __future__ import annotations

import sys
from pathlib import Path

from neural_net import FEATURES, HIDDEN_SIZES, save_model, train


APP_DIR = Path(__file__).resolve().parent
DATASET_FILE = APP_DIR / "dataset.txt"


def load_dataset(path: Path = DATASET_FILE) -> tuple[list[dict[str, str]], int]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    examples: list[dict[str, str]] = []
    current_input: str | None = None
    current_output: str | None = None
    epochs: int | None = None

    def finish_pair(line_number: int) -> None:
        nonlocal current_input, current_output
        if current_input is None:
            return
        if current_output is None:
            raise ValueError(f"Line {line_number}: input is missing its 'o = ...' output line.")
        examples.append({"input": current_input, "output": current_output})
        current_input = None
        current_output = None

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Line {line_number}: expected 'i = ...', 'o = ...', or 'epoch = ...'.")

        key, value = (part.strip() for part in line.split("=", 1))
        key = key.casefold()
        if key == "i":
            finish_pair(line_number)
            if not value:
                raise ValueError(f"Line {line_number}: input text cannot be empty.")
            current_input = value
        elif key == "o":
            if current_input is None:
                raise ValueError(f"Line {line_number}: 'o = ...' must follow an 'i = ...' line.")
            if current_output is not None:
                raise ValueError(f"Line {line_number}: this input already has an output.")
            if not value:
                raise ValueError(f"Line {line_number}: output text cannot be empty.")
            current_output = value
        elif key == "epoch":
            finish_pair(line_number)
            try:
                epochs = int(value)
            except ValueError as error:
                raise ValueError(f"Line {line_number}: epoch must be a positive integer.") from error
            if epochs < 1:
                raise ValueError(f"Line {line_number}: epoch must be a positive integer.")
        else:
            raise ValueError(f"Line {line_number}: unknown field '{key}'; use i, o, or epoch.")

    finish_pair(len(path.read_text(encoding="utf-8").splitlines()) + 1)
    if not examples:
        raise ValueError("dataset.txt has no input/output pairs.")
    if epochs is None:
        raise ValueError("Add a final line such as 'epoch = 1000' to dataset.txt.")
    return examples, epochs


def main() -> None:
    examples, epochs = load_dataset()
    output_count = len({row["output"] for row in examples})
    layer_sizes = (FEATURES, *HIDDEN_SIZES, output_count)
    parameter_count = sum(left * right + right for left, right in zip(layer_sizes, layer_sizes[1:]))
    print(f"Loaded {len(examples)} input/output pair(s) from {DATASET_FILE.name}.")
    print(f"Network: {' → '.join(map(str, layer_sizes[:-1]))} → {output_count} output class(es)")
    print(f"Trainable parameters: {parameter_count:,}")
    print(f"Training the four-hidden-layer network for {epochs} epoch(s)...")

    def show_progress(completed: int, total: int) -> None:
        sys.stdout.write(f"\rEpoch {completed:,}/{total:,}")
        sys.stdout.flush()

    model, loss, accuracy = train(examples, epochs, progress=show_progress)
    print()
    save_model(model, APP_DIR)
    print(f"Saved trained model to {APP_DIR / 'model.npz'}")
    print(f"Training accuracy: {accuracy:.1%}  |  loss: {loss:.4f}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
