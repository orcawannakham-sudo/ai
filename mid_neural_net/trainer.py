#!/usr/bin/env python3
"""Read dataset.txt and train the local text neural network."""

from __future__ import annotations

import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

from neural_net import (
    DEFAULT_PARAMETER_COUNT,
    architecture_for_parameters,
    save_model,
    train,
)


APP_DIR = Path(__file__).resolve().parent
DATASET_FILE = APP_DIR / "dataset.txt"
PARAMETER_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([kmb]?)$", re.IGNORECASE)
SUFFIX_MULTIPLIERS = {"": 1, "k": 1_000, "m": 1_000_000, "b": 1_000_000_000}


def parse_parameter_count(value: str) -> int:
    match = PARAMETER_RE.fullmatch(value.strip())
    if not match:
        raise ValueError("use a whole number or a K, M, or B suffix, such as 120M or 1B")
    try:
        count = int(Decimal(match.group(1)) * SUFFIX_MULTIPLIERS[match.group(2).casefold()])
    except InvalidOperation as error:
        raise ValueError("parameter count is not a valid number") from error
    if count < 1:
        raise ValueError("parameter count must be positive")
    return count


def load_dataset(
    path: Path = DATASET_FILE,
) -> tuple[list[dict[str, str]], int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    examples: list[dict[str, str]] = []
    current_input: str | None = None
    current_output: str | None = None
    epochs: int | None = None
    parameter_count: int | None = None

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
            raise ValueError(
                f"Line {line_number}: expected 'i = ...', 'o = ...', 'epoch = ...', or 'p = ...'."
            )

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
        elif key == "p":
            finish_pair(line_number)
            try:
                parameter_count = parse_parameter_count(value)
            except ValueError as error:
                raise ValueError(f"Line {line_number}: invalid p value; {error}.") from error
        else:
            raise ValueError(f"Line {line_number}: unknown field '{key}'; use i, o, epoch, or p.")

    finish_pair(len(path.read_text(encoding="utf-8").splitlines()) + 1)
    if not examples:
        raise ValueError("dataset.txt has no input/output pairs.")
    if epochs is None:
        raise ValueError("Add a final line such as 'epoch = 1000' to dataset.txt.")
    if parameter_count is None:
        parameter_count = DEFAULT_PARAMETER_COUNT
    return examples, epochs, parameter_count


def available_memory_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def main() -> None:
    examples, epochs, target_parameters = load_dataset()
    output_count = len({row["output"] for row in examples})
    features, hidden_sizes, parameter_count = architecture_for_parameters(
        target_parameters, output_count
    )
    layer_sizes = (features, *hidden_sizes, output_count)
    estimated_memory = parameter_count * 32 + 512 * 1024**2
    available_memory = available_memory_bytes()
    print(f"Loaded {len(examples)} input/output pair(s) from {DATASET_FILE.name}.")
    print(f"Network: {' → '.join(map(str, layer_sizes[:-1]))} → {output_count} output class(es)")
    print(f"Requested parameters: {target_parameters:,}")
    print(f"Actual parameters: {parameter_count:,}")
    print(f"Estimated training memory: about {estimated_memory / 1024**3:.1f} GiB")
    if available_memory is not None:
        print(f"Available memory: about {available_memory / 1024**3:.1f} GiB")
        if estimated_memory > available_memory:
            raise ValueError(
                "This model is too large for the currently available memory. "
                "Lower the p value in dataset.txt or free memory, then try again."
            )
    print(f"Training the four-hidden-layer network for {epochs} epoch(s)...")

    def show_progress(completed: int, total: int) -> None:
        sys.stdout.write(f"\rEpoch {completed:,}/{total:,}")
        sys.stdout.flush()

    model, loss, accuracy = train(
        examples,
        epochs,
        progress=show_progress,
        target_parameters=target_parameters,
    )
    print()
    save_model(model, APP_DIR)
    print(f"Saved trained model to {APP_DIR / 'model.npz'}")
    print(f"Training accuracy: {accuracy:.1%}  |  loss: {loss:.4f}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
