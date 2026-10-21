"""Interactive runner for the local Mid neural network."""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

from internet_search import search_web
from neural_net import load_model, predict, vectorize
from trainer import load_dataset


APP_DIR = Path(__file__).resolve().parent
MAX_COMMAND_LENGTH = 2_000
MAX_ARGUMENTS = 64
MAX_OUTPUT_LENGTH = 12_000
MIN_INPUT_WORD_COVERAGE = 0.80
MIN_INPUT_SIMILARITY = 0.30
# Softmax scores are not calibrated; search unless the model is fairly sure.
MIN_NN_SCORE = 0.80
SEARCH_ACTION = "__SEARCH__"
BLOCKED_EXECUTABLES = {"bash", "dash", "fish", "pkexec", "sh", "sudo", "su", "zsh"}


def run_command(command: str) -> None:
    """Run an explicitly requested command after showing it for confirmation."""
    if len(command) > MAX_COMMAND_LENGTH:
        print(f"Command is too long (limit: {MAX_COMMAND_LENGTH} characters).")
        return

    try:
        arguments = shlex.split(command)
    except ValueError as error:
        print(f"Could not parse command: {error}")
        return
    if not arguments:
        print("Usage: /run COMMAND")
        return
    if len(arguments) > MAX_ARGUMENTS:
        print(f"Command has too many arguments (limit: {MAX_ARGUMENTS}).")
        return
    if Path(arguments[0]).name.lower() in BLOCKED_EXECUTABLES:
        print("Direct shells and privilege-escalation commands are blocked.")
        return

    print(f"Command: {shlex.join(arguments)}")
    try:
        answer = input("Run this command? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nCommand cancelled.")
        return
    if answer not in {"y", "yes"}:
        print("Command cancelled.")
        return

    try:
        result = subprocess.run(
            arguments,
            cwd=APP_DIR,
            shell=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=30,
            check=False,
        )
    except FileNotFoundError:
        print(f"Command not found: {arguments[0]}")
        return
    except subprocess.TimeoutExpired:
        print("Command stopped after the 30-second time limit.")
        return
    except OSError as error:
        print(f"Could not run command: {error}")
        return

    output = "\n".join(part for part in (result.stdout.rstrip(), result.stderr.rstrip()) if part)
    if output:
        print(output[:MAX_OUTPUT_LENGTH])
        if len(output) > MAX_OUTPUT_LENGTH:
            print("\n[output truncated]")
    else:
        print("(no output)")
    print(f"Exit code: {result.returncode}")


def print_help() -> None:
    print("Type a message and the local NN will choose a reply from its dataset.")
    print("Inputs unlike the training examples automatically search the web.")
    print("/search QUERY  Search the web and show source snippets")
    print("/run COMMAND   Review and run a command")
    print("Ask for common system info or file-list commands in plain language.")
    print("/help          Show this help")
    print("Press Ctrl+C to stop.")


def main() -> None:
    try:
        model = load_model(APP_DIR)
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Could not load Mid's neural network: {error}")
        print("Train it with: python3 trainer.py")
        return

    try:
        examples, _, _, _ = load_dataset(APP_DIR / "dataset.txt")
        feature_count = int(model.get("features", 2048))
        training_words = [
            set(re.findall(r"\w+", example["input"].casefold())) for example in examples
        ]
        training_vectors = [
            vectorize(example["input"], feature_count) for example in examples
        ]
    except (FileNotFoundError, ValueError) as error:
        training_words = []
        training_vectors = []
        print(f"Warning: cannot read training inputs; automatic search fallback is off: {error}")

    print("Mid AI neural-network runner. Type a message; press Ctrl+C to stop.")
    print("Replies are selected by your local NN from the response texts in dataset.txt.")
    print("Unfamiliar inputs automatically search the web.")
    print("Use /help for the optional search and command controls.\n")

    while True:
        try:
            text = input("Input: ").strip()
            if not text:
                continue
            if text == "/help":
                print_help()
                continue
            if text == "/search" or text.startswith("/search "):
                query = text[len("/search") :].strip()
                try:
                    print(search_web(query))
                except (RuntimeError, ValueError) as error:
                    print(f"Search failed: {error}")
                continue
            if text == "/run" or text.startswith("/run "):
                run_command(text[len("/run") :].strip())
                continue

            reply, score = predict(model, text)
            word_coverage = None
            input_similarity = None
            input_words = set(re.findall(r"\w+", text.casefold()))
            if training_words and input_words:
                word_coverage = max(
                    len(input_words & example_words) / len(input_words)
                    for example_words in training_words
                )
            if training_vectors:
                input_vector = vectorize(text, int(model.get("features", 2048)))
                input_similarity = max(
                    float(example_vector @ input_vector) for example_vector in training_vectors
                )
            if reply == SEARCH_ACTION:
                print("Your NN chose to search the web.")
                try:
                    print(search_web(text))
                except (RuntimeError, ValueError) as error:
                    print(f"Search failed: {error}")
                print()
                continue

            is_command_reply = reply.startswith("/run ")
            if (
                score < MIN_NN_SCORE
                or (
                    not is_command_reply
                    and word_coverage is not None
                    and word_coverage < MIN_INPUT_WORD_COVERAGE
                )
                or (
                    not is_command_reply
                    and input_similarity is not None
                    and input_similarity < MIN_INPUT_SIMILARITY
                )
            ):
                details = [f"NN score {score:.1%}"]
                if word_coverage is not None:
                    details.append(f"training-word coverage {word_coverage:.1%}")
                if input_similarity is not None:
                    details.append(f"nearest-example similarity {input_similarity:.1%}")
                print(
                    "Input looks unfamiliar to the NN; searching the web. "
                    f"({', '.join(details)})"
                )
                try:
                    print(search_web(text))
                except (RuntimeError, ValueError) as error:
                    print(f"Search failed: {error}")
                    print(f"NN fallback: {reply} (score {score:.1%})")
                print()
                continue
            if is_command_reply:
                print("The NN suggested a command; review it before running:")
                run_command(reply[len("/run ") :])
                print()
                continue
            print(f"Output: {reply}")
            print(f"NN score: {score:.1%}\n")
        except (EOFError, KeyboardInterrupt):
            print("\nStopping Mid AI.")
            return


if __name__ == "__main__":
    main()
