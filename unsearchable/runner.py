#!/usr/bin/env python3
"""Interactive runner that only returns the local NN's top answer."""

from __future__ import annotations

from pathlib import Path

from neural_net import load_model, predict


APP_DIR = Path(__file__).resolve().parent
UNKNOWN_ANSWER = "I am not sure yet. Add an example to dataset.txt so I can answer this."


def main() -> None:
    try:
        model = load_model(APP_DIR)
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Could not load the unsearchable model: {error}")
        print("Train it with: python3 trainer.py")
        return

    print("Unsearchable Mid AI. Replies come only from its local trained answers.")
    print("It always returns the highest-confidence class. Press Ctrl+C to stop.")
    try:
        while True:
            message = input("\nInput: ").strip()
            if not message:
                continue
            response, confidence = predict(model, message)
            if response == "__SEARCH__":
                response = UNKNOWN_ANSWER
            print(f"Output: {response}")
            print(f"NN score: {confidence:.1%}")
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")


if __name__ == "__main__":
    main()
