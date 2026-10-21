# Unsearchable Mid AI

This is a separate model that only predicts an answer from its local dataset.
It always prints the class with the highest NN score. It does not search the web
or run commands. Add examples to `dataset.txt`, then train it with:

```sh
cd ~/mid_neural_net/unsearchable
python3 trainer.py
```

Run the local model with:

```sh
python3 runner.py
```

The dataset maps requests that would trigger web search in the searchable model
to a local “I am not sure yet” answer. The displayed NN score is the classifier's
confidence for its selected class, not a guarantee that the answer is correct.

## Export for C++ projects

The [`integrate/`](integrate/README.md) folder contains a converter that exports
the trained weights and model metadata to a C++ header.
