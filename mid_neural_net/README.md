# Mid AI

Mid uses your local neural network for ordinary replies. It does not use Ollama
or Qwen. The network turns the input text into features and selects one of the
reply texts learned from `dataset.txt`.

## Dataset format

Add `i` / `o` pairs, then finish with the training settings:

```text
i = hi
o = Hello!

i = how are you?
o = I'm doing well.

epoch = 1000
p = 120M
```

The trainer also accepts an `epoch = ...` line after individual pairs; the last
epoch value in the file is used for training the whole dataset. Text after the
first `=` is treated as the value.

Set `p` below `epoch` to choose the network's parameter budget. Values may be a
plain count (`200`) or use `K`, `M`, or `B` suffixes (`120M`, `1B`). The trainer
prints the actual size it builds and checks the available memory before training.

## Run Mid

```sh
cd ~/mid_neural_net
python3 runner.py
```

The runner loads `model.npz` and `model.json` and uses `neural_net.py` to produce
every ordinary chat reply. It does not need Ollama. Since this is a response
classifier, replies are selected from the output texts in the dataset; the network
does not write new sentences. Its displayed NN score is the classifier's softmax
score, not a calibrated measure that the answer is correct.

If fewer than 80% of the input's words appear together in any one training input,
the NN score is below 80%, or its feature-vector similarity to every training
input is below 30%, the runner automatically searches the web and prints the
results and source URLs. It tries DuckDuckGo's HTML and Lite pages, then Bing's
RSS results if those pages have no readable links. Search results are shown
directly; no second chat model is used to summarize them, and they are not
automatically added to training data. You can also use `/search your query` to
search explicitly.

For queries containing `code`, `program`, `script`, `example`, `snippet`, or
`calculator`, the runner also opens up to two public result pages and prints any
code blocks it can extract before the links. It displays source code as found; it
does not generate or execute that code.

`/run your command` shows the parsed command and asks you to confirm before
running it without a shell. The same confirmation is required when the NN
predicts a `/run ...` response. The included training examples let it suggest
read-only commands for listing files, checking the current directory, Python
version, Git status, disk space, and memory. Commands time out after 30 seconds
and output is capped at 12,000 characters. Press **Ctrl+C** to stop.

## Train the custom neural network

```sh
python3 trainer.py
```

This trains a four-hidden-layer classifier sized to `p` and saves `model.npz`
and `model.json`. `epoch` controls how many full training passes it makes. The
trainer reports the actual parameter count because the nearest smaller network
shape is used to stay within the requested budget. The runner only loads saved
weights; it does not start training when launched. Raw search results are not
fed into training automatically. `__SEARCH__` is a learned action class: when
the NN predicts it, the runner searches using the user's input.
