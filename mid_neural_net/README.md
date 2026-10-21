# Mid AI

Mid uses your local neural network for ordinary replies. It does not use Ollama
or Qwen. The network turns the input text into features and selects one of the
reply texts learned from `dataset.txt`.

## Dataset format

Add as many `i` / `o` pairs as you like. Put the epoch count on the final line:

```text
i = hi
o = Hello!

i = how are you?
o = I'm doing well.

epoch = 1000
```

The trainer also accepts an `epoch = ...` line after individual pairs; the last
epoch value in the file is used for training the whole dataset. Text after the
first `=` is treated as the value.

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
or the NN score is below 80%, the runner automatically searches the web and prints
the results and source URLs. It tries DuckDuckGo's HTML and Lite pages, then Bing's
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
predicts a `/run ...` response. Commands time out after 30 seconds and output
is capped at 12,000 characters. Press **Ctrl+C** to stop.

## Train the custom neural network

```sh
python3 trainer.py
```

This trains the four-hidden-layer classifier (8192, 8192, 4096, and 619 neurons)
and saves `model.npz` and `model.json`. With the current nine distinct dataset
responses, it has 120,002,615 trainable parameters. `epoch` controls how many
full training passes it makes. The Hello World code reply is a learned output
class with several matching prompts in `dataset.txt`; adding other code replies
also requires training. The runner only loads saved weights; it does not start
training when launched. Raw search results are not fed into training
automatically. `__SEARCH__` is a learned action class: when the NN predicts it,
the runner searches using the user's input.
