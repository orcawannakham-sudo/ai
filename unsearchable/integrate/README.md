# C++ header export

`model_to_header.py` converts the trained model in `../model.npz` and
`../model.json` into a C++ header containing its layer sizes, response strings,
and row-major weight and bias arrays.

Run it from this folder:

```sh
python3 model_to_header.py
```

This writes `model_data.h`. You can choose another output path:

```sh
python3 model_to_header.py ./generated/my_model.h
```

The header uses C++17 `inline constexpr` values and places everything in the
`mid_ai_unsearchable` namespace. After retraining the Python model, rerun the
converter to refresh the header.

The header exports weights and model metadata. A C++ application still needs to
implement the forward pass and the same text feature pipeline used by
`../neural_net.py` before its predictions will match the Python runner.
