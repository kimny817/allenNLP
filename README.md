## Wine Rating Predictor

This is a simple neural wine rating predictor written with allennlp.

### Requirements

* python (3.6)
* [allennlp](https://github.com/allenai/allennlp/) (0.8.2)

### Model
This model treats each sentence as a sequence of words. We encode the sequence using a bidirectional LSTM with 2 hidden layers, 50 dimensional character embeddings (input), 150 dimensional hidden size, and dropout of 0.25.

### Scoring
A cautionary note is in order. The pytorch model optimizes for _word level_ F1 score, but it is more common to measure
on the _word level_. So, after training a model, get a comparable score using `word_eval.py` (which I copied from [here](https://gitlab.com/raymondhs/char-rnn-truecase/blob/master/word_eval.py))

For example, to score on the Wiki test data:

```bash
$ allennlp predict wiki-truecaser-model.tar.gz data/data.v1.split/normal.testing.txt --use-dataset-reader --output-file out_preds.txt --include-package mylib --predictor truecaser-predictor
$ python word_eval.py data/data.v1.split/normal.testing.txt out_preds.txt
```

### Usage

If you just want to predict, you can run:
```bash
$ allennlp predict wiki-truecaser-model.tar.gz data/test.txt --output-file test-out.txt --include-package mylib --use-dataset-reader --predictor truecaser-predictor
```

Where `data/test.txt` is a file with one sentence per line.

See `example.py` for an example of how to use it programmatically.


#### Training
The dataset reader requires text that has one sentence per line. The model expects tokenized text. If your text is already tokenized
(the Wiki data is), then you can use `just_spaces` as the `word_splitter` in the config. If you want to tokenize text first,
you can use `spacy`.

You can get the Wikipedia data by running:
```bash
$ cd data
$ ./get_data.sh
```

Run:
```bash
$ allennlp train truecaser.json --include-package mylib -s /path/to/save/model/
```

If you have a GPU, set `cuda_device` to 0 in `truecaser.json`. This will make training much faster.
