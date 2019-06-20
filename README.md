# Named Entity Recognition for slavic laguages using pretrained embeddings, attention mechanism and NCRF
# Instructions

## 0. Refer to:
[nert-bert](https://github.com/sberbank-ai/ner-bert)

<i>(OLD), unused:</i>
## 1. Loading a TensorFlow checkpoint (e.g. [Google's pre-trained models](https://github.com/google-research/bert#pre-trained-models))

You can convert any TensorFlow checkpoint for BERT (in particular [the pre-trained models released by Google](https://github.com/google-research/bert#pre-trained-models)) in a PyTorch save file by using the [`convert_tf_checkpoint_to_pytorch.py`](convert_tf_checkpoint_to_pytorch.py) script.

This script takes as input a TensorFlow checkpoint (three files starting with `bert_model.ckpt`) and the associated configuration file (`bert_config.json`), and creates a PyTorch model for this configuration, loads the weights from the TensorFlow checkpoint in the PyTorch model and saves the resulting model in a standard PyTorch save file that can be imported using `torch.load()`.

You only need to run this conversion script **once** to get a PyTorch model. You can then disregard the TensorFlow checkpoint (the three files starting with `bert_model.ckpt`) but be sure to keep the configuration file (`bert_config.json`) and the vocabulary file (`vocab.txt`) as these are needed for the PyTorch model too.

To run this specific conversion script you will need to have TensorFlow and PyTorch installed (`pip install tensorflow`). The rest of the repository only requires PyTorch.

Here is an example of the conversion process for a pre-trained `BERT-Base Uncased` model:

```shell
export BERT_BASE_DIR=/path/to/bert/multilingual_L-12_H-768_A-12

python3 convert_tf_checkpoint_to_pytorch.py \
    --tf_checkpoint_path $BERT_BASE_DIR/bert_model.ckpt \
    --bert_config_file $BERT_BASE_DIR/bert_config.json \
    --pytorch_dump_path $BERT_BASE_DIR/pytorch_model.bin
```

You can download Google's pre-trained models for the conversion [here](https://github.com/google-research/bert#pre-trained-models).

There is used the [BERT-Cased, Multilingual](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip) (recommended) in this solution.

## 2. Installation, requirements, test

This code was tested on Python 3.6. The requirements are:

- PyTorch (>= 0.4.1)
- tqdm
- tensorflow (for convertion)

To install the dependencies:

````bash
pip install -r ./requirements.txt
````

## 3. Usage
Solution located in notebooks: [slavic.ipynb](exps/slavic.ipynb) and [slavic.ipynb](exps/slavic_data_prc.ipynb).
There are two solutions:
1. model trained only on train set;
2. model trained on train and dev set.

You should run the first cell in notebook before all.

### 3.1 Data preparation
For train models you should run section `0. Parse data` in notebook.
### 3.2 Learn model on train data
In section  you should specify your own paths:
* train_path - path to train.csv file;
* valid_path - path to valid.csv file;
* vocab_file - path to google bert pretrained vocab;
* bert_config_file - path to google bert pretrained config;
* init_checkpoint_pt - path to google bert pretrained weights.

After that run section `1. Create dataloaders` with changed paths.

Run section `2. Create model`.

Before run section `3. Create learner` you should specify argument `best_model_path` (as your own path).

Run section `4. Learn your NER model`.

For get results on dev set run sections `5. Evaluate` and `6. To needle format` (don't forget change paths for dev dataset and parsed dev dataset).

I obtained the following results by your script:

```Binary classification quality (f1-score): 0.9583778014941302```
```Gapping resolution quality (symbol-wise f-measure): 0.9576077060524616```

### 3.3 Make prediction for model trained only on train set
Run section `6. To needle format` (if didn't run on prev step).

For get test predictions run section `7. Make prediction` (don't forget change paths for dev dataset and parsed dev dataset).

### 3.4 Make prediction for model trained on train and dev set
Run section `8. Merge train and dev` (don't forget change paths for dev dataset and parsed dev dataset).

Run sections `9. Train full`, `9.2. Create model`, `9.3. Create learner` (don't forget change paths for dev dataset and parsed dev dataset).

After that call only one (you can call all cells) cell with code `learner.fit(num_epochs, target_metric='f1')`.

For get test prediction on  run section `9.5 Make prediction with model trained on full data` (don't forget change paths for dev dataset and parsed dev dataset).
