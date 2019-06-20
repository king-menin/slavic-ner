# Named Entity Recognition for slavic laguages using pretrained embeddings, attention mechanism and NCRF
# Instructions

## 0. Refer to:
[nert-bert](https://github.com/sberbank-ai/ner-bert)

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
