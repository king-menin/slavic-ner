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

For detatiles see of lib see [nert-bert](https://github.com/sberbank-ai/ner-bert).
