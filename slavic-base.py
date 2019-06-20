import warnings
import sys
import numpy as np
import torch

sys.path.append("../")

warnings.filterwarnings("ignore")


from modules import BertNerData as NerData
from modules.models.bert_models import BertBiLSTMAttnNMTJoint, BertBiLSTMAttnNCRFJoint, BertBiLSTMAttnNCRF
from modules import NerLearner
import os

data_dir = "~/ae/work/data/training_data_v10/processed/"
train_path = data_dir + "brexit_parsed.csv"
valid_path = data_dir + "asia_bibi_parsed.csv"
model_dir = "/home/eartemov/ae/work/models/multilingual_L-12_H-768_A-12"
# init_checkpoint_pt = os.path.join(model_dir, "pytorch_model.bin")
init_checkpoint_pt = "/home/eartemov/ae/work/models/bert-base-multilingual-cased.tar.gz"
bert_config_file = os.path.join(model_dir, "bert_config.json")
token = "96435fa287fbf7e469185f1062386e05a075cadbf6838b74da22bf64b080bc32.99bcd55fc66f4f3360bc49ba472b940b8dcf223ea6a345deb969d607ca900729"
vocab_file = "/home/eartemov/models/bert-base-multilingual-cased/" + token


if __name__ == "__main__":
    torch.cuda.set_device(0)
    data = NerData.create(train_path, valid_path, vocab_file, is_cls=False, batch_size=16)
    print(data.label2idx)
    print("data.tokenizer.vocab:", len(data.tokenizer.vocab))
    model = BertBiLSTMAttnNCRF.create(
       len(data.label2idx), bert_config_file, init_checkpoint_pt,
       enc_hidden_dim=1024, rnn_layers=1, num_heads=6, input_dropout=0.5, nbest=11)
    print(model)
    # model = BertBiLSTMAttnNMTJoint.create(
    #    len(data.label2idx), len(data.cls2idx), bert_config_file, init_checkpoint_pt,
    #    enc_hidden_dim=128, rnn_layers=1, dec_embedding_dim=32, dec_hidden_dim=128, input_dropout=0.5, nbest=11)
    # model = torch.nn.DataParallel(model, [2, 3])
    num_epochs = 150
    learner = NerLearner(model, data,
                         best_model_path="/home/eartemov/ae/work/models/AGRR-2019/slavic_without_clf.cpt",
                         lr=0.0001, clip=1.0, sup_labels=data.id2label[1:],
                         t_total=num_epochs * len(data.train_dl))
    learner.fit(num_epochs, target_metric='f1')