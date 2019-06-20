import warnings
import sys
import numpy as np
import torch

sys.path.append("../")

warnings.filterwarnings("ignore")


from modules import BertNerData as NerData
from modules.models.bert_models import BertBiLSTMAttnNMTJoint, BertBiLSTMAttnNCRFJoint
from modules import NerLearner
import os

data_dir = "/data/aaemeljanov/training_data_v10/processed/"
train_path = data_dir + "brexit_parsed.csv"
valid_path = data_dir + "asia_bibi_parsed.csv"
model_dir = "/home/aaemeljanov/models/multi_cased_L-12_H-768_A-12/"
init_checkpoint_pt = os.path.join(model_dir, "pytorch_model.bin")
bert_config_file = os.path.join(model_dir, "bert_config.json")
vocab_file = os.path.join(model_dir, "vocab.txt")


if __name__ == "__main__":
    torch.cuda.set_device(5)
    data = NerData.create(train_path, valid_path, vocab_file, is_cls=True, batch_size=16)
    print(data.label2idx)
    # model = BertBiLSTMAttnNCRFJoint.create(
    #      len(data.label2idx), len(data.cls2idx), bert_config_file, init_checkpoint_pt,
    #     enc_hidden_dim=1024, rnn_layers=1, num_heads=5, input_dropout=0.5, nbest=12)
    model = BertBiLSTMAttnNMTJoint.create(
        len(data.label2idx), len(data.cls2idx), bert_config_file, init_checkpoint_pt,
        enc_hidden_dim=512, rnn_layers=1, dec_embedding_dim=128, dec_hidden_dim=512, input_dropout=0.5, nbest=8)
    # model = torch.nn.DataParallel(model, [2, 3])
    num_epochs = 150
    learner = NerLearner(model, data,
                         best_model_path="/home/aaemeljanov/models/AGRR-2019/slavic-nmt.cpt",
                         lr=0.0001, clip=1.0, sup_labels=data.id2label[1:],
                         t_total=num_epochs * len(data.train_dl))
    learner.fit(num_epochs, target_metric='f1')