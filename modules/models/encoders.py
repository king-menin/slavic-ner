from torch import nn
import torch
from . import bert_modeling


class BertEmbedder(nn.Module):

    # @property
    def get_config(self):
        config = {
            "name": "BertEmbedder",
            "params": {
                "bert_config_file": self.bert_config_file,
                "init_checkpoint_pt": self.init_checkpoint_pt,
                "freeze": self.is_freeze,
                "embedding_dim": self.embedding_dim,
                "use_cuda": self.use_cuda
            }
        }
        return config

    def __init__(self, model, bert_config_file, init_checkpoint_pt,
                 freeze=True, embedding_dim=768, use_cuda=True):
        super(BertEmbedder, self).__init__()
        self.bert_config_file = bert_config_file
        self.init_checkpoint_pt = init_checkpoint_pt
        self.is_freeze = freeze
        self.embedding_dim = embedding_dim
        self.model = model
        self.use_cuda = use_cuda

        if use_cuda:
            self.cuda()

    @classmethod
    def from_config(cls, config):
        return cls.create(**config)

    def forward(self, *batch):
        input_ids, input_mask, input_type_ids = batch[:3]
        all_encoder_layers, _ = self.model(input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
        return all_encoder_layers

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def get_n_trainable_params(self):
        pp = 0
        for p in list(self.parameters()):
            if p.requires_grad:
                num = 1
                for s in list(p.size()):
                    num = num * s
                pp += num
        return pp

    @classmethod
    def create(cls,
               bert_config_file, init_checkpoint_pt, embedding_dim=768, use_cuda=True, freeze=True):
        bert_config = bert_modeling.BertConfig.from_json_file(bert_config_file)
        model = bert_modeling.BertModel(bert_config)
        if use_cuda:
            device = torch.device("cuda")
            map_location = "cuda"
        else:
            map_location = "cpu"
            device = torch.device("cpu")
        model.load_state_dict(torch.load(init_checkpoint_pt, map_location=map_location))
        model = model.to(device)
        model = cls(model=model, embedding_dim=embedding_dim, use_cuda=use_cuda,
                    bert_config_file=bert_config_file, init_checkpoint_pt=init_checkpoint_pt, freeze=freeze)
        if freeze:
            model.freeze()
        return model


class BertBiLSTMEncoder(nn.Module):

    def __init__(self, embeddings,
                 hidden_dim=128, rnn_layers=1, dropout=0.5, use_cuda=True, bert_mode="weighted"):
        super(BertBiLSTMEncoder, self).__init__()
        self.embeddings = embeddings
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.use_cuda = use_cuda
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            self.embeddings.embedding_dim,
            hidden_dim // 2,
            rnn_layers, batch_first=True, bidirectional=True)
        if use_cuda:
            self.cuda()
        self.output_dim = hidden_dim
        self.hidden = None
        self.bert_mode = bert_mode
        if self.bert_mode == "weighted":
            self.bert_weights = nn.Parameter(torch.FloatTensor(12, embeddings.embedding_dim))
            self.bert_gamma = nn.Parameter(torch.FloatTensor(1, embeddings.embedding_dim))
        self.init_weights()

    def init_weights(self):
        if self.bert_mode == "weighted":
            nn.init.xavier_normal(self.bert_gamma)
            nn.init.xavier_normal(self.bert_weights)
        for param in self.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

    def forward(self, batch):
        input_mask = batch[1]
        all_encoder_layers = self.embeddings(*batch)
        if self.bert_mode == "last":
            output = all_encoder_layers[-1]
        else:
            all_encoder_layers = torch.stack([a * b for a, b in zip(all_encoder_layers, self.bert_weights)])
            output = self.bert_gamma * torch.sum(all_encoder_layers, dim=0)
        output = self.dropout(output)
        lens = input_mask.sum(-1)
        output = nn.utils.rnn.pack_padded_sequence(
            output, lens.tolist(), batch_first=True)
        output, self.hidden = self.lstm(output)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, self.hidden

    def get_n_trainable_params(self):
        pp = 0
        for p in list(self.parameters()):
            if p.requires_grad:
                num = 1
                for s in list(p.size()):
                    num = num * s
                pp += num
        return pp

    @classmethod
    def create(cls, bert_config_file, init_checkpoint_pt, embedding_dim=768,
               bert_mode="weighted", freeze=True,
               hidden_dim=128, rnn_layers=1, dropout=0.5, use_cuda=True):
        if bert_mode not in ["weighted", "last"]:
            raise NotImplemented
        embeddings = BertEmbedder.create(
            bert_config_file, init_checkpoint_pt, embedding_dim, use_cuda, freeze)
        model = cls(
            embeddings, hidden_dim, rnn_layers, dropout, use_cuda=use_cuda, bert_mode=bert_mode)
        return model
