import torch
from torch.nn import functional
from torch import nn
from .layers import Linears, MultiHeadAttention
from .ncrf import NCRF


class PoolingLinearClassifier(nn.Module):
    """Create a linear classifier with pooling."""

    def __init__(self, input_dim, intent_size, input_dropout=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.intent_size = intent_size
        self.input_dropout = input_dropout
        self.dropout = nn.Dropout(p=input_dropout)
        self.linear = Linears(input_dim * 3, intent_size, [input_dim // 2], activation="relu")

    @staticmethod
    def pool(x, bs, is_max):
        """Pool the tensor along the seq_len dimension."""
        f = functional.adaptive_max_pool1d if is_max else functional.adaptive_avg_pool1d
        return f(x.permute(1, 2, 0), (1,)).view(bs, -1)

    def forward(self, output):
        output = self.dropout(output).transpose(0, 1)
        sl, bs, _ = output.size()
        avgpool = self.pool(output, bs, False)
        mxpool = self.pool(output, bs, True)
        x = torch.cat([output[-1], mxpool, avgpool], 1)
        return self.linear(x)


class AttnNCRFJointDecoder(nn.Module):
    def __init__(self,
                 crf, label_size, input_dim, intent_size, input_dropout=0.5,
                 key_dim=64, val_dim=64, num_heads=3, nbest=8):
        super(AttnNCRFJointDecoder, self).__init__()
        self.input_dim = input_dim
        self.attn = MultiHeadAttention(key_dim, val_dim, input_dim, num_heads, input_dropout)
        self.linear = Linears(in_features=input_dim,
                              out_features=label_size,
                              hiddens=[input_dim // 2])
        self.crf = crf
        self.label_size = label_size
        self.intent_size = intent_size
        self.intent_out = PoolingLinearClassifier(input_dim, intent_size, input_dropout)
        self.intent_loss = nn.CrossEntropyLoss()
        self.nbest = nbest

    def forward_model(self, inputs, labels_mask=None):
        batch_size, seq_len, input_dim = inputs.size()
        inputs, hidden = self.attn(inputs, inputs, inputs, labels_mask)
        intent_output = self.intent_out(inputs)
        output = inputs.contiguous().view(-1, self.input_dim)
        # Fully-connected layer
        output = self.linear.forward(output)
        output = output.view(batch_size, seq_len, self.label_size)
        return output, intent_output

    def forward(self, inputs, labels_mask):
        self.eval()
        logits, intent_output = self.forward_model(inputs)
        _, preds = self.crf._viterbi_decode_nbest(logits, labels_mask, self.nbest)
        preds = preds[:, :, 0]
        self.train()
        return preds, intent_output.argmax(-1)

    def score(self, inputs, labels_mask, labels, cls_ids):
        logits, intent_output = self.forward_model(inputs)
        crf_score = self.crf.neg_log_likelihood_loss(logits, labels_mask, labels) / logits.size(0)
        return crf_score + self.intent_loss(intent_output, cls_ids)

    @classmethod
    def create(cls, label_size, input_dim, intent_size, input_dropout=0.5, key_dim=64,
               val_dim=64, num_heads=3, device="cuda:0", nbest=8):
        return cls(NCRF(label_size, device), label_size + 2, input_dim, intent_size, input_dropout,
                   key_dim, val_dim, num_heads, nbest)


class AttnNCRFDecoder(nn.Module):
    def __init__(self,
                 crf, label_size, input_dim, input_dropout=0.5,
                 key_dim=64, val_dim=64, num_heads=3, nbest=8):
        super(AttnNCRFDecoder, self).__init__()
        self.input_dim = input_dim
        self.attn = MultiHeadAttention(key_dim, val_dim, input_dim, num_heads, input_dropout)
        self.linear = Linears(in_features=input_dim,
                              out_features=label_size,
                              hiddens=[input_dim // 2])
        self.nbest = nbest
        self.crf = crf
        self.label_size = label_size

    def forward_model(self, inputs, labels_mask=None):
        batch_size, seq_len, input_dim = inputs.size()
        inputs, _ = self.attn(inputs, inputs, inputs, labels_mask)

        output = inputs.contiguous().view(-1, self.input_dim)
        # Fully-connected layer
        output = self.linear.forward(output)
        output = output.view(batch_size, seq_len, self.label_size)
        return output

    def forward(self, inputs, labels_mask):
        self.eval()
        logits = self.forward_model(inputs)
        _, preds = self.crf._viterbi_decode_nbest(logits, labels_mask, self.nbest)
        # print(preds.shape)
        preds = preds[:, :, 0]
        self.train()
        return preds

    def score(self, inputs, labels_mask, labels):
        logits = self.forward_model(inputs)
        crf_score = self.crf.neg_log_likelihood_loss(logits, labels_mask, labels) / logits.size(0)
        return crf_score

    @classmethod
    def create(cls, label_size, input_dim, input_dropout=0.5, key_dim=64,
               val_dim=64, num_heads=3, device="cuda:0", nbest=8):
        return cls(NCRF(label_size, device), label_size + 2, input_dim, input_dropout,
                   key_dim, val_dim, num_heads, nbest)

    
class NCRFDecoder(nn.Module):

    # TODO: TRY TO FIX THIS SHIT (get attribute error)
    def get_config(self):
        config = {
            "name": "NCRFDecoder",
            "params": {
                "label_size": self.label_size,
                "input_dim": self.input_dim,
                "input_dropout": self.dropout.p,
                "nbest": self.nbest
            }
        }
        return config

    def __init__(self,
                 crf, label_size, input_dim, input_dropout=0.5, nbest=8):
        super(NCRFDecoder, self).__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(input_dropout)
        self.linear = Linears(in_features=input_dim,
                              out_features=label_size,
                              hiddens=[input_dim // 2])
        self.nbest = nbest
        self.crf = crf
        self.label_size = label_size

    def forward_model(self, inputs):
        batch_size, seq_len, input_dim = inputs.size()
        inputs = self.dropout(inputs)

        output = inputs.contiguous().view(-1, self.input_dim)
        # Fully-connected layer
        output = self.linear.forward(output)
        output = output.view(batch_size, seq_len, self.label_size)
        return output

    def forward(self, inputs, labels_mask):
        self.eval()
        logits = self.forward_model(inputs)
        _, preds = self.crf._viterbi_decode_nbest(logits, labels_mask, self.nbest)
        preds = preds[:, :, 0]
        self.train()
        return preds

    def score(self, inputs, labels_mask, labels):
        logits = self.forward_model(inputs)
        crf_score = self.crf.neg_log_likelihood_loss(logits, labels_mask, labels) / logits.size(0)
        return crf_score

    @classmethod
    def from_config(cls, config):
        return cls.create(**config)

    @classmethod
    def create(cls, label_size, input_dim, input_dropout=0.5, device="cuda:0", nbest=8):
        return cls(NCRF(label_size, device), label_size + 2, input_dim, input_dropout, nbest)
