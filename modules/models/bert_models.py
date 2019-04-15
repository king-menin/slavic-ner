import abc
import sys
from torch import nn
from .released_models import released_models
from .encoders import BertBiLSTMEncoder
from .decoders import AttnNCRFJointDecoder, AttnNCRFDecoder, NCRFDecoder


class NerModel(nn.Module, metaclass=abc.ABCMeta):

    """Base class for all Models"""
    def __init__(self, encoder, decoder, device="cuda:0"):
        super(NerModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        if device != "cpu":
            self.cuda(int(device.split(":")[1]))

    @abc.abstractmethod
    def forward(self, *batch):
        # return self.decoder(self.encoder(batch))
        raise NotImplementedError("abstract method forward must be implemented")

    @abc.abstractmethod
    def score(self, *batch):
        # return self.decoder.score(self.encoder(batch))
        raise NotImplementedError("abstract method score must be implemented")

    @abc.abstractmethod
    def create(self, *args):
        raise NotImplementedError("abstract method create must be implemented")

    def get_n_trainable_params(self):
        pp = 0
        for p in list(self.parameters()):
            if p.requires_grad:
                num = 1
                for s in list(p.size()):
                    num = num * s
                pp += num
        return pp

    def get_config(self):
        try:
            config = {
                "name": self.__class__.__name__,
                "params": {
                    "encoder": self.encoder.get_config(),
                    "decoder": self.decoder.get_config(),
                    "device": self.device
                }
            }
        except AttributeError:
            config = {}
            print("config is empty :(. Maybe for this model from_config has not implemented yet.", file=sys.stderr)
        except NotImplemented:
            config = {}
            print("config is empty :(. Maybe for this model from_config has not implemented yet.", file=sys.stderr)
        return config

    @classmethod
    def from_config(cls, config):
        encoder = released_models["encoder"].from_config(**config["encoder"]["params"])
        decoder = released_models["decoder"].from_config(**config["decoder"]["params"])
        return cls(encoder, decoder, config["device"])


class BertBiLSTMAttnNCRFJoint(NerModel):

    def forward(self, batch):
        output, _ = self.encoder(batch)
        return self.decoder(output, batch[-3])

    def score(self, batch):
        output, _ = self.encoder(batch)
        # labels_mask, labels, cls_ids
        return self.decoder.score(output, batch[-3], batch[-2], batch[-1])

    @classmethod
    def create(cls,
               label_size, intent_size,
               # BertEmbedder params
               bert_config_file, init_checkpoint_pt, embedding_dim=768, bert_mode="weighted",
               freeze=True,
               # BertBiLSTMEncoder params
               enc_hidden_dim=128, rnn_layers=1,
               # AttnNCRFDecoder params
               key_dim=64, val_dim=64, num_heads=3,
               input_dropout=0.5,
               # Global params
               device="cuda:0",
               # NCRFpp
               nbest=8):
        encoder = BertBiLSTMEncoder.create(
            bert_config_file, init_checkpoint_pt, embedding_dim, bert_mode, freeze,
            enc_hidden_dim, rnn_layers, input_dropout, device)
        decoder = AttnNCRFJointDecoder.create(
            label_size, encoder.output_dim, intent_size, input_dropout, key_dim, val_dim, num_heads, device,
            nbest=nbest)
        return cls(encoder, decoder, device)


class BertBiLSTMAttnNCRF(NerModel):

    def forward(self, batch):
        output, _ = self.encoder(batch)
        return self.decoder(output, batch[-2])

    def score(self, batch):
        output, _ = self.encoder(batch)
        return self.decoder.score(output, batch[-2], batch[-1])

    @classmethod
    def create(cls,
               label_size,
               # BertEmbedder params
               bert_config_file, init_checkpoint_pt, embedding_dim=768, bert_mode="weighted",
               freeze=True,
               # BertBiLSTMEncoder params
               enc_hidden_dim=128, rnn_layers=1,
               # AttnNCRFDecoder params
               key_dim=64, val_dim=64, num_heads=3,
               input_dropout=0.5,
               # Global params
               device="cuda:0",
               # NCRFpp
               nbest=8):
        encoder = BertBiLSTMEncoder.create(
            bert_config_file, init_checkpoint_pt, embedding_dim, bert_mode, freeze,
            enc_hidden_dim, rnn_layers, input_dropout, device)
        decoder = AttnNCRFDecoder.create(
            label_size, encoder.output_dim, input_dropout, key_dim, val_dim, num_heads, nbest, device)
        return cls(encoder, decoder, device)


class BertBiLSTMNCRF(NerModel):

    def forward(self, batch):
        output, _ = self.encoder(batch)
        return self.decoder(output, batch[-1])

    def score(self, batch):
        output, _ = self.encoder(batch)
        return self.decoder.score(output, batch[-2], batch[-1])

    @classmethod
    def create(cls,
               label_size,
               # BertEmbedder params
               bert_config_file, init_checkpoint_pt, embedding_dim=768, bert_mode="weighted",
               freeze=True,
               # BertBiLSTMEncoder params
               enc_hidden_dim=128, rnn_layers=1,
               input_dropout=0.5,
               output_dropout=0.4,
               # Global params
               device="cuda:0",
               # NCRFpp
               nbest=8):
        encoder = BertBiLSTMEncoder.create(
            bert_config_file, init_checkpoint_pt, embedding_dim, bert_mode, freeze,
            enc_hidden_dim, rnn_layers, input_dropout, device)
        decoder = NCRFDecoder.create(
            label_size, encoder.output_dim, output_dropout, nbest)
        return cls(encoder, decoder, device)
