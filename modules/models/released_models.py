from .encoders import *
from .decoders import *


released_models = {
    "BertBiLSTMNCRF": {
        "encoder": BertBiLSTMEncoder,
        "decoder": NCRFDecoder
    },
    "BertBiLSTMAttnNCRFJoint": {
        "encoder": BertBiLSTMEncoder,
        "decoder": AttnNCRFJointDecoder
    },
    "BertBiLSTMAttnNCRF": {
        "encoder": AttnNCRFDecoder,
        "decoder": NCRFDecoder
    }
}
