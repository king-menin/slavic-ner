import json
import numpy


def if_none(arg, val):
    return val if arg is None else arg


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)


def jsonify(data):
    return json.dumps(data, cls=JsonEncoder)


def read_json(config):
    if isinstance(config, str):
        with open(config, "r") as f:
            config = json.load(f)
    return config


def save_json(config, path):
    with open(path, "w") as file:
        json.dump(config, file, cls=JsonEncoder)
