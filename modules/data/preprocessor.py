from torch.utils.data import DataLoader
from modules.data import tokenization
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from modules.utils import read_json, save_json
import logging
import os
from modules.utils import if_none


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            # Bert data
            bert_tokens, input_ids, input_mask, input_type_ids,
            # Origin data
            tokens, labels, labels_ids, labels_mask, cls=None, cls_idx=None):
        """
        Data has the following structure.

        Parameters
        ----------
        bert_tokens : list[str]
            List of words that was tokenized by BERT tokenizer.
            Represented at data[0].
        input_ids: list[int]
            Encoded BERT tokens.
            Represented at data[1].
        input_mask : list[int]
            Mask of BERT tokens.
            Represented at data[2].
        input_type_ids : list[int]
            Segment mask of BERT tokens.
            Represented at data[3].
        tokens : list[str]
            Origin tokens.
        labels : list[str]
            Origin labels.
        labels_ids : list[int]
            Encoded origin labels.
            Represented at data[4].
        labels_mask : list[int]
            Mask of origin labels (not equally to input_mask).
            Represented at data[5].
        cls : str or None, optional (default=None)
            If not None cls is label of sample class (used in joint learning).
        cls_idx : int or None, optional (default=None)
            If not None cls_idx is encoded label of sample class (used in joint learning).
            If not None represented at data[6].
        """
        self.data = []
        # Bert data
        self.bert_tokens = bert_tokens
        self.input_ids = input_ids
        self.data.append(input_ids)
        self.input_mask = input_mask
        self.data.append(input_mask)
        self.input_type_ids = input_type_ids
        self.data.append(input_type_ids)
        # Origin data
        self.tokens = tokens
        self.labels = labels
        # Labels data
        self.labels_mask = labels_mask
        self.data.append(labels_mask)
        self.labels_ids = labels_ids
        self.data.append(labels_ids)
        # Used for joint model
        self.cls = cls
        self.cls_idx = cls_idx
        if cls is not None:
            self.data.append(cls_idx)


class NerDataLoader(DataLoader):

    def __init__(self, data_set, shuffle, cuda, **kwargs):
        super(NerDataLoader, self).__init__(
            dataset=data_set,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            **kwargs
        )
        self.cuda = cuda

    def collate_fn(self, data):
        res = []
        token_ml = max(map(lambda x_: sum(x_.data[0]), data))
        sorted_idx = np.argsort(list(map(lambda x_: sum(x_.data[1]), data)))[::-1]
        for idx in sorted_idx:
            f = data[idx]
            example = []
            for idx_, x in enumerate(f.data):
                if isinstance(x, list):
                    x = x[:token_ml]
                example.append(x)
            res.append(example)
        res_ = []
        for idx, x in enumerate(zip(*res)):
            res_.append(torch.LongTensor(x))
        if self.cuda:
            res_ = [t.cuda() for t in res_]
        return res_, sorted_idx


class BertNerData(object):

    def get_config(self):
        config = {
            "train_path": self.train_path,
            "valid_path": self.valid_path,
            "bert_vocab_file": self.bert_vocab_file,
            "bert_model_type": self.bert_model_type,
            "idx2label_path": self.idx2label_path,
            "idx2cls_path": self.idx2cls_path,
            "max_seq_len": self.max_seq_len,
            "batch_size": self.batch_size,
            "is_cls": self.is_cls,
            "pad": "<pad>",
            "use_cuda": self.use_cuda,
            "config_path": self.config_path,
            "shuffle": self.shuffle
        }
        return config

    def __init__(self, bert_vocab_file=None, train_path=None, valid_path=None, idx2label=None, config_path=None,
                 tokenizer=None,
                 bert_model_type="bert_cased", idx2cls=None, max_seq_len=424,
                 batch_size=16, is_cls=False,
                 idx2label_path=None, idx2cls_path=None, pad="<pad>", use_cuda=True, data_columns=["0", "1", "2"],
                 shuffle=True):
        """Store attributes in one cls. For more doc see BertNerData.create"""
        self.train_path = train_path
        self.valid_path = valid_path
        self.config_path = config_path
        self.bert_model_type = bert_model_type
        self.bert_vocab_file = bert_vocab_file
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        self.idx2label = idx2label
        if idx2label is not None:
            self.label2idx = {label: idx for idx, label in enumerate(idx2label)}

        self.idx2cls = idx2cls
        self.is_cls = is_cls
        if is_cls:
            self.cls2idx = {label: idx for idx, label in enumerate(idx2cls)}

        self.use_cuda = use_cuda

        self.pad = pad

        self.idx2label_path = idx2label_path
        self.idx2cls_path = idx2cls_path

        if is_cls and not idx2cls:
            raise ValueError("Must set idx2cls if run on classification mode.")

        self.train_dl = None
        self.valid_dl = None
        self.test_dl = None
        self.data_columns = data_columns

        self.shuffle = shuffle

    # TODO: write docs
    @classmethod
    def from_config(cls, config, clear_cache=True):
        """Read config and call create. For more docs, see BertNerData.create"""
        config = read_json(config)
        config["clear_cache"] = clear_cache
        return cls.create(**config)

    @classmethod
    def create(cls,
               bert_vocab_file, config_path=None, train_path=None, valid_path=None,
               idx2label=None, bert_model_type="bert_cased", idx2cls=None,
               max_seq_len=424,
               batch_size=16, is_cls=False,
               idx2label_path=None, idx2cls_path=None, pad="<pad>", use_cuda=True,
               clear_cache=True, data_columns=["0", "1", "2"], shuffle=True, dir_config=None):
        """
        Create or skip data loaders, load or create vocabs.
        DataFrame should has 2 or 3 columns. Structure see in data_columns description.

        Parameters
        ----------
        bert_vocab_file : str
            Path of vocabulary for BERT tokenizer.
        config_path : str, or None, optional (default=None)
            Path of config of BertNerData.
        train_path : str or None, optional (default=None)
            Path of train data frame. If not None update idx2label, idx2cls, idx2meta.
        valid_path : str or None, optional (default=None)
            Path of valid data frame. If not None update idx2label, idx2cls, idx2meta.
        idx2label : list or None, optional (default=None)
            Map form index to label.
        bert_model_type : str, optional (default="bert_cased")
            Mode of BERT model (CASED or UNCASED).
        idx2cls : list or None, optional (default=None)
            Map form index to cls.
        max_seq_len : int, optional (default=424)
            Max sequence length.
        batch_size : int, optional (default=16)
            Batch size.
        is_cls : bool, optional (default=False)
            Use joint model or single.
        idx2label_path : str or None, optional (default=None)
            Path to idx2label map. If not None and idx2label is None load idx2label.
        idx2cls_path : str or None, optional (default=None)
            Path to idx2cls map. If not None and idx2cls is None load idx2cls.
        pad : str, optional (default="<pad>")
            Padding token.
        use_cuda : bool, optional (default=True)
            Run model on gpu or cpu. If gpu pin tensors in data loaders to gpu.
        clear_cache : bool, optional (default=True)
            If True, rewrite all vocabs and BertNerData config.
        data_columns : list[str]
            Columns if pandas.DataFrame.
                data_columns[0] - represent labels column. Each label should be joined by space;
                data_columns[1] - represent tokens column. Input sequence should be tokenized and joined by space;
                data_columns[2] - represent cls column (if is_cls is not None).
        shuffle : bool, optional (default=True)
            Is shuffle data.
        dir_config : str or None, optional (default=None)
            Dir for store vocabs if paths is not set.

        Returns
        ----------
        data : BertNerData
            Created object of BertNerData.
        """
        idx2label_path = if_none(
            idx2label_path, os.path.join(dir_config, "idx2label.json") if dir_config is not None else None)

        if idx2label is None and idx2label_path is None:
            raise ValueError("Must set idx2label_path.")

        if bert_model_type == "bert_cased":
            do_lower_case = False
        elif bert_model_type == "bert_uncased":
            do_lower_case = True
        else:
            raise NotImplementedError("No requested mode :(.")

        tokenizer = tokenization.FullTokenizer(vocab_file=bert_vocab_file, do_lower_case=do_lower_case)

        if idx2label is None and os.path.exists(str(idx2label_path)) and not clear_cache:
            idx2label = read_json(idx2label_path)
        if is_cls:
            idx2cls_path = if_none(
                idx2cls_path, os.path.join(dir_config, "idx2cls.json") if dir_config is not None else None)
        if is_cls and idx2cls is None and os.path.exists(str(idx2cls_path)) and not clear_cache:
            idx2cls = read_json(idx2cls_path)

        config_path = if_none(
            config_path, os.path.join(dir_config, "data_ner.json") if dir_config is not None else None)

        data = cls(bert_vocab_file=bert_vocab_file, idx2label=idx2label, config_path=config_path,
                   tokenizer=tokenizer,
                   bert_model_type=bert_model_type, idx2cls=idx2cls, max_seq_len=max_seq_len,
                   batch_size=batch_size, is_cls=is_cls,
                   idx2label_path=idx2label_path, idx2cls_path=idx2cls_path,
                   pad=pad, use_cuda=use_cuda, data_columns=data_columns, shuffle=shuffle)

        if train_path is not None:
            _ = data.load_train_dl(train_path)

        if valid_path is not None:
            _ = data.load_valid_dl(valid_path)

        if clear_cache:
            data.save_vocabs_and_config()
        return data

    def save_vocabs_and_config(self, idx2label_path=None, idx2cls_path=None, config_path=None):
        logging.info("Saving vocabs...")
        save_json(self.idx2label, idx2label_path)
        save_json(self.idx2cls, idx2cls_path)
        save_json(self.get_config(), config_path)

    def load_train_dl(self, features):
        if isinstance(features, str):
            features, self.label2idx, self.cls2idx = self.load_df(features)
        self.train_dl = NerDataLoader(
            features, batch_size=self.batch_size, shuffle=self.shuffle, cuda=self.use_cuda)

        self.idx2label = sorted(self.label2idx, key=lambda x: self.label2idx[x])
        if self.is_cls:
            self.idx2cls = sorted(self.cls2idx, key=lambda x: self.cls2idx[x])

        return self.train_dl

    def load_valid_dl(self, features):
        if isinstance(features, str):
            features, self.label2idx, self.cls2idx = self.load_df(features)
        self.valid_dl = NerDataLoader(
            features, batch_size=self.batch_size, shuffle=self.shuffle, cuda=self.use_cuda)

        self.idx2label = sorted(self.label2idx, key=lambda x: self.label2idx[x])
        if self.is_cls:
            self.idx2cls = sorted(self.cls2idx, key=lambda x: self.cls2idx[x])

        return self.valid_dl

    def load_test_dl(self, features):
        if isinstance(features, str):
            features, _, _ = self.load_df(features)
        self.test_dl = NerDataLoader(
            features, batch_size=self.batch_size, shuffle=self.shuffle, cuda=self.use_cuda)

        return self.test_dl

    def load_df(self, df):
        if isinstance(df, str):
            df = pd.read_csv(df)
        tokenizer = self.tokenizer
        label2idx = self.label2idx
        cls2idx = self.cls2idx
        is_cls = self.is_cls
        if label2idx is None:
            label2idx = {self.pad: 0, '[CLS]': 1}
        features = []
        all_args = [df[self.data_columns[1]].tolist(), df[self.data_columns[0]].tolist()]
        if is_cls:
            # Use joint model
            if cls2idx is None:
                cls2idx = dict()
            all_args.append(df[self.data_columns[2]].tolist())
        # TODO: add chunks
        total = len(df[self.data_columns[0]].tolist())
        cls = None
        for args in tqdm(enumerate(zip(*all_args)), total=total, leave=False):
            if is_cls:
                idx, (text, labels, cls) = args
            else:
                idx, (text, labels) = args
            bert_tokens = []
            res_labels = []
            bert_tokens.append("[CLS]")
            res_labels.append("[CLS]")
            orig_tokens = str(text).split()
            labels = str(labels).split()
            pad_idx = label2idx[self.pad]
            assert len(orig_tokens) == len(labels)
            args = [orig_tokens, labels]
            for idx_, ars in enumerate(zip(*args)):
                orig_token, label = ars[:2]

                cur_tokens = tokenizer.tokenize(orig_token)
                if self.max_seq_len - 1 < len(bert_tokens) + len(cur_tokens):
                    break

                bert_tokens.extend(cur_tokens)
                res_labels.append(label)

            orig_tokens = ["[CLS]"] + orig_tokens

            input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
            for l in res_labels:
                if l not in label2idx:
                    label2idx[l] = len(label2idx)
            labels_ids = [label2idx[l] for l in res_labels]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended too.
            input_mask = [1] * len(input_ids)
            labels_mask = [1] * len(labels_ids)
            # Zero-pad up to the sequence length.
            while len(input_ids) < self.max_seq_len:
                input_ids.append(0)
                input_mask.append(0)
                labels_ids.append(pad_idx)
                labels_mask.append(0)
            input_type_ids = [0] * len(input_ids)
            while len(labels_ids) < self.max_seq_len:
                labels_ids.append(pad_idx)
                labels_mask.append(0)
            # For joint model
            cls_idx = None
            if is_cls:
                if cls not in cls2idx:
                    cls2idx[cls] = len(cls2idx)
                cls_idx = cls2idx[cls]
            features.append(InputFeatures(
                # Bert data
                bert_tokens=bert_tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
                # Origin data
                tokens=orig_tokens,
                labels=labels,
                labels_ids=labels_ids,
                labels_mask=labels_mask,
                # Joint data
                cls=cls,
                cls_idx=cls_idx
            ))

            assert len(input_ids) == len(input_mask)
            assert len(input_ids) == len(input_type_ids)
            assert len(input_ids) == len(labels_ids)
            assert len(input_ids) == len(labels_mask)
        return features, label2idx, cls2idx
