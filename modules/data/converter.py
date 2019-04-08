from nltk import tokenize
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from modules.utils import if_none
from collections import defaultdict, Counter


class DataConverter(object):
    def __init__(
            self, root, save_dir_name="processed", annotated_prefix="annotated", raw_prefix="raw",
            other_cat="O", dropna=False):
        self.root = Path(root)
        self.save_dir_name = save_dir_name
        self.annotated_prefix = annotated_prefix
        self.raw_prefix = raw_prefix
        self.toktok = tokenize.toktok.ToktokTokenizer()
        self.word_tokenize = self.toktok.tokenize
        self.sent_tokenize = tokenize.sent_tokenize
        self.other_cat = other_cat
        self.dropna = dropna
        self.annotation_conflicts = dict()
        self.annotation_conflicts["interpretation"] = []
        self.annotation_conflicts["errors"] = []
        self.stats = defaultdict()
        self.res_dfs = dict()

    def parse_raw(self, raw_path, sent_tokenize=None):
        sent_tokenize = if_none(sent_tokenize, self.sent_tokenize)
        with open(raw_path, "r", encoding="utf-8") as sample:
            lines = [line.strip() for line in sample.readlines()]
            text = " ".join(lines[6:])
        idx, lang, date, url, title = lines[:5]
        return idx, lang, date, url, title, text, sent_tokenize(text)

    def parse_annotation(self, annotated_path):
        annotation = dict()
        with open(annotated_path, "r", encoding="utf-8") as sample:
            lines = [line.strip() for line in sample.readlines()]
        for line in lines[1:]:
            if len(line.split("\t")) == 4:
                mention, base_form, cat, clidx = line.split("\t")
                key = mention
                clidx = "-".join(clidx.split())
                if annotation.get(mention) is not None:
                    self.annotation_conflicts["interpretation"].append(
                        {"path": annotated_path, "line": line})
                    key = mention + "_" + cat
                annotation[key] = {"text": mention, "base_form": base_form, "cat": cat, "clidx": clidx}
        return lines[0], annotation

    def match_raw_and_annotation(self, sents, annotation, word_tokenize=None, other_cat="O"):
        word_tokenize = if_none(word_tokenize, self.word_tokenize)
        res = []
        _ = list(map(
            res.append,
            list(map(lambda sent: self.match_sent_and_annotation(
                sent, annotation, word_tokenize, other_cat), sents))))
        return res

    @staticmethod
    def mention2bio(mention, word_tokenize, other_cat="O"):
        tokens = word_tokenize(mention["text"])
        if mention["cat"] == other_cat:
            tags = [mention["cat"]] * len(tokens)
            clidx = [mention["clidx"]] * len(tokens)
            base_tokens = [other_cat] * len(tokens)
        else:
            tags = ["B_" + mention["cat"]] + ["I_" + mention["cat"]] * (len(tokens) - 1)
            clidx = ["B_" + mention["clidx"]] + ["I_" + mention["clidx"]] * (len(tokens) - 1)
            base_tokens = word_tokenize(mention["base_form"])
        return tokens, tags, base_tokens, clidx

    def match_sent_and_annotation(self, sent, annotation, word_tokenize=None, other_cat="O"):
        word_tokenize = if_none(word_tokenize, self.word_tokenize)
        info = []
        sent_len = len(sent)
        pos = 0
        prev_pos = 0
        while pos < sent_len:
            for mention in annotation:
                if sent[pos:].startswith(mention):
                    info.append({"cat": other_cat, "text": sent[prev_pos:pos], "clidx": other_cat})
                    sent = sent[pos:]
                    info.append(annotation.get(mention))
                    pos += len(mention)
                    prev_pos = pos
            pos += 1
        if prev_pos != sent_len - 1:
            info.append({"cat": other_cat, "text": sent[prev_pos:pos], "clidx": other_cat})
        res_info = dict()
        res_info["tokens"] = []
        res_info["tags"] = []
        res_info["clidxs"] = []
        res_info["base_forms"] = []
        for mention in info:
            for token, tag, base_form, clidx in zip(*self.mention2bio(mention, word_tokenize, other_cat)):
                res_info["tokens"].append(token)
                res_info["tags"].append(tag)
                res_info["base_forms"].append(base_form)
                res_info["clidxs"].append(clidx)
        return res_info

    def prc_sample(self, raw_path, annotated_path, sent_tokenize=None, word_tokenize=None, other_cat="O"):
        sent_tokenize = if_none(sent_tokenize, self.sent_tokenize)
        word_tokenize = if_none(word_tokenize, self.word_tokenize)
        idx_r, lang, date, url, title, text, sents = self.parse_raw(raw_path, sent_tokenize)
        idx_a, annotation = self.parse_annotation(annotated_path)
        assert idx_r == idx_a
        columns = ["doc_idxs", "tokens", "tags", "clidxs", "base_forms"]
        res_info = dict()
        res_info["doc_idxs"] = []
        res_info["tokens"] = []
        res_info["tags"] = []
        res_info["clidxs"] = []
        res_info["base_forms"] = []
        for info in self.match_raw_and_annotation(sents, annotation, word_tokenize, other_cat):
            res_info["doc_idxs"].append(idx_r)
            res_info["tokens"].append(" ".join(info["tokens"]))
            res_info["tags"].append(" ".join(info["tags"]))
            res_info["base_forms"].append(" ".join(info["base_forms"]))
            res_info["clidxs"].append(" ".join(info["clidxs"]))
        return pd.DataFrame(res_info, columns=columns)

    def prc_topic(self, root, topic_name,
                  annotated_prefix="annotated",
                  raw_prefix="raw",
                  sent_tokenize=None,
                  word_tokenize=None,
                  other_cat=None):
        sent_tokenize = if_none(sent_tokenize, self.sent_tokenize)
        word_tokenize = if_none(word_tokenize, self.word_tokenize)
        other_cat = if_none(other_cat, self.other_cat)
        df = pd.DataFrame(columns=["doc_idxs", "tokens", "tags", "clidxs", "base_forms"])
        for lang in os.listdir(str(root/annotated_prefix/topic_name)):
            files = os.listdir(str(root/annotated_prefix/topic_name/lang))
            for file_name in tqdm(files, total=len(files), leave=False,
                                  desc="cat: {}, lang: {}".format(topic_name, lang)):
                annotated_path = str(root/annotated_prefix/topic_name/lang/file_name)
                raw_path = str(root/raw_prefix/topic_name/lang/file_name[:-3]) + "txt"
                try:
                    df = df.append(
                        self.prc_sample(raw_path, annotated_path, sent_tokenize, word_tokenize, other_cat)
                    )
                except ValueError:
                    print("ValueError in path: {}".format(annotated_path))

        return df

    def prc_data(self, root=None,
                 save_dir_name="processed",
                 annotated_prefix="annotated",
                 raw_prefix="raw",
                 sent_tokenize=None,
                 word_tokenize=None,
                 other_cat="O"):
        self.res_dfs = dict()
        root = if_none(root, self.root)
        sent_tokenize = if_none(sent_tokenize, self.sent_tokenize)
        word_tokenize = if_none(word_tokenize, self.word_tokenize)
        for topic_name in os.listdir(str(root/annotated_prefix)):
            df = self.prc_topic(
                root, topic_name, annotated_prefix, raw_prefix,
                sent_tokenize, word_tokenize, other_cat)
            if not os.path.exists(str(root/save_dir_name)):
                os.mkdir(str(root/save_dir_name))
            if self.dropna:
                df = df.dropna()
            df.to_csv(str(root/save_dir_name/topic_name) + ".csv", index=False)
            self.stats[topic_name] = self.get_stats(df, other_cat)
            self.res_dfs[topic_name] = df
        return self.res_dfs

    @staticmethod
    def get_stats(df, other_cat="O"):
        langs = Counter(list(df.doc_idxs.apply(lambda x: x[:2])))
        tokens = Counter()
        _ = df.tokens.apply(lambda x: tokens.update(x.split()))
        bio_tags = Counter()
        _ = df.tags.apply(lambda x: bio_tags.update(x.split()))
        tags = Counter()
        _ = df.tags.apply(lambda x: tags.update([tag[2:] if tag != other_cat else tag for tag in x.split()]))
        clidxs = Counter()
        _ = df.clidxs.apply(lambda x: clidxs.update(x.split()))

        return {"langs": langs, "tokens": tokens, "bio_tags": bio_tags, "clidxs": clidxs, "tags": tags}
