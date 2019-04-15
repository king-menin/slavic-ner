from tqdm import tqdm
import logging
import torch
from .optimization import BertAdam
import json
from modules.data.preprocessor import NerDataLoader
from modules.evaluation.metrics import tokens_scores


logging.basicConfig(level=logging.INFO)


def train_step(dl, model, optimizer, num_epoch=1):
    model.train()
    epoch_loss = 0
    idx = 0
    pr = tqdm(dl, total=len(dl), leave=False)
    for batch, _ in pr:
        idx += 1
        model.zero_grad()
        loss = model.score(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss = loss.data.cpu().tolist()
        epoch_loss += loss
        pr.set_description("Epoch {}, average train loss: {}".format(num_epoch, epoch_loss / idx))
        torch.cuda.empty_cache()
    logging.info("\nEpoch {}, average train epoch loss={:.5}\n".format(
        num_epoch, epoch_loss / idx))


def transformed_result(preds, mask, id2label, target_all=None, pad_idx=0):
    preds_cpu = []
    targets_cpu = []
    lc = len(id2label)
    if target_all is not None:
        for batch_p, batch_t, batch_m in zip(preds, target_all, mask):
            for pred, true_, bm in zip(batch_p, batch_t, batch_m):
                sent = []
                sent_t = []
                bm = bm.sum().cpu().data.tolist()
                for p, t in zip(pred[:bm], true_[:bm]):
                    p = p.cpu().data.tolist()
                    p = p if p < lc else pad_idx
                    sent.append(p)
                    sent_t.append(t.cpu().data.tolist())
                preds_cpu.append([id2label[w] for w in sent])
                targets_cpu.append([id2label[w] for w in sent_t])
    else:
        for batch_p, batch_m in zip(preds, mask):
            for pred, bm in zip(batch_p, batch_m):
                assert len(pred) == len(bm)
                bm = bm.sum().cpu().data.tolist()
                sent = pred[:bm].cpu().data.tolist()
                preds_cpu.append([id2label[w] for w in sent])
    if target_all is not None:
        return preds_cpu, targets_cpu
    else:
        return preds_cpu

    
def transformed_result_cls(preds, target_all, cls2label, return_target=True):
    preds_cpu = []
    targets_cpu = []
    for batch_p, batch_t in zip(preds, target_all):
        for pred, true_ in zip(batch_p, batch_t):
            preds_cpu.append(cls2label[pred.cpu().data.tolist()])
            if return_target:
                targets_cpu.append(cls2label[true_.cpu().data.tolist()])
    if return_target:
        return preds_cpu, targets_cpu
    return preds_cpu


def validate_step(dl, model, data, sup_labels=None):
    model.eval()
    idx = 0
    preds_cpu, targets_cpu = [], []
    preds_cpu_cls, targets_cpu_cls = [], []
    for batch, _ in tqdm(dl, total=len(dl), leave=False):
        idx += 1
        if data.idx2cls is not None:
            labels_mask, labels_ids = batch[-3], batch[-2]
        else:
            labels_mask, labels_ids = batch[-2:]
        preds = model.forward(batch)
        if data.idx2cls is not None:
            preds, preds_cls = preds
            preds_cpu_, targets_cpu_ = transformed_result_cls([preds_cls], [batch[-1]], data.idx2cls)
            preds_cpu_cls.extend(preds_cpu_)
            targets_cpu_cls.extend(targets_cpu_)
        preds_cpu_, targets_cpu_ = transformed_result([preds], [labels_mask], data.idx2label, [labels_ids])
        preds_cpu.extend(preds_cpu_)
        targets_cpu.extend(targets_cpu_)
    tags_report = tokens_scores(targets_cpu, preds_cpu, sup_labels)
    cls_report = None
    if data.idx2cls is not None:
        cls_report = tokens_scores([targets_cpu_cls], [preds_cpu_cls])
    return {"tags_report": tags_report, "cls_report": cls_report}


def predict(dl, model, data):
    model.eval()
    idx = 0
    preds_cpu = []
    preds_cpu_cls = []
    for batch, sorted_idx in tqdm(dl, total=len(dl), leave=False):
        idx += 1
        labels_mask, labels_ids = batch[-2:]
        preds = model.forward(batch)
        bs = batch[0].shape[0]
        if data.idx2cls is not None:
            preds, preds_cls = preds
            preds_cpu_ = transformed_result_cls([preds_cls], [preds_cls], data.idx2cls, False)
            unsorted_pred = [0] * bs
            for idx, sidx in enumerate(sorted_idx):
                unsorted_pred[sidx] = preds_cpu_[idx]
            preds_cpu_cls.extend(unsorted_pred)
        unsorted_mask = [0] * bs
        unsorted_pred = [0] * bs
        for idx, sidx in enumerate(sorted_idx):
            unsorted_pred[sidx] = preds[idx]
            unsorted_mask[sidx] = labels_mask[idx]
        
        preds_cpu_ = transformed_result([unsorted_pred], [unsorted_mask], data.idx2label)
        preds_cpu.extend(preds_cpu_)

    return {"tags": preds_cpu, "cls": preds_cpu_cls}


class NerLearner(object):

    @property
    def config(self):
        config = {
            "data": self.data.config,
            "model": self.model.config,
            "learner": {
                "best_model_path": self.best_model_path,
                "lr": self.lr,
                "betas": self.betas,
                "verbose": self.verbose,
                "sup_labels": self.sup_labels,
                "t_total": self.t_total,
                "warmup": self.warmup,
                "weight_decay": self.weight_decay,
                "validate_every": self.validate_every,
                "schedule": self.schedule,
                "e": self.e
            }
        }
        return config

    def __init__(self, model, data, best_model_path, lr=0.001, betas=[0.8, 0.9], clip=5,
                 verbose=True, sup_labels=None, t_total=-1, warmup=0.1, weight_decay=0.01,
                 validate_every=1, schedule="warmup_linear", e=1e-6, save_every_epoch=False):
        self.model = model
        self.optimizer = BertAdam(model, lr, t_total=t_total, b1=betas[0], b2=betas[1], max_grad_norm=clip)
        self.optimizer_defaults = dict(
            model=model, lr=lr, warmup=warmup, t_total=t_total, schedule="warmup_linear",
            b1=betas[0], b2=betas[1], e=1e-6, weight_decay=weight_decay,
            max_grad_norm=clip)

        self.lr = lr
        self.betas = betas
        self.clip = clip
        self.sup_labels = sup_labels
        self.t_total = t_total
        self.warmup = warmup
        self.weight_decay = weight_decay
        self.validate_every = validate_every
        self.schedule = schedule
        self.data = data
        self.e = e
        if sup_labels is None:
            sup_labels = data.idx2label
        self.sup_labels = sup_labels
        self.best_model_path = best_model_path
        self.verbose = verbose
        self.history = []
        self.epoch = 0
        self.clip = clip
        self.best_target_metric = 0.

        self.save_every_epoch = save_every_epoch

    def save_config(self, path):
        with open(path, "w") as file:
            json.dump(self.config, file)

    @classmethod
    def from_config(cls, config, for_train=True):
        raise NotImplemented

    def fit(self, epochs=100, resume_history=True, target_metric="f1"):
        if not resume_history:
            self.optimizer_defaults["t_total"] = epochs * len(self.data.train_dl)
            self.optimizer = BertAdam(**self.optimizer_defaults)
            self.history = []
            self.epoch = 0
            self.best_target_metric = 0.
        elif self.verbose:
            logging.info("Resuming train... Current epoch {}.".format(self.epoch))
        try:
            for _ in range(epochs):
                self.epoch += 1
                train_step(self.data.train_dl, self.model, self.optimizer, self.epoch)
                saved = False
                if self.epoch % self.validate_every == 0:
                    metrics = validate_step(
                        self.data.valid_dl,
                        self.model,
                        self.data,
                        self.sup_labels)
                    metrics["epoch"] = self.epoch
                    self.history.append(metrics)
                    if self.verbose:
                        logging.info("On epoch {} tags {} score: {}".format(
                            self.epoch, target_metric, metrics["tags_report"].get(target_metric)))
                        if metrics["cls_report"] is not None:
                            logging.info("On epoch {} cls {} score: {}".format(
                                self.epoch, target_metric, metrics["cls_report"].get(target_metric)))
                    # Store best model
                    if self.best_target_metric < metrics["tags_report"].get(target_metric):
                        self.best_target_metric = metrics["tags_report"].get(target_metric)
                        if self.verbose:
                            logging.info("Saving new best model...")
                        self.save_model()
                        saved = True
                if not saved and self.save_every_epoch:
                    if self.verbose:
                        logging.info("Saving new best model...")
                    self.save_model()
        except KeyboardInterrupt:
            pass

    def predict(self, dl):
        """
        Predict tags for dl, pandas.DataFrame or path to pandas.DataFrame.

        Parameters
        ----------
        dl : NerDataLoader, pandas.DataFrame or str
            If pandas.DataFrame or create NerDataLoader and predict.

        Returns
        ----------
        res : dict
            Predicted labels for input dl.
        """
        if not isinstance(dl, NerDataLoader):
            dl = self.data.load_test_dl(dl)
        return predict(dl, self.model, self.data)

    def save_model(self, path=None):
        path = path if path else self.best_model_path
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path=None):
        path = path if path else self.best_model_path
        self.model.load_state_dict(torch.load(path))
