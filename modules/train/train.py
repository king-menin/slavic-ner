from tqdm import tqdm
from sklearn_crfsuite.metrics import flat_classification_report
import logging
import torch
from modules.utils.plot_metrics import get_mean_max_metric
from .optimization import BertAdam
import json
from modules.data.bert_data import BertNerData
from modules.models.released_models import released_models


logging.basicConfig(level=logging.INFO)


def train_step(dl, model, optimizer, lr_scheduler=None, clip=None, num_epoch=1):
    model.train()
    epoch_loss = 0
    idx = 0
    pr = tqdm(dl, total=len(dl), leave=False)
    for batch in pr:
        idx += 1
        model.zero_grad()
        try:
            loss = model.score(batch)
        except AttributeError:
            loss = model.module.score(batch)
        loss.backward()
        if clip is not None:
            _ = torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()
        optimizer.zero_grad()
        loss = loss.data.cpu().tolist()
        epoch_loss += loss
        pr.set_description("train loss: {}".format(epoch_loss / idx))
        if lr_scheduler is not None:
            lr_scheduler.step()
        torch.cuda.empty_cache()
    if lr_scheduler is not None:
        logging.info("\nlr after epoch: {}".format(lr_scheduler.lr))
    logging.info("\nepoch {}, average train epoch loss={:.5}\n".format(
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


def validate_step(dl, model, id2label, sup_labels, id2cls=None):
    model.eval()
    idx = 0
    preds_cpu, targets_cpu = [], []
    preds_cpu_cls, targets_cpu_cls = [], []
    for batch in tqdm(dl, total=len(dl), leave=False):
        idx += 1
        labels_mask, labels_ids = batch[-2:]
        preds = model(batch)
        if id2cls is not None:
            preds, preds_cls = preds
            preds_cpu_, targets_cpu_ = transformed_result_cls([preds_cls], [batch[-3]], id2cls)
            preds_cpu_cls.extend(preds_cpu_)
            targets_cpu_cls.extend(targets_cpu_)
        preds_cpu_, targets_cpu_ = transformed_result([preds], [labels_mask], id2label, [labels_ids])
        preds_cpu.extend(preds_cpu_)
        targets_cpu.extend(targets_cpu_)
    clf_report = flat_classification_report(targets_cpu, preds_cpu, labels=sup_labels, digits=3)
    if id2cls is not None:
        clf_report_cls = flat_classification_report([targets_cpu_cls], [preds_cpu_cls], digits=3)
        return clf_report, clf_report_cls
    return clf_report


def predict(dl, model, id2label, id2cls=None):
    model.eval()
    idx = 0
    preds_cpu = []
    preds_cpu_cls = []
    for batch, sorted_idx in tqdm(dl, total=len(dl), leave=False):
        idx += 1
        labels_mask, labels_ids = batch[-2:]
        preds = model.forward(batch)
        bs = batch[0].shape[0]
        if id2cls is not None:
            preds, preds_cls = preds
            unsorted_pred = [0] * bs
            for idx, sidx in enumerate(sorted_idx):
                unsorted_pred[sidx] = preds_cls[idx]
            preds_cls = unsorted_pred
            preds_cpu_ = transformed_result_cls([preds_cls], [preds_cls], id2cls, False)
            preds_cpu_cls.extend(preds_cpu_)
        unsorted_mask = [0] * bs
        unsorted_pred = [0] * bs
        for idx, sidx in enumerate(sorted_idx):
            unsorted_pred[sidx] = preds[idx]
            unsorted_mask[sidx] = labels_mask[idx]
        preds_cpu_ = transformed_result([unsorted_pred], [unsorted_mask], id2label)
        # print(len(preds_cpu_))
        preds_cpu.extend(preds_cpu_)
    if id2cls is not None:
        return preds_cpu, preds_cpu_cls
    # print(preds_cpu)
    return preds_cpu


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
                "clip": self.clip,
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
                 validate_every=1, schedule="warmup_linear", e=1e-6):
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
            sup_labels = data.id2label[1:]
        self.sup_labels = sup_labels
        self.best_model_path = best_model_path
        self.verbose = verbose
        self.history = []
        self.cls_history = []
        self.epoch = 0
        self.clip = clip
        self.best_target_metric = 0.
        self.lr_scheduler = None

    def save_config(self, path):
        with open(path, "w") as file:
            json.dump(self.config, file)

    @classmethod
    def from_config(cls, path, for_train=True):
        with open(path, "r") as file:
            config = json.load(file)
        data = BertNerData.from_config(config["data"], for_train)
        name = config["model"]["name"]
        # TODO: release all models (now only for BertBiLSTMNCRF)
        if name not in released_models:
            raise NotImplemented("from_config is implemented only for {} model :(".format(config["name"]))
        model = released_models[name].from_config(**config["model"]["params"])
        return cls(data, model, **config["learner"])

    def fit(self, epochs=100, resume_history=True, target_metric="f1"):
        if not resume_history:
            self.optimizer_defaults["t_total"] = epochs * len(self.data.train_dl)
            self.optimizer = BertAdam(**self.optimizer_defaults)
            self.history = []
            self.cls_history = []
            self.epoch = 0
            self.best_target_metric = 0.
        elif self.verbose:
            logging.info("Resuming train... Current epoch {}.".format(self.epoch))
        try:
            for _ in range(epochs):
                self.epoch += 1
                self.fit_one_cycle(self.epoch, target_metric)
        except KeyboardInterrupt:
            pass

    def fit_one_cycle(self, epoch, target_metric="f1"):
        train_step(self.data.train_dl, self.model, self.optimizer, self.lr_scheduler, self.clip, epoch)
        if epoch % self.validate_every == 0:
            if self.data.is_cls:
                rep, rep_cls = validate_step(self.data.valid_dl, self.model, self.data.id2label, self.sup_labels,
                                             self.data.id2cls)
                self.cls_history.append(rep_cls)
            else:
                rep = validate_step(self.data.valid_dl, self.model, self.data.id2label, self.sup_labels)
            self.history.append(rep)
        idx, metric = get_mean_max_metric(self.history, target_metric, True)
        if self.verbose:
            logging.info("on epoch {} by max_{}: {}".format(idx, target_metric, metric))
            print(self.history[-1])
            if self.data.is_cls:
                logging.info("on epoch {} classification report:")
                print(self.cls_history[-1])
        # Store best model
        if self.best_target_metric < metric:
            self.best_target_metric = metric
            if self.verbose:
                logging.info("Saving new best model...")
            self.save_model()

    def predict(self, dl):
        if self.data.is_cls:
            return predict(dl, self.model, self.data.id2label, self.data.id2cls)
        return predict(dl, self.model, self.data.id2label)
    
    def save_model(self, path=None):
        path = path if path else self.best_model_path
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path=None):
        path = path if path else self.best_model_path
        self.model.load_state_dict(torch.load(path, map_location="cuda:0"))
