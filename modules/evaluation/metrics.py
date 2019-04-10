from sklearn.metrics import f1_score, precision_score, recall_score


def tokens_scores(true_labels, pred_labels, sup_labels=None):
    pl = []
    for line in pred_labels:
        for p in line:
            pl.append(p)

    tl = []
    for line in true_labels:
        for p in line:
            tl.append(p)
    f1 = f1_score(tl, pl, average="macro", labels=sup_labels)
    rec = recall_score(tl, pl, average="macro", labels=sup_labels)
    prec = precision_score(tl, pl, average="macro", labels=sup_labels)
    return {"f1": f1, "rec": rec, "prec": prec}
