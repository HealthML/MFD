from sklearn.metrics import roc_auc_score


def multilabel_auroc_np(targets, preds):
    auroc = []
    for i in range(targets.shape[-1]):
        # ignore cases where all labels are equal - roc_auc_score will raise an Exception there
        unique_targets = set()
        for target in targets[:, i]:
            unique_targets.add(target)
            if len(unique_targets) > 1:
                break
        if len(unique_targets) > 1:
            auroc.append(roc_auc_score(targets[:, i], preds[:, i]))
    auroc = sum(auroc) / len(auroc)

    return auroc
