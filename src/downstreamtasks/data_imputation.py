import argparse
from pathlib import Path
import sys
import os

import pandas as pd
import numpy as np
import random
import torch

np.random.seed(1)
torch.manual_seed(1)
random.seed(1)

from tqdm import tqdm
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score

import models.models as src_models
from dataloading.dataloaders import LitCoverageDatasetHDF5
from util.load_config import config


class ImputationModelWrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x, y = x[0], y[0]

        return self.model.imputation_step(x), y[:, self.model.loss_fun.missing_class_idxs]


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

    return auroc


def main(ckpt, debug, batch_size):
    hyperparams = torch.load(ckpt, map_location='cpu')['hyper_parameters']
    model_cls_str = hyperparams.get('model_class', 'IEAquaticDilated')
    model_cls = getattr(src_models, model_cls_str)
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    model = model_cls.load_from_checkpoint(ckpt, map_location=device)

    batch_size = hyperparams.get('batch_size', batch_size)

    datamodule = LitCoverageDatasetHDF5(
        seq_order=2,
        seq_len=model.hparams.seq_len,
        basepath=f"data/processed/GRCh38/{'toydata' if debug else '221111_128bp_minoverlap64_mincov2_nc10_tissues'}",
        ref_path=config['reference']['GRCh38'],
        batch_size=batch_size,
    )
    datamodule.setup(stage='test')
    dloader = datamodule.test_dataloader()

    model.eval()
    trainer = pl.Trainer()

    ret = trainer.predict(
        model=ImputationModelWrapper(model),
        dataloaders=dloader,
    )
    y = torch.cat([batch[1] for batch in ret])
    y_hats = {k: torch.cat([batch[0][k] for batch in ret]) for k in ret[0][0]}
    for k, y_hat in y_hats.items():
        print(k, y_hat.shape)
    print(y.shape)

    rows = []
    for k, y_hat in y_hats.items():
        auroc = multilabel_auroc_np(y.cpu().numpy(), y_hat.cpu().numpy())
        row = {model.hparams.ignore_classes[i]: a for i, a in enumerate(auroc)}
        row['average'] = sum(auroc) / len(auroc)
        row['model_cls'] = model_cls_str
        row['imputation_type'] = k
        rows.append(row)

    df_res = pd.DataFrame(rows)
    print(df_res)

    dir_results = Path(ckpt).parent / 'imputation'
    dir_results.mkdir(exist_ok=True)
    df_res.to_csv(
        dir_results / 'auroc.tsv',
        sep='\t',
        index=False,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt',
        type=str,
        required=True,
    )
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    main(ckpt=args.ckpt, debug=args.debug, batch_size=args.batch_size)
