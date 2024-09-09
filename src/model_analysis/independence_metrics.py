import argparse

import wandb
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

from models.models import IEAquaticDilated
from dataloading.dataloaders import LitCoverageDatasetHDF5
from util.load_config import config


def predict_val_collate_fn(batch):
    batch = torch.cat([x for x, _ in batch])
    return batch


class BioTechDataset(torch_data.Dataset):
    def __init__(self, bio, tech):
        self.bio = bio
        self.tech = tech

    def __len__(self):
        return len(self.bio)

    def __getitem__(self, idx):
        return self.bio[idx], self.tech[idx]


class ShuffleDiscriminator(pl.LightningModule):
    def __init__(
            self,
            features_biological,
            features_technical,
            batch_size=512,
            lr=1e-3,
            dim_hidden=512,
    ):
        super().__init__()

        self.dataset = BioTechDataset(
            bio=features_biological,
            tech=features_technical,
        )
        self.dataset_train, self.dataset_val = torch_data.random_split(self.dataset, [.7, .3])

        input_size = self.dataset.bio.shape[1] * 2
        self.discriminator = nn.Sequential(
            nn.Linear(input_size, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, 1),
        )

        self.save_hyperparameters()

    def forward(self, x):
        return self.discriminator(x)

    def loss_fn(self, preds, target):
        return F.binary_cross_entropy_with_logits(preds, target)

    def train_dataloader(self):
        return torch_data.DataLoader(
            dataset=self.dataset_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch_data.DataLoader(
            dataset=self.dataset_val,
            batch_size=self.hparams.batch_size,
            shuffle=True,
        )

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            lr=self.hparams.lr,
            params=self.parameters(),
        )
        return optim

    def training_step(self, batch, batch_idx=None):
        return self._step(batch=batch, batch_idx=batch_idx, step_name='train')

    def validation_step(self, batch, batch_idx=None):
        return self._step(batch=batch, batch_idx=batch_idx, step_name='val')

    def _step(self, batch, step_name, batch_idx=None):
        bio, tech = batch

        batch_true = torch.cat([bio, tech], dim=1)
        tech_shuffled = tech[torch.randperm(bio.shape[0])]
        batch_fake = torch.cat([bio, tech_shuffled], dim=1)

        x = torch.cat([batch_true, batch_fake], dim=0)
        y_hat = self.forward(x)

        y = torch.ones_like(y_hat)
        y[y_hat.shape[0] // 2:] = 0

        loss = self.loss_fn(target=y, preds=y_hat)
        accuracy = torchmetrics.functional.accuracy(preds=y_hat, target=y, task='binary')

        self.log(f'{step_name}_loss', loss, prog_bar=True)
        self.log(f'{step_name}_accuracy', accuracy, prog_bar=True)
        return loss


class Predictor(pl.LightningModule):
    def __init__(
            self,
            features_biological,
            features_technical,
            batch_size=256,
            lr=1e-3,
            dim_hidden=256,
    ):
        super().__init__()

        self.dataset = BioTechDataset(
            bio=features_biological,
            tech=features_technical,
        )
        self.dataset_train, self.dataset_val = torch_data.random_split(self.dataset, [.7, .3])

        input_size = self.dataset.bio.shape[1]
        self.predictor = nn.Sequential(
            nn.Linear(input_size, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, input_size),
        )

        self.save_hyperparameters()

    def forward(self, x):
        return self.predictor(x)

    def loss_fn(self, preds, target):
        preds = (preds - preds.mean(0)) / (preds.std(0))
        preds = torch.nan_to_num(preds)
        target = (target - target.mean(0)) / (target.std(0))
        target = torch.nan_to_num(target)

        cc = (preds.T @ target / preds.shape[0]).clip(min=-1, max=1).pow(2)
        cc = torch.triu(cc, diagonal=0)
        cc = torch.tril(cc, diagonal=0).sum()
        return -cc

    def train_dataloader(self):
        return torch_data.DataLoader(
            dataset=self.dataset_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch_data.DataLoader(
            dataset=self.dataset_val,
            batch_size=self.hparams.batch_size,
            shuffle=True,
        )

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            lr=self.hparams.lr,
            params=self.parameters(),
        )
        return optim

    def training_step(self, batch, batch_idx=None):
        return self._step(batch=batch, batch_idx=batch_idx, step_name='train')

    def validation_step(self, batch, batch_idx=None):
        return self._step(batch=batch, batch_idx=batch_idx, step_name='val')

    def _step(self, batch, step_name, batch_idx=None):
        bio, tech = batch
        x, y = bio, tech

        y_hat = self.predictor(x)

        loss = self.loss_fn(target=y, preds=y_hat)

        self.log(f'{step_name}_loss', loss, prog_bar=True, on_step='train' in step_name)
        return loss


def get_predictions(model, datamodule):
    datamodule.setup(stage='test')
    datamodule.data.augment_off()
    dloader = datamodule.test_dataloader()
    dloader.collate_fn = predict_val_collate_fn

    model.eval()
    trainer = pl.Trainer(enable_checkpointing=False)
    ret = trainer.predict(
        model=model,
        dataloaders=dloader,
    )
    predictions, features_all, features_biological, features_technical = (
        # inner loop - iterate over all returned batches and select the appropriate output type
        torch.cat([batch[output_idx] for batch in ret])
        # outer loop - iterate over the 4 output types (predictions and 3 feature types)
        for output_idx in range(4)
    )
    print(predictions.shape)
    print(features_all.shape)
    print(features_biological.shape)
    print(features_technical.shape)

    return features_biological, features_technical


def eval_shuffled_discriminator(features_biological, features_technical):
    model = ShuffleDiscriminator(features_biological, features_technical)
    trainer = pl.Trainer(max_epochs=1, enable_checkpointing=False)
    trainer.fit(model)

    return {
        'train_accuracy': trainer.logged_metrics['train_accuracy'].item(),
        'val_accuracy': trainer.logged_metrics['val_accuracy'].item(),
    }


def eval_predictor(features_biological, features_technical):
    model = Predictor(features_biological, features_technical)
    trainer = pl.Trainer(max_epochs=1, enable_checkpointing=False)
    trainer.fit(model)

    return {
        'train_R2': trainer.logged_metrics['train_loss'].item(),
        'val_R2': trainer.logged_metrics['val_loss'].item(),
    }


def main(
        checkpoint_path,
        toy_data,
):
    model = IEAquaticDilated.load_from_checkpoint(checkpoint_path)
    hparams = model.hparams

    datamodule = LitCoverageDatasetHDF5(
        seq_order=hparams.seq_order,
        seq_len=hparams.seq_len,
        basepath='data/processed/GRCh38/toydata' if toy_data else hparams.basepath,
        ref_path=config['reference']['GRCh38'] if toy_data else hparams.ref,
        batch_size=hparams.batch_size,
    )
    features_biological, features_technical = get_predictions(model, datamodule)

    metrics = {}
    metrics.update(**{
        f'independence/shuffled_{key}': val
        for key, val in eval_shuffled_discriminator(features_biological, features_technical).items()
    })
    metrics.update(**{
        f'independence/pred_bio_tech_{key}': val
        for key, val in eval_predictor(features_biological, features_technical).items()
    })
    metrics.update(**{
        f'independence/pred_tech_bio_{key}': val
        for key, val in eval_predictor(features_technical, features_biological).items()
    })
    print(metrics)

    run_id = checkpoint_path.split('/')[-2]

    api = wandb.Api()
    run = api.run(f"nucleotran-alex/nucleotran-disentanglement/{run_id}")
    for k, v in metrics.items():
        wandb.run.summary[k] = v
    run.summary.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'checkpoint_path',
        type=str,
        help='',
    )
    parser.add_argument(
        '--toy_data',
        action='store_true',
        help='',
    )
    args = parser.parse_args()

    main(**vars(args))
