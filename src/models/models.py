from decimal import Decimal
from pathlib import Path

import pytorch_lightning
import torch
import torchmetrics.functional
import wandb
import yaml
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import random_projection
from torch import nn, sigmoid
import numpy as np
from typing import Optional
from torchmetrics.classification import MultilabelAccuracy, MultilabelAUROC
import torch.nn.functional as F
import datetime
import pandas as pd
import seaborn as sns

import util.metrics as metrics
import dataloading.metadata as data_metadata
import gc


class SeqModel(pytorch_lightning.LightningModule):

    def __init__(
            self,
            lr: float = 1e-3,
            seq_len: int = 128,
            n_classes: int = 1,
            class_freq=None,
            metadata_loader=None,
            ignore_classes=None,
            log_hyperparams=False,  # why set this to False?
            *args,
            **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(logger=log_hyperparams)

        # these are used to compute epoch-wise metrics, e.g., auroc
        self.training_step_preds = []
        self.training_step_targets = []
        self.validation_step_preds = []
        self.validation_step_targets = []

        # initialize all needed objects e.g., dataset, model etc.

    def train_dataloader(self):
        raise NotImplementedError()

    def val_dataloader(self):
        raise NotImplementedError()

    def test_dataloader(self):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        # using self.log_dict or self.log will automatically log the values to Weights and Biases
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def test_step(self, batch, batch_idx, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, x, **kwargs):
        raise NotImplementedError()

    def configure_optimizers(self):
        raise NotImplementedError()

    def on_train_epoch_start(self):
        # this is is potentially redundant / doesn't work (?)
        self.trainer.datamodule.data.augment_on()

    def on_validation_model_eval(self):
        # this is is potentially redundant / doesn't work (?)
        self.trainer.datamodule.data.augment_off()

    def on_validation_model_train(self):
        # this is is potentially redundant / doesn't work (?)
        self.trainer.datamodule.data.augment_on()


class DinucleotideBaseline(SeqModel):
    """
    baseline model that works only with dinulceotide frequencies.

    expects input to be shape (16, l), where l is the sequence length
    """

    def __init__(self,
                 seq_len: int,  # = 128
                 n_classes: int,  # = 1
                 lr: float,  # =1e-3
                 init_bias=None,
                 class_freq=None,
                 log_hyperparams=False,
                 *args,
                 **kwargs
                 ):
        super().__init__()

        self.avgpool = nn.AvgPool1d(kernel_size=seq_len, stride=1, padding=0)
        self.linear = nn.Linear(in_features=16, out_features=n_classes)
        self.sigmoid = nn.Sigmoid()

        if init_bias is not None:
            # initialize bias with pre-defined values
            self.linear.bias.data = init_bias

        if class_freq is not None:
            # initialize bias from class frequencies
            init_bias = -np.log((1 / class_freq - 1))
            self.linear.bias.data = init_bias

        self.loss_fun = nn.BCELoss(reduction='mean')

        self.n_classes = n_classes
        self.accuracy = MultilabelAccuracy(num_labels=self.n_classes, average='macro')
        self.save_hyperparameters(logger=log_hyperparams)

    def forward(self, x):
        x = torch.flatten(self.avgpool(x), 1, 2)
        x = self.linear(x)
        out = self.sigmoid(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x[
            0]  # TODO: why does the loaded data have an additional dimension in front (?) -> this is because we use BatchSampler as sampler and leave batch size = 1 (the default)
        y = y[0]
        y_hat = self(x)
        loss = self.loss_fun(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        self.log("train_accuracy", accuracy, on_step=False, prog_bar=True, on_epoch=True, logger=True)
        self.log("train_loss", loss, on_step=True, prog_bar=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x[0]  # TODO: why does the loaded data have an additional dimension in front (?)
        y = y[0]
        y_hat = self(x)
        loss = self.loss_fun(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        self.log("val_accuracy", accuracy, on_step=False, prog_bar=True, on_epoch=True, logger=True)
        self.log("val/loss", loss, on_step=True, prog_bar=True, on_epoch=True, logger=True)

    def configure_optimizers(self, lr: Optional[float] = None):

        if lr is None:
            lr = float(self.hparams.lr)

        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

        return optimizer


class VikiValentinConvModel(SeqModel):
    """
    baseline model that works only with dinulceotide frequencies.

    expects input to be shape (16, l), where l is the sequence length
    """

    def __init__(self,
                 seq_len: int = 128,
                 n_classes: int = 1,
                 init_bias=None,
                 class_freq=None,
                 log_hyperparams=False,
                 lr=1e-3,
                 *args,
                 **kwargs
                 ):

        super().__init__()

        self.n_classes = n_classes
        self.accuracy = MultilabelAccuracy(num_labels=self.n_classes, average='macro')

        input_size = (200, 16, seq_len)
        output_size = input_size

        # model architecture
        print("initialize architecture")
        print("first conv block -----------------")
        n_filters_first_conv = 32
        self.first_conv = nn.Conv1d(in_channels=16, out_channels=n_filters_first_conv, kernel_size=3)
        output_size = (output_size[0], n_filters_first_conv, output_size[2] - 2)
        print("first conv output size: ", output_size)
        self.first_bn = nn.BatchNorm1d(num_features=n_filters_first_conv)
        self.relu = nn.ReLU()
        self.first_max_pool = nn.MaxPool1d(kernel_size=3)
        output_size = (output_size[0], output_size[1], output_size[2] // 3)
        print("first max pool output size: ", output_size)

        print("second conv block -----------------")
        n_filters_second_conv = 64
        self.second_conv = nn.Conv1d(in_channels=n_filters_first_conv, out_channels=n_filters_second_conv,
                                     kernel_size=3)
        output_size = (output_size[0], n_filters_second_conv, output_size[2] - 2)
        print("second conv output size: ", output_size)
        self.second_bn = nn.BatchNorm1d(num_features=n_filters_second_conv)
        self.second_max_pool = nn.MaxPool1d(kernel_size=3)
        output_size = (output_size[0], output_size[1], output_size[2] // 3)
        print("second max pool output size: ", output_size)

        print("linear block -----------------")
        self.flatten = nn.Flatten()
        output_size = (output_size[0], output_size[1] * output_size[2])
        print("flatten output shape: ", output_size)
        self.linear = nn.Linear(in_features=output_size[1],
                                out_features=self.n_classes)  # TODO: this is hardcoded, should be calculated from the input shape
        output_size = (output_size[0], self.n_classes)
        print("linear output shape: ", output_size)
        self.sigmoid = nn.Sigmoid()

        """
        input shape:  torch.Size([200, 16, 128])
        first conv block -----------------
        torch.Size([200, 32, 126])
        torch.Size([200, 32, 126])
        torch.Size([200, 32, 126])
        torch.Size([200, 32, 42])
        second conv block -------------
        torch.Size([200, 64, 40])
        torch.Size([200, 64, 40])
        torch.Size([200, 64, 40])
        torch.Size([200, 64, 13])
        linear block ---------------------
        torch.Size([200, 832])
        ...

        input shape:  torch.Size([4096, 16, 256])                                                                                                        
        first conv block -----------------
        conv1 torch.Size([4096, 32, 254])
        bn1 torch.Size([4096, 32, 254])
        relu1 torch.Size([4096, 32, 254])
        maxpool1 torch.Size([4096, 32, 84])
        second conv block -------------
        conv2 torch.Size([4096, 64, 82])
        bn2 torch.Size([4096, 64, 82])
        relu2 torch.Size([4096, 64, 82])
        maxpool2 torch.Size([4096, 64, 27])
        linear block ---------------------
        flatten torch.Size([4096, 1728])
        linear torch.Size([4096, 2106])
        sigmoid torch.Size([4096, 2106])
        output shape:  torch.Size([4096, 2106])
        """

        # bias initialization
        if init_bias is not None:
            # initialize bias with pre-defined values
            self.linear.bias.data = init_bias

        if class_freq is not None:
            # initialize bias from class frequencies
            init_bias = -np.log((1 / class_freq - 1))
            self.linear.bias.data = init_bias

        # loss function

        self.loss_fun = nn.BCELoss(reduction='mean')
        # nn.MultiLabelSoftMarginLoss() # nn.CrossEntropyLoss(reduction='mean')
        self.save_hyperparameters(logger=log_hyperparams)

    def forward(self, x):
        activate_print_shape = False
        print_shape = lambda *args: print(*args) if activate_print_shape else None

        print_shape("forward pass shapes")
        print_shape("input shape: ", x.shape)
        print_shape("first conv block -----------------")
        out = self.first_conv(x)
        print_shape("conv1", out.shape)
        out = self.first_bn(out)
        print_shape("bn1", out.shape)
        out = self.relu(out)
        print_shape("relu1", out.shape)
        out = self.first_max_pool(out)
        print_shape("maxpool1", out.shape)

        print_shape("second conv block -------------")
        out = self.second_conv(out)
        print_shape("conv2", out.shape)
        out = self.second_bn(out)
        print_shape("bn2", out.shape)
        out = self.relu(out)
        print_shape("relu2", out.shape)
        out = self.second_max_pool(out)
        print_shape("maxpool2", out.shape)

        print_shape("linear block ---------------------")
        out = self.flatten(out)
        print_shape("flatten", out.shape)
        out = self.linear(out)
        print_shape("linear", out.shape)
        out = self.sigmoid(out)
        print_shape("sigmoid", out.shape)
        print_shape("output shape: ", out.shape)

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x[0]  # TODO: why does the loaded data have an additional dimension in front (?)
        y = y[0]
        y_hat = self(x)
        loss = self.loss_fun(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x[0]  # TODO: why does the loaded data have an additional dimension in front (?)
        y = y[0]
        y_hat = self(x)
        loss = self.loss_fun(y_hat, y)
        self.log("val/loss", loss, on_step=False, prog_bar=True, on_epoch=True, logger=True)
        accuracy = self.accuracy(y_hat, y)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, logger=True)

    def configure_optimizers(self, lr: Optional[float] = None):

        if lr is None:
            lr = float(self.hparams.lr)

        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

        return optimizer


# Beluga Model
class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class Beluga(VikiValentinConvModel):
    def __init__(self, **kwargs):
        super(Beluga, self).__init__(**kwargs)
        self.model = nn.Sequential(
            nn.Sequential(
                # nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 8)),
                nn.Conv2d(in_channels=1, out_channels=320, kernel_size=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(320, 320, (1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4), (1, 4)),
                nn.Conv2d(320, 480, (1, 8)),
                nn.ReLU(),
                nn.Conv2d(480, 480, (1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4), (1, 4)),
                nn.Conv2d(480, 640, (1, 8)),
                nn.ReLU(),
                nn.Conv2d(640, 640, (1, 8)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0), -1)),
                nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(3968, 2003)),
                nn.ReLU(),
                nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(2003, 2002)),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.unsqueeze(0).permute(1, 0, 2, 3)
        return self.model(x)


class Seal(VikiValentinConvModel):
    def __init__(self, **kwargs):
        super(Dolphin, self).__init__(**kwargs)

        def conv_block(in_channels, out_channels, kernel_size, dropout=0.0, max_pool=False, max_kernel_size=4,
                       max_stride=4):
            assert isinstance(kernel_size, int)
            assert isinstance(in_channels, int)
            assert isinstance(out_channels, int)

            block = [
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
                nn.ReLU(),
            ]
            if dropout > 0.0:
                block.append(nn.Dropout(p=dropout))
            if max_pool:
                block.append(nn.MaxPool1d(kernel_size=max_kernel_size, stride=max_stride))
            return block

        self.model = nn.Sequential(
            nn.Sequential(
                *conv_block(in_channels=16, out_channels=64, kernel_size=3),
                *conv_block(in_channels=64, out_channels=128, kernel_size=3, dropout=0.0, max_pool=True,
                            max_kernel_size=4, max_stride=4),
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0), -1)),
                nn.Linear(3968, 2003),
                nn.ReLU(),
                nn.Linear(2003, 2106),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


##################
# Basenji2 Model #
##################


class AquaticBasenji(SeqModel):
    """
    Basenji2-like model with binary output

    For an illustration of the model consider

    Avsec, Žiga, et al. "Effective gene expression prediction from sequence by integrating long-range interactions." Nature methods 18.10 (2021): 1196-1203.
    Extended Data Fig. 1

    The default parameters follow those presented there, except L2, crop, seq_len

        Original Parameters:

        C:  number of channels (original: 768)
        L1: number of conv. layers + pooling in the Conv Tower (original: 6)
        L2: numver of residual dilated conv blocks (original: 11)
        D:  Dilation rate, rate by which dilation increases in every residual dilated conv block (original: 1.5)
        activation: activation function (original: "gelu")
        input_kernel_size:  kernel-size in the input layer (original: 15)
        tower_kernel_size:  kernel-size in the Conv Tower (original: 5)
        dilated_residual_kernel_size:   kernel-size in the residual dilated convs (original: 3)
        dilated_residual_dropout:       dropout rate in the residual dilated convs (original: 0.3)
        pointwise_dropout:  dropout in the Pointwise conv layer (original: 0.05)

        Other Parameters:

        out_pool_type:  if sequence is longer than 1 after cropping, which pooling type to use
        seq_order:      the order of the input sequence encoding
        crop:           how many positions to crop on ether size of the output after the residual dilated conv


    """

    def __init__(self,
                 seq_len: int = 2176,
                 n_classes: int = 1,
                 init_bias=None,
                 class_freq=None,
                 log_hyperparams=False,
                 lr=1e-3,
                 C=768,
                 L1=6,
                 L2=2,
                 D=1.5,
                 activation='gelu',
                 input_kernel_size=15,
                 tower_kernel_size=5,
                 dilated_residual_kernel_size=3,
                 dilated_residual_dropout=0.3,
                 pointwise_dropout=0.05,
                 out_pool_type=None,
                 seq_order=1,
                 crop=8,
                 weight_decay=0.,
                 *args,
                 **kwargs
                 ):

        super().__init__()

        in_channels = 4 ** seq_order

        self.h_seq_len = seq_len / 2 ** (L1 + 1)
        self.out_seq_len = self.h_seq_len - 2 * crop
        self.out_pool_type = out_pool_type

        assert self.out_seq_len > 0, f'Error: specified cropping is too large! {crop} * 2 >= {self.h_seq_len}'

        if self.out_seq_len != 1:
            print(f'Warnign: after cropping, the output sequence length ({self.out_seq_len}) is greater than one')
            assert out_pool_type is not None, 'Error: need to specify pooling type ("avg","max")'
            if out_pool_type == 'avg':
                self.out_pool = nn.AvgPool1d(self.out_seq_len)
            elif out_pool_type == 'max':
                self.out_pool = nn.MaxPool1d(self.out_seq_len)
            else:
                raise NotImplementedError(f'pooling type "{out_pool_type}" is not supported.')
        else:
            self.out_pool = nn.Identity()

        self.stem = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=int(.375 * C), kernel_size=input_kernel_size,
                      activation=activation),
            nn.MaxPool1d(2, 2)
        )

        self.tower = Basenji2ConvTower(int(.375 * C), C, kernel_size=tower_kernel_size, L=L1, activation=activation)
        self.dilated_residual_block = Basenji2DilatedResidualBlock(C, kernel_size=dilated_residual_kernel_size,
                                                                   dropout=dilated_residual_dropout, D=D,
                                                                   channel_multiplier=0.5, L=L2, activation=activation)
        self.cropping = Cropping1d(crop)
        self.pointwise = ConvBlock(C, 2 * C, kernel_size=1, dropout=pointwise_dropout, activation=activation)
        self.head = nn.Conv1d(C * 2, n_classes, 1, padding=0)

        self.model = nn.Sequential(
            self.stem,
            self.tower,
            self.dilated_residual_block,
            self.cropping,
            self.pointwise,
            self.head,
            nn.Sigmoid()
        )

        self.loss_fun = nn.BCELoss(reduction='mean')
        self.n_classes = n_classes
        self.accuracy = MultilabelAccuracy(num_labels=self.n_classes, average='macro')
        self.auroc = MultilabelAUROC(num_labels=self.n_classes, average='macro')
        self.save_hyperparameters(logger=log_hyperparams)

    def forward(self, x):
        a = self.model(x)

        return a

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x[0]  # TODO: why does the loaded data have an additional dimension in front (?)
        y = y[0]
        y_hat = self(x)
        loss = self.loss_fun(y_hat.squeeze(dim=-1), y)
        accuracy = self.accuracy(y_hat.squeeze(dim=-1), y)
        self.log("train_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # auroc = self.auroc(y_hat.squeeze(dim=-1),y.int())
        # self.log("train_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x[0]  # TODO: why does the loaded data have an additional dimension in front (?)
        y = y[0]
        y_hat = self(x)
        loss = self.loss_fun(y_hat.squeeze(dim=-1), y)
        accuracy = self.accuracy(y_hat.squeeze(dim=-1), y)
        auroc = self.auroc(y_hat.squeeze(dim=-1), y.int())
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self, lr: Optional[float] = None, weight_decay: Optional[float] = None):

        if lr is None:
            lr = float(self.hparams.lr)

        if weight_decay is None:
            weight_decay = float(self.hparams.weight_decay)

        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)

        return optimizer


class MaskedBCELoss(nn.BCELoss):
    def __init__(self, metadata_df, ignore_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.metadata_df = metadata_df.copy().reset_index(drop=True)
        self.metadata_df['proc_target_2'] = self.metadata_df.apply(
            lambda row: row['proc_Assay_lvl1'] if row['proc_target'] == 'Accessible DNA' else row['proc_target'],
            axis=1,
        )

        self.ignore_classes = ignore_classes
        if self.ignore_classes is not None:
            self.df_kept = self.metadata_df.loc[~self.metadata_df['File_accession'].isin(self.ignore_classes)]
            self.class_idxs = self.df_kept.index.values
            self.df_ignore = self.metadata_df.loc[self.metadata_df['File_accession'].isin(self.ignore_classes)]
            self.missing_class_idxs = self.df_ignore.index.values

            # select classes with the same target as the ignored classes, and group them by tissue
            # having the tissue counts we can account for tissue imbalance in the experiments
            # we do the same for tissue/target below
            self.same_target_weights = [0 for _ in range(len(self.metadata_df))]
            self.df_same_target = self.df_kept.loc[self.df_kept['proc_target_2'].isin(self.df_ignore['proc_target_2'].unique())]
            organ_freqs = self.df_same_target['Biosample_organ_slims'].value_counts()
            for i, row in self.metadata_df.iterrows():
                if (row['proc_target_2'] in self.df_ignore['proc_target_2'].unique()
                        and row['File_accession'] not in self.ignore_classes
                        and str(row['Biosample_organ_slims']).lower() != 'nan'):
                    self.same_target_weights[i] = 1 / organ_freqs[row['Biosample_organ_slims']]

            self.same_tissue_weights = [0 for _ in range(len(self.metadata_df))]
            self.df_same_tissue = self.df_kept.loc[self.df_kept['Biosample_organ_slims'].isin(self.df_ignore['Biosample_organ_slims'].unique())]
            target_freqs = self.df_same_tissue['proc_target_2'].value_counts()
            for i, row in self.metadata_df.iterrows():
                if (row['Biosample_organ_slims'] in self.df_ignore['Biosample_organ_slims'].unique()
                        and row['File_accession'] not in self.ignore_classes
                        and str(row['proc_target_2']).lower() != 'nan'):
                    self.same_tissue_weights[i] = 1 / target_freqs[row['proc_target_2']]

            self.lab_aggregation_weights = {}
            for col_sets in (
                    ('proc_target',),
                    ('proc_target', 'proc_Biosample_organ_slims'),
                    ('proc_target', 'proc_Biosample_organ_slims', 'proc_Biosample_life_stage'),
            ):
                W = np.zeros((len(self.missing_class_idxs), len(self.metadata_df)))

                for missing_idx_i, missing_idx in enumerate(self.missing_class_idxs):
                    md_row = self.metadata_df.iloc[missing_idx]

                    df = self.df_kept
                    for col in col_sets:
                        df = df.loc[(df[col] == md_row[col])]

                    if len(df) < 1:
                        W[missing_idx_i] = W_prev[missing_idx_i]
                    else:
                        lab_denominators = df.value_counts('Library_lab')
                        df['Library_lab_denominator'] = df.apply(
                            lambda row: 1 / lab_denominators[row['Library_lab']],
                            axis=1,
                        )
                        for k, v in df['Library_lab_denominator'].to_dict().items():
                            W[missing_idx_i, k] = v
                W_prev = W
                self.lab_aggregation_weights['-'.join(col_sets)] = W.T

    def forward(self, input, target):
        if self.ignore_classes is not None:
            input = torch.stack([input[:, i] for i in self.class_idxs], dim=1)
            target = torch.stack([target[:, i] for i in self.class_idxs], dim=1)
        return super().forward(input=input, target=target)


class AquaticDilated(SeqModel):
    """
    Basenji2-like model with binary output

    For an illustration of the model consider

    Avsec, Žiga, et al. "Effective gene expression prediction from sequence by integrating long-range interactions." Nature methods 18.10 (2021): 1196-1203.
    Extended Data Fig. 1

    The default parameters follow those presented there, except L2, crop, seq_len

        Original Parameters:

        C:  number of channels (original: 1536)
        L1: number of conv. layers + pooling in the Conv Tower (original: 6)
        L2: numver of residual dilated conv blocks (original: 11)
        D:  Dilation rate, rate by which dilation increases in every residual dilated conv block (original: 1.5)
        activation: activation function (original: "gelu")
        input_kernel_size:  kernel-size in the input layer (original: 15)
        tower_kernel_size:  kernel-size in the Conv Tower (original: 5)
        dilated_residual_kernel_size:   kernel-size in the residual dilated convs (original: 3)
        dilated_residual_dropout:       dropout rate in the residual dilated convs (original: 0.3)
        pointwise_dropout:  dropout in the Pointwise conv layer (original: 0.05)

        Other Parameters:

        out_pool_type:  if sequence is longer than 1 after cropping, which pooling type to use
        seq_order:      the order of the input sequence encoding
        crop:           how many positions to crop on ether size of the output after the residual dilated conv


    """

    def __init__(self,
                 seq_len: int = 2176,
                 n_classes: int = 1,
                 init_bias=None,
                 class_freq=None,
                 log_hyperparams=False,
                 lr=1e-3,
                 C=768,
                 C_embed=None,
                 L1=6,
                 L2=2,
                 D=1.5,
                 activation='gelu',
                 input_kernel_size=15,
                 tower_kernel_size=5,
                 dilated_residual_kernel_size=3,
                 dilated_residual_dropout=0.0,
                 pointwise_dropout=0.0,
                 out_pool_type=None,
                 seq_order=1,
                 crop=8,
                 weight_decay=0.,
                 auc_agg_size=10,
                 train_step_log_interval=10,
                 *args,
                 **kwargs
                 ):

        super().__init__(*args, **kwargs)

        in_channels = 4 ** seq_order
        self.save_hyperparameters(logger=log_hyperparams)

        self.h_seq_len = seq_len / 2 ** (L1 + 1)
        self.out_seq_len = self.h_seq_len - 2 * crop
        self.out_pool_type = out_pool_type

        assert self.out_seq_len > 0, f'Error: specified cropping is too large! {crop} * 2 >= {self.h_seq_len}'

        if self.out_seq_len != 1:
            print(f'Warnign: after cropping, the output sequence length ({self.out_seq_len}) is greater than one')
            assert out_pool_type is not None, 'Error: need to specify pooling type ("avg","max")'
            if out_pool_type == 'avg':
                self.out_pool = nn.AvgPool1d(self.out_seq_len)
            elif out_pool_type == 'max':
                self.out_pool = nn.MaxPool1d(self.out_seq_len)
            else:
                raise NotImplementedError(f'pooling type "{out_pool_type}" is not supported.')
        else:
            self.out_pool = nn.Identity()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=C // 2, kernel_size=15, padding='same'),
            ConvBlock(in_channels=C // 2, out_channels=C // 2, kernel_size=1, activation=activation, residual=True),
            nn.MaxPool1d(2, 2)
        )

        self.tower = Basenji2ResConvTower(C // 2, C, kernel_size=tower_kernel_size, L=L1, activation=activation)
        self.dilated_residual_block = Basenji2DilatedResidualBlock(C, kernel_size=dilated_residual_kernel_size,
                                                                   dropout=dilated_residual_dropout, D=D,
                                                                   channel_multiplier=1., L=L2, activation=activation)
        self.cropping = Cropping1d(crop)
        self.pointwise = ConvBlock(C, self.C_embed, kernel_size=1, dropout=pointwise_dropout, activation=activation)
        self.head = nn.Conv1d(self.C_embed, n_classes, 1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.model = nn.Sequential(
            self.stem,
            self.tower,
            self.dilated_residual_block,
            self.cropping,
            self.pointwise,
            self.head,
            self.sigmoid,
        )
        self.features = nn.Sequential(
            self.stem,
            self.tower,
            self.dilated_residual_block,
            self.cropping,
            self.pointwise,
        )
        self.predictions = nn.Sequential(
            self.head,
            self.sigmoid,
        )

        self.loss_fun = MaskedBCELoss(
            metadata_df=self.hparams.metadata_loader.df_raw,
            ignore_classes=self.hparams.ignore_classes,
            reduction='mean',
        )
        self.n_classes = n_classes

        self.accuracy = MultilabelAccuracy(num_labels=self.n_classes, average='macro')
        self.auroc = MultilabelAUROC(num_labels=self.n_classes, average='macro')

        # determines how many batches to use in batched auc calculation
        self.auc_agg_size = auc_agg_size
        # can be adjusted to adjust frequency of train/val AUROC logging
        self.train_step_log_interval = train_step_log_interval

        self.val_step_log_interval = 1
        self.val_aucs_agg = []

    @property
    def C_embed(self):
        return self.hparams.C_embed or self.hparams.C * 2

    def forward(self, x, return_features=False):

        if return_features:
            features = self.features(x)
            predictions = self.predictions(features).squeeze()
            features = features.squeeze()
            return predictions, features, features, features

        a = self.model(x)
        return a

    def get_logits(self, features_biological, features_technical):
        """
        Implemented for compatibility with downstream tasks which expect a method with such signature from the
        disentangled models.
        """
        logits = self.head(features_biological.unsqueeze(-1))
        preds = self.sigmoid(logits).squeeze()
        logits = logits.squeeze()

        return preds, logits, logits, logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x[0]  # TODO: why does the loaded data have an additional dimension in front (?)
        y = y[0]
        y_hat = self(x)
        loss = self.loss_fun(y_hat.squeeze(dim=-1), y)
        accuracy = self.accuracy(y_hat.squeeze(dim=-1), y)
        self.log("train_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # auroc = self.auroc(y_hat.squeeze(dim=-1),y.int())
        # self.log("train_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x[0]  # TODO: why does the loaded data have an additional dimension in front (?)
        y = y[0]
        y_hat = self(x)
        loss = self.loss_fun(y_hat.squeeze(dim=-1), y)
        accuracy = self.accuracy(y_hat.squeeze(dim=-1), y)
        if self.hparams.ignore_classes is None:
            # currently there is a memory leak issue when using a subset of classes, so ignore auroc in the
            # data imputation setting
            auroc = self.auroc(y_hat.squeeze(dim=-1), y.int())
            self.log("val/auroc", auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # used for AUC
        if (batch_idx % self.val_step_log_interval) == 0:
            # if self.hparams.ignore_classes is not None:
            #     y_hat, y = y_hat[:, self.loss_fun.class_idxs].squeeze(dim=-1), y[:, self.loss_fun.class_idxs]
            # TODO: fill up a pre-defined array instead of concatenating a list at the end of validation
            self.validation_step_preds.append(y_hat.detach().cpu())
            self.validation_step_targets.append(y.detach().cpu())

        if len(self.validation_step_preds) == self.auc_agg_size:
            all_preds = np.concatenate(self.validation_step_preds)
            all_targets = np.concatenate(self.validation_step_targets)

            auroc = metrics.multilabel_auroc_np(targets=all_targets, preds=all_preds)
            self.val_aucs_agg.append(auroc)
            # self.log("val/auroc", auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            del self.validation_step_preds[:]
            del self.validation_step_targets[:]

    def configure_optimizers(self, lr: Optional[float] = None, weight_decay: Optional[float] = None):

        if lr is None:
            lr = float(self.hparams.lr)

        if weight_decay is None:
            weight_decay = float(self.hparams.weight_decay)

        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)

        return optimizer

    def report_train_auroc(self):
        if len(self.training_step_preds) == self.auc_agg_size:
            with torch.no_grad():
                all_preds = np.concatenate(self.training_step_preds)
                all_targets = np.concatenate(self.training_step_targets)

                auroc = metrics.multilabel_auroc_np(targets=all_targets, preds=all_preds)
                self.log("train/auroc", auroc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

                # apparently that's the memory-safe way to empty arrays
                del self.training_step_preds[:]
                del self.training_step_targets[:]

    def on_validation_epoch_end(self):
        with torch.no_grad():

            # auroc logging
            if len(self.validation_step_preds) > 0:
                all_preds = np.concatenate(self.validation_step_preds)
                all_targets = np.concatenate(self.validation_step_targets)
                auroc = metrics.multilabel_auroc_np(targets=all_targets, preds=all_preds)
                del all_preds
                del all_targets
                weights = [1. for _ in self.val_aucs_agg]
                weights.append(len(self.validation_step_preds) / float(self.auc_agg_size))
                self.val_aucs_agg.append(auroc)
                auroc_agg = np.average(self.val_aucs_agg, weights=weights)
            else:
                auroc_agg = np.mean(self.val_aucs_agg)
            self.log("val/auroc", auroc_agg, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            del self.val_aucs_agg[:]

            # apparently that's the memory-safe way to empty arrays
            del self.validation_step_preds[:]
            del self.validation_step_targets[:]

    def imputation_step(self, x):
        features = self.features(x).squeeze()
        _, logits, _, _ = self.get_logits(
            features_biological=features,
            features_technical=features,
        )

        ret = {}
        logits_same_target = (logits * torch.FloatTensor(self.loss_fun.same_target_weights).to(logits.device)).mean(dim=-1)
        ret['same_target'] = torch.stack([logits_same_target for _ in self.loss_fun.missing_class_idxs], dim=1)
        logits_same_tissue = (logits * torch.FloatTensor(self.loss_fun.same_tissue_weights).to(logits.device)).mean(dim=-1)
        ret['same_tissue'] = torch.stack([logits_same_tissue for _ in self.loss_fun.missing_class_idxs], dim=1)
        ret['average'] = sum(l for l in ret.values())

        for k, v in self.loss_fun.lab_aggregation_weights.items():
            ret[f'lab_same_{k}'] = (logits @ torch.FloatTensor(v).to(logits.device))
        return ret


############################
# Basenji2 building blocks #
############################

class Cropping1d(nn.Module):
    """
    Symmetric cropping along position

        crop: number of positions to crop on either side
    """

    def __init__(self, crop):
        super().__init__()
        self.crop = crop

    def forward(self, x):
        return x[:, :, self.crop:-self.crop]


class ConvBlock(nn.Module):
    """
    ConvBlock Layer as defined in Extended Data Fig.1 (Avsec 2021)

    When residual = True, this becomes RConvBlock
    """

    def __init__(self, in_channels, out_channels=None, kernel_size=3, dropout=0., activation='relu', padding='same',
                 residual=False, bias=False, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.activation = activation
        self.dropout = dropout
        self.residual = residual

        if out_channels is None:
            self.out_channels = in_channels
        else:
            self.out_channels = out_channels

        if residual:
            assert self.out_channels == in_channels, 'Error: when residual = True, the number input channels and output channels must be the same.'
            assert padding == 'same', 'Error: need "same" padding when residual==True'

        self.conv1d_block = nn.Sequential(nn.BatchNorm1d(self.in_channels))

        if activation is not None:
            if activation == 'gelu':
                self.conv1d_block.append(nn.GELU(approximate='tanh'))
            elif activation == 'relu':
                self.conv1d_block.append(nn.ReLU())

        self.conv1d_block.append(
            nn.Conv1d(in_channels, self.out_channels, kernel_size, bias=bias, padding=padding, **kwargs))

        if dropout > 0:
            self.conv1d_block.append(nn.Dropout(dropout))

    def forward(self, x):
        if self.residual:
            return x + self.conv1d_block(x)
        else:
            return self.conv1d_block(x)


class ResidualDilatedConvBlock(nn.Module):
    """
    Dilated Residual ConvBlock as defined in Extended Data Fig.1, green (Avsec 2021)

    chains together 2 convolutions + dropout, has a residual connection

    the first convolution uses dilated kernels with dilation given by the dilation argument
    the second colvolution is a pointwise convolution

    convolutions have (in_channels * channel_multiplier) and (in_channels) channels, respectively

    in Basenji2, the channel_multiplier is 0.5

    """

    def __init__(self, in_channels, kernel_size=3, dropout=0., dilation=1, activation='relu', channel_multiplier=1.):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.activation = activation
        self.dropout = dropout
        self.dilation = 1
        self.channel_multiplier = channel_multiplier

        self.conv1d_block_1 = ConvBlock(self.in_channels, int(self.out_channels * self.channel_multiplier),
                                        kernel_size=kernel_size, dropout=0, dilation=dilation, activation=activation)
        self.conv1d_block_2 = ConvBlock(int(self.in_channels * self.channel_multiplier), self.out_channels,
                                        kernel_size=1, dropout=dropout, dilation=1, activation=activation)

    def forward(self, x):
        a1 = self.conv1d_block_1(x)
        a2 = self.conv1d_block_2(a1)

        return x + a2


class Basenji2ConvTower(nn.Module):
    """
    Conv Tower as defined in Extended Data Fig.1, green (Avsec 2021)

    The number of channels grows by a constant multiplier from in_channels to out_channels.

    in the original, there is no dropout.
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, dropout=0., activation='relu', L=6):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        self.channels_mult = (out_channels / in_channels) ** (1.0 / L)

        self.tower = nn.Sequential(*[
            nn.Sequential(
                ConvBlock(int(self.in_channels * self.channels_mult ** l),
                          int(self.in_channels * self.channels_mult ** (l + 1)), kernel_size, dropout=dropout,
                          activation=activation),
                nn.MaxPool1d(2, 2)
            ) for l in range(L)
        ])

    def forward(self, x):
        return self.tower(x)


class Basenji2ResConvTower(nn.Module):
    """
    Conv Tower as defined in Extended Data Fig.1, green (Avsec 2021)

    The number of channels grows by a constant multiplier from in_channels to out_channels.

    in the original, there is no dropout.
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, dropout=0., activation='relu', L=6):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        self.channels_mult = (out_channels / in_channels) ** (1.0 / L)

        self.tower = nn.Sequential(*[
            nn.Sequential(
                ConvBlock(int(self.in_channels * self.channels_mult ** l),
                          int(self.in_channels * self.channels_mult ** (l + 1)), kernel_size, dropout=dropout,
                          activation=activation),
                ConvBlock(int(self.in_channels * self.channels_mult ** (l + 1)),
                          int(self.in_channels * self.channels_mult ** (l + 1)), 1, dropout=dropout,
                          activation=activation, residual=True),
                nn.MaxPool1d(2, 2)
            ) for l in range(L)
        ])

    def forward(self, x):
        return self.tower(x)


class Basenji2DilatedResidualBlock(nn.Module):
    """
    Basenji 2 Dilated Residual Block

    chains together residual dilated convolution blocks, with dilation starting at 2 and growing by a factor of D in ever layer
    """

    def __init__(self, in_channels, kernel_size=3, dropout=0.0, activation='relu', L=4, D=1.5, channel_multiplier=1.):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.dropout = dropout
        self.D = D

        self.dilatedresidualblock = nn.Sequential(*[
            ResidualDilatedConvBlock(in_channels=in_channels, kernel_size=3, dropout=dropout, activation=activation,
                                     channel_multiplier=channel_multiplier, dilation=int(2 * D ** l))
            for l in range(L)
        ])

    def forward(self, x):
        return self.dilatedresidualblock(x)


# baseline model (oligonucleotide frequency)

class OligonucleotideModel(SeqModel):
    """
    Model that uses just oligonucleotide frequencies,

    when forward is called with return_freq = True, will return predictions and frequencies

    """

    def __init__(self, seq_order, n_classes, seq_len=None, batch_size=None):
        super().__init__()
        self.loss_fun = torch.nn.BCELoss(reduction='mean')
        self.onflinear = OligonucleotideFreqLinear(seq_order=seq_order, out_features=n_classes, bias=True)

    def configure_optimizers(self, lr=0.001, weight_decay=0.):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
        return optimizer

    def forward(self, x, return_freq=False):
        if return_freq:
            y_logit, frq = self.onflinear(x, return_freq)
            return sigmoid(y_logit), frq
        else:
            return sigmoid(self.onflinear(x, return_freq))

    def training_step(self, batch, batch_idx):

        x, y = batch
        x = x[0]
        y = y[0]

        y_hat = self.forward(x)
        loss = self.loss_fun(y_hat, y)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        x = x[0]
        y = y[0]

        y_hat = self.forward(x, return_freq=False)
        loss = self.loss_fun(y_hat, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)


class OligonucleotideFreqLinear(nn.Module):
    """
    calculates the (oligo-) nucleotide frequencies and feeds them through a linear layer (by default has no bias)
    """

    def __init__(self, seq_order, out_features, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_features=4 ** seq_order, out_features=out_features, bias=bias)

    def forward(self, x, return_freq=True):

        freq = (x.sum(dim=2, keepdims=True) / x.sum(dim=[1, 2], keepdims=True)).squeeze(-1)

        if return_freq:
            return self.linear(freq), freq
        else:
            return self.linear(freq)


# region Disentanglement
class LinearScaling(nn.Module):
    """ Layer that simply scales each column of the input"""

    def __init__(self, size_in, scale_init=1., scale_independent=False):
        super().__init__()
        self.size_in = size_in
        self.scale_independent = scale_independent
        if scale_independent:
            weights = torch.Tensor(size_in)
        else:
            weights = torch.Tensor(1)
        self.weights = nn.Parameter(weights)
        # initialize weights
        nn.init.constant_(self.weights, scale_init)

    def forward(self, x):
        if not self.scale_independent:
            return x @ (torch.eye(self.size_in) * self.weights)
        else:
            return x @ torch.diag(self.weights)


class MetadataColumnEmbedding(nn.Module):
    def __init__(
            self,
            var_def,
            var_vals,
    ):
        super().__init__()

        self.var_type = var_def['type']
        if self.var_type == 'category':
            labels = var_def['labels'] if 'labels' in var_def else None
            var_vals, self.vals_labels = data_metadata.categorize_multilabel_column(var_vals, return_labels=True,
                                                                                    labels=labels)
            var_vals = np.stack(var_vals)
        elif self.var_type in ('float', 'int'):
            var_vals = np.stack(var_vals.astype(float)).reshape((len(var_vals), 1))
            self.vals_mean = var_def['mean'] if 'mean' in var_def else var_vals.mean()
            self.vals_std = var_def['std'] if 'std' in var_def else var_vals.std()
            var_vals = (var_vals - self.vals_mean) / self.vals_std
        elif self.var_type == 'date.year':
            labels = var_def['labels'] if 'labels' in var_def else None
            var_vals, self.vals_labels = data_metadata.categorize_multilabel_column(
                pd.Series(pd.DatetimeIndex(var_vals).year), return_labels=True)
            var_vals = np.stack(var_vals)
        else:
            raise ValueError(f'Unknown variable type: {self.var_type}')
        self.dim_input = var_vals.shape[1]

        self.intermediate = var_def['intermediate']
        if self.intermediate == 'embed':
            self.dim_intermediate = var_def['intermediate_dim'] if 'intermediate_dim' in var_def else self.dim_input
            embedding = nn.Sequential(
                # for now hardcode the embedding model as a NN with a single hidden layer of dim = 2*embedd dim
                nn.Linear(in_features=self.dim_input, out_features=self.dim_intermediate * 2),
                nn.ReLU(),
                nn.Linear(in_features=self.dim_intermediate * 2, out_features=self.dim_intermediate * 2),
                nn.ReLU(),
                nn.Linear(in_features=self.dim_intermediate * 2, out_features=self.dim_intermediate)
            )
        elif self.intermediate == 'linear':
            self.dim_intermediate = var_def['intermediate_dim'] if 'intermediate_dim' in var_def else self.dim_input
            embedding = nn.Linear(in_features=self.dim_input, out_features=self.dim_intermediate)
        elif self.intermediate == 'rescale':
            assert self.var_type == 'category', '"rescale" is only meaningful for categorical variables'
            # allows a learnable scale parameter per column (i.e., multi-hot label)
            self.dim_intermediate = self.dim_input
            embedding = LinearScaling(self.dim_intermediate, scale_init=1., scale_independent=False)
        elif self.intermediate == 'rescale_indep':
            assert self.var_type == 'category', '"rescale" is only meaningful for categorical variables'
            # allows a learnable scale parameter per column (i.e., multi-hot label)
            self.dim_intermediate = self.dim_input
            embedding = LinearScaling(self.dim_intermediate, scale_init=1., scale_independent=True)
        elif self.intermediate is None:
            # forwards features directly without transformation
            self.dim_intermediate = self.dim_input
            embedding = nn.Identity()
        else:
            raise ValueError(f'Unknown embedding intermediate type: {self.intermediate}')
        assert hasattr(self,
                       'dim_intermediate')  # this has to be true in order to determine dimensions for the MetadataGroupEmbedding

        self.embedding = embedding
        self.register_buffer('var_vals', torch.FloatTensor(var_vals))

    def forward(self, x=None):
        if x is None:
            x = self.var_vals
        return self.embedding(x)


class MetadataGroupEmbedding(nn.Module):
    def __init__(
            self,
            group_definitions,
            mapping_config,
            df_metadata,
            embedding_size,
            linearly_embed_direct_features=False
    ):
        super().__init__()

        self.group_definitions = group_definitions
        self.mapping_config = mapping_config
        self.df_metadata = df_metadata
        self.embedding_size = embedding_size

        self.column_embeddings = nn.ModuleDict({
            var_name: MetadataColumnEmbedding(
                var_def=mapping_config[var_name],
                var_vals=df_metadata[var_name],
            ) for var_name in self.variables_all
        })
        if not linearly_embed_direct_features:
            self.mappings_direct = nn.ModuleDict({
                var_name: nn.Identity() for var_name in self.variables_direct
            })
        else:
            # this is the old (deprecated) way of incorporating the "direct" variables
            # TODO: this feature can be removed once we don't need the old checkpoints anymore
            self.mappings_direct = nn.ModuleDict({
                var_name: nn.Linear(
                    in_features=self.column_embeddings[var_name].dim_intermediate,
                    out_features=self.column_embeddings[var_name].dim_intermediate
                ) for var_name in self.variables_direct
            })
        self.mapping_interact = nn.Sequential(
            nn.Linear(in_features=self.dim_input_interact, out_features=self.dim_embedding_interact * 2),
            nn.ReLU(),
            nn.Linear(in_features=self.dim_embedding_interact * 2, out_features=self.dim_embedding_interact * 2),
            nn.ReLU(),
            nn.Linear(in_features=self.dim_embedding_interact * 2, out_features=self.dim_features_interact)
        )
        self.config_sanity_check()

    @property
    def variables_direct(self):
        return self.group_definitions['direct']['variables']

    @property
    def variables_interact(self):
        return self.group_definitions['interact']['variables']

    @property
    def variables_all(self):
        return set(self.variables_direct + self.variables_interact)

    @property
    def dim_input_interact(self):
        return sum(self.column_embeddings[var_name].dim_intermediate for var_name in self.variables_interact)

    @property
    def dim_embedding_interact(self):
        return self.group_definitions['interact']['out_dim']

    @property
    def dim_features_direct(self):
        return sum(self.column_embeddings[var_name].dim_intermediate for var_name in self.variables_direct)

    @property
    def dim_features_interact(self):
        return self.embedding_size - self.dim_features_direct

    def get_column_embeddings(self):
        return {
            var_name: embedding()
            for var_name, embedding in self.column_embeddings.items()
        }

    def get_feature_weights(self):
        column_embeddings = self.get_column_embeddings()

        feature_weights_direct = torch.cat([
            mapping_direct(column_embeddings[var_name]) for var_name, mapping_direct in self.mappings_direct.items()
        ], dim=1, )
        feature_weights_interact = self.mapping_interact(torch.cat([
            column_embeddings[var_name] for var_name in self.variables_interact
        ], dim=1, ))
        return torch.cat([feature_weights_direct, feature_weights_interact], dim=1)

    def config_sanity_check(self):
        for variable in self.group_definitions['direct']['variables']:
            if self.mapping_config[variable]['intermediate'] is None:
                print(
                    f'Warning: directly forwarded variable "{variable}" has no intermediate mapping defined (consider at least "linear" or "rescale_indep").')
        for variable in self.group_definitions['interact']['variables']:
            if self.mapping_config[variable]['type'] in ('float', 'int'):
                print(
                    f'Warning: interacting metadata column "{variable}" has type float or int. Variables like these can easily act as unique sample identifiers.')


class MetadataEmbedding(nn.Module):
    def __init__(
            self,
            metadata_loader,
            metadata_mapping_config,
            embedding_size_biological,
            embedding_size_technical,
            linearly_embed_direct_features=False
    ):
        super().__init__()

        self.metadata_loader = metadata_loader
        self.embedding_size_biological = embedding_size_biological
        self.embedding_size_technical = embedding_size_technical
        self.mapping_config = yaml.safe_load(open(metadata_mapping_config))

        self.embedding_biological = MetadataGroupEmbedding(
            mapping_config=self.mapping_config,
            group_definitions=self.mapping_config['biological_features'],
            df_metadata=self.metadata_loader.df_raw,
            embedding_size=embedding_size_biological,
            linearly_embed_direct_features=linearly_embed_direct_features
        )
        self.embedding_technical = MetadataGroupEmbedding(
            mapping_config=self.mapping_config,
            group_definitions=self.mapping_config['technical_features'],
            df_metadata=self.metadata_loader.df_raw,
            embedding_size=embedding_size_technical,
            linearly_embed_direct_features=linearly_embed_direct_features
        )

    @property
    def weights_biological(self):
        return self.embedding_biological.get_feature_weights()

    @property
    def weights_technical(self):
        return self.embedding_technical.get_feature_weights()

    def features_to_logits_biological(self, features):
        W = self.weights_biological
        return features @ W.T

    def features_to_logits_technical(self, features):
        W = self.weights_technical
        return features @ W.T

    def dump_mapping_config(self, file=None):

        mapping_config = self.mapping_config.copy()

        for k, _ in self.mapping_config.items():
            if k in ['biological_features', 'technical_features']:
                continue
            for e in (self.embedding_biological.column_embeddings, self.embedding_technical.column_embeddings):
                if k in e:
                    colembedding = e[k]
                    if colembedding.var_type == 'category':
                        mapping_config[k]['labels'] = colembedding.vals_labels
                    elif colembedding.var_type == 'date.year':
                        mapping_config[k]['labels'] = colembedding.vals_labels
                    elif colembedding.var_type in ['float', 'int']:
                        mapping_config[k]['mean'] = float(colembedding.vals_mean)
                        mapping_config[k]['std'] = float(colembedding.vals_std)

        if file is None:
            return mapping_config
        else:
            with open(file, 'x') as outfile:
                yaml.dump(mapping_config, outfile, default_flow_style=False, sort_keys=False)


def z_score(x):
    x = x - x.mean(0)
    x = x / x.std(0)
    x = torch.nan_to_num(x, posinf=0, neginf=0)

    return x


class _IndependentEmbeddingsMixin:
    """
    This class assumes the original model class defines a 'model' attribute, where the ouput of model[:-2]
    (i.e., of the layer third to last) constitutes the latent features.

    biological_subspace_ratio - defines the ratio of the sizes of the biological and technical embeddings, e.g.,
    with a latent size of 10, and the biological_subspace_ratio=0.8, the biological subspace will have a dimensionality
    of 8, and the technical one of 2.

    Alternatively, the number of features can be set explicitly with biological_subspace_n and/or technical_subspace_n

    If biological_subspace_ratio, biological_subspace_n and technical_subspace_n are all None, the model will instead determine the dimensionalities by itself, by learning mappings of the
    features into two orthogonal spaces.
    """

    # TODO: is the behavior above still supported (automatically learning the ratio?)

    def __init__(
            self,
            num_hidden_embedding=0,
            dim_hidden_embedding=256,
            coeff_cov=1,
            biological_subspace_ratio=.5,
            metadata_mapping_config=None,
            num_batches_for_variance_estimation=1000,
            num_steps_discriminator=1,
            use_discriminator=False,
            use_predictor=False,
            lr_discriminator=1e-3,
            dim_hidden_discriminator=1024,
            num_hidden_discriminator=3,
            cov_norm=True,
            cov_and_adv=False,
            num_random_projections=0,
            dim_random_projections=10,
            linearly_embed_direct_features=False,
            labels_encoder=False,
            num_hidden_labels_encoder=2,
            biological_subspace_n=None,
            technical_subspace_n=None,
            learn_signal_to_noise=False,
            *args,
            **kwargs,
    ):
        # not the most elegant thing to do - it captures all parameters to this method, other than self, args,
        # and kwargs, and passes them to the parent's init
        # this way we do not have to type all the parameters by hand, but still get them stored in self.hparams
        all_args = dict(locals())
        del all_args['self']
        del all_args['args']
        del all_args['kwargs']
        super().__init__(*args, **all_args, **kwargs)

        # this is for backwards compatibility with when loading older checkpoints
        # TODO change these fields in the parent class (AquaticDilated) to a property, instead of an additional module
        del self.features
        del self.predictions

        assert self.hparams.metadata_loader is not None

        self.latent_size = self.model[-2].in_channels
        assert self.latent_size % 2 == 0
        self.model = self.model[:-2]

        if biological_subspace_ratio is None:
            print(self.latent_size)
            self.subspace_size_biological, self.subspace_size_technical = self._calc_subspace_size(
                biological_subspace_n, technical_subspace_n)
        else:
            # subspace size is set by ratio
            if technical_subspace_n is not None:
                raise ValueError('cannot set both biological_subspace_ratio and technical_subspace_n')
            if biological_subspace_n is not None:
                raise ValueError('cannot set both biological_subspace_ratio and biological_subspace_n')
            self.subspace_size_biological = int(self.latent_size * biological_subspace_ratio)
            self.subspace_size_technical = self.latent_size - self.subspace_size_biological
        if (biological_subspace_ratio is None) and (biological_subspace_n is None) and (technical_subspace_n is None):
            print('learning subspace dimensions automatically. Warning: this may no longer work')
            self.learn_subspace_ratio = True
        else:
            self.learn_subspace_ratio = False
        print('Biological subspace size: ' + str(self.subspace_size_biological) + '\nTechnical subspace size: ' + str(
            self.subspace_size_technical))
        self.n_pc_plot = min(10, min(self.subspace_size_biological, self.subspace_size_technical))

        self.metadata_embedding = MetadataEmbedding(
            metadata_loader=self.hparams.metadata_loader,
            embedding_size_biological=self.subspace_size_biological,
            embedding_size_technical=self.subspace_size_technical,
            metadata_mapping_config=metadata_mapping_config,
            linearly_embed_direct_features=linearly_embed_direct_features
        )
        self.bias = nn.Parameter(data=torch.zeros((self.hparams.n_classes,)).float())
        if learn_signal_to_noise:
            self.stn_bio = nn.Parameter(data=torch.zeros((self.hparams.n_classes,)).float())
            self.stn_tech = nn.Parameter(data=torch.zeros((self.hparams.n_classes,)).float())
            self.learn_stn = True
        else:
            self.learn_stn = False
        self.bn_features = nn.BatchNorm1d(num_features=self.latent_size)
        if self.learn_subspace_ratio:
            self.mapping_biological, self.mapping_technical = (nn.Sequential(
                nn.Linear(in_features=self.latent_size, out_features=self.latent_size),
                nn.BatchNorm1d(num_features=self.latent_size),
            ) for _ in range(2))

        if use_predictor:
            self.automatic_optimization = False
            if self.hparams.num_hidden_discriminator == 0:
                self.predictor_bio = nn.Sequential(
                    nn.Linear(in_features=self.subspace_size_technical, out_features=self.subspace_size_biological),
                )
                self.predictor_tech = nn.Sequential(
                    nn.Linear(in_features=self.subspace_size_biological, out_features=self.subspace_size_technical),
                )
            else:
                def _make_layers():
                    layers = []
                    for _ in range(self.hparams.num_hidden_discriminator - 1):
                        layers.append(
                            nn.Linear(in_features=dim_hidden_discriminator, out_features=dim_hidden_discriminator))
                        layers.append(nn.BatchNorm1d(num_features=dim_hidden_discriminator)),
                        layers.append(nn.LeakyReLU())
                    return layers
                self.predictor_bio = nn.Sequential(
                    nn.Linear(in_features=self.subspace_size_technical, out_features=dim_hidden_discriminator),
                    nn.BatchNorm1d(num_features=dim_hidden_discriminator),
                    nn.LeakyReLU(),
                    *_make_layers(),
                    nn.Linear(in_features=dim_hidden_discriminator, out_features=self.subspace_size_biological)
                )
                self.predictor_tech = nn.Sequential(
                    nn.Linear(in_features=self.subspace_size_biological, out_features=dim_hidden_discriminator),
                    nn.BatchNorm1d(num_features=dim_hidden_discriminator),
                    nn.LeakyReLU(),
                    *_make_layers(),
                    nn.Linear(in_features=dim_hidden_discriminator, out_features=self.subspace_size_technical)
                )
        if use_discriminator:
            self.automatic_optimization = False
            if self.hparams.num_hidden_discriminator == 0:
                self.discriminator = nn.Sequential(
                    nn.Linear(in_features=self.latent_size, out_features=1),
                )
            else:
                self.discriminator = nn.Sequential(
                    nn.Linear(in_features=self.latent_size, out_features=dim_hidden_discriminator),
                    nn.LeakyReLU(),
                    *[nn.Sequential(
                        nn.Linear(in_features=dim_hidden_discriminator, out_features=dim_hidden_discriminator),
                        nn.LeakyReLU(),
                    ) for _ in range(self.hparams.num_hidden_discriminator - 1)],
                    nn.Linear(in_features=dim_hidden_discriminator, out_features=1)
                )
        if labels_encoder:
            self.model = nn.Sequential(
                nn.Linear(in_features=self.hparams.n_classes, out_features=dim_hidden_embedding),
                nn.BatchNorm1d(num_features=dim_hidden_embedding),
                nn.LeakyReLU(),
                *(
                    nn.Sequential(
                        nn.Linear(in_features=dim_hidden_embedding, out_features=dim_hidden_embedding),
                        nn.BatchNorm1d(num_features=dim_hidden_embedding),
                        nn.LeakyReLU(),
                    ) for _ in range(num_hidden_labels_encoder - 1)),
                nn.Linear(in_features=dim_hidden_embedding, out_features=self.latent_size),
            )
        self.val_step_log_interval = 1
        self.val_aucs_agg = []

        # log norm_ratio
        self.validation_step_preds_biological = []
        self.validation_step_preds_technical = []
        self.variance_ratio_log = []
        now = datetime.datetime.now()
        date_string = now.strftime("%Y-%m-%d_%H-%M")
        base_path = f"norm_ratio_log/{date_string}_"
        if wandb.run is not None:
            base_path = base_path + str(wandb.run.id) + '_'
        # self.variance_ratio_file_path = base_path + "variances_ratio_log.csv"
        # self.variances_file_path = base_path + "variances_log.csv"
        # self.variance_ratio_group_file_path = base_path + "variances_ratio_group_log.csv"

        self.indep_batches_train = []
        self.indep_batches_val = []

    def _calc_subspace_size(self, nbio, ntech):
        if nbio is None:
            if ntech is None:
                return self.latent_size, self.latent_size
            else:
                return int(self.latent_size - ntech), int(ntech)
        else:
            if ntech is None:
                return int(nbio), int(self.latent_size - nbio)
            else:
                assert int(nbio) + int(
                    ntech) == self.latent_size, f'Error: manually specified number of bio ({nbio}) and tech ({ntech}) features do not add up to latent size ({self.latent_size})'
                return int(nbio), int(ntech)

    def configure_optimizers(self, lr: Optional[float] = None, weight_decay: Optional[float] = None):

        if lr is None:
            lr = float(self.hparams.lr)

        if weight_decay is None:
            weight_decay = float(self.hparams.weight_decay)

        if self.hparams.use_predictor:
            params = set(self.parameters()) - set(self.predictor_bio.parameters()) - set(
                self.predictor_tech.parameters())
            optimizer = torch.optim.Adam(params=params, lr=lr, weight_decay=weight_decay, amsgrad=True)

            opt_d = torch.optim.Adam(
                params=set(list(self.predictor_bio.parameters()) + list(self.predictor_tech.parameters())),
                lr=self.hparams.lr_discriminator,
                weight_decay=weight_decay,
                amsgrad=True,
            )
            return [optimizer, opt_d]

        if self.hparams.use_discriminator:
            params = set(self.parameters()) - set(self.discriminator.parameters())
            optimizer = torch.optim.Adam(params=params, lr=lr, weight_decay=weight_decay, amsgrad=True)

            opt_d = torch.optim.Adam(
                params=self.discriminator.parameters(),
                lr=self.hparams.lr_discriminator,
                weight_decay=weight_decay,
                amsgrad=True,
            )
            return [optimizer, opt_d]
        else:
            optimizer = torch.optim.Adam(params=self.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
            return optimizer

    def get_features(self, x):
        features = self.model(x)
        # batch_norm is useful here to constrain the features to a variance of 1
        # this makes it easier to penalize the covariance matrix
        # (otherwise the model can just scale the features to an arbitrarily low magnitude)
        features = self.bn_features(features)
        features = features.reshape(features.shape[0], self.latent_size)

        if self.learn_subspace_ratio:
            # map the features to new spaces via learned mappings
            features_biological = self.mapping_biological(features)
            features_technical = self.mapping_technical(features)
        else:
            # separate the features into subspaces
            features_technical = features[:, :self.subspace_size_technical]
            features_biological = features[:, self.subspace_size_technical:]

        return features, features_biological, features_technical

    def get_logits(self, features_biological, features_technical):
        logits_biological = self.metadata_embedding.features_to_logits_biological(features_biological)
        logits_technical = self.metadata_embedding.features_to_logits_technical(features_technical)

        if self.learn_stn:
            logits_biological = logits_biological * sigmoid(self.stn_bio)
            logits_technical = logits_technical * sigmoid(self.stn_tech)

        logits = logits_biological + logits_technical + self.bias

        preds = sigmoid(logits)

        return preds, logits, logits_biological, logits_technical

    def forward(self, x, compute_covariance=False, return_features=False, return_logits=False):
        features, features_biological, features_technical = self.get_features(x)

        preds, logits, logits_biological, logits_technical = self.get_logits(
            features_biological=features_biological,
            features_technical=features_technical,
        )

        # TODO refactor this into returning preds + a dict of all possible outputs, instead of having multiple if/else
        # paths
        batch_size = features_technical.shape[0]
        if compute_covariance:
            # happens during model fitting
            if self.learn_subspace_ratio:
                # enforce orthogonality of the biological and technical mappings
                cov = self.mapping_biological[0].weight @ self.mapping_technical[0].weight.T
            elif self.hparams.num_random_projections > 0:
                proj_bio, proj_tech = [], []
                for _ in range(self.hparams.num_random_projections):
                    proj_bio.append(
                        random_projections(features_biological, n_components=self.hparams.dim_random_projections))
                    proj_tech.append(
                        random_projections(features_technical, n_components=self.hparams.dim_random_projections))
                proj_bio = torch.cat(proj_bio, dim=1)
                proj_tech = torch.cat(proj_tech, dim=1)
                cov = z_score(proj_tech).T @ z_score(proj_bio) / batch_size
            else:
                # penalize the cross-covariance matrix between the two subspaces
                cov = z_score(features_technical).T @ z_score(features_biological) / batch_size

            if self.hparams.cov_norm:
                norm = torch.linalg.matrix_norm(cov)
            else:
                # L1 norm
                norm = torch.abs(cov).sum()
            if return_logits:
                if return_features:
                    # TODO: I think applying sigmoid to the logits only makes sense if the bias is added!
                    return preds, norm, cov, features, features_biological, features_technical, sigmoid(
                        logits_biological), sigmoid(logits_technical)
                return preds, norm, cov, sigmoid(logits_biological), sigmoid(logits_technical)
            return preds, norm, cov
        if return_features:
            return preds, features, features_biological, features_technical
        return preds

    def make_adversarial_batch(self, bio, tech):
        batch_true = torch.cat([bio, tech], dim=1)
        tech_shuffled = tech[torch.randperm(bio.shape[0])]
        batch_fake = torch.cat([bio, tech_shuffled], dim=1)

        x = torch.cat([batch_true, batch_fake], dim=0)

        y = torch.ones_like(x[:, 0]).unsqueeze(1)
        y[y.shape[0] // 2:] = 0

        return x, y

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x[0]  # TODO: why does the loaded data have an additional dimension in front (?)
        y = y[0]

        if self.hparams.labels_encoder:
            x = y

        y_hat, indep_value, cov, _, features_biological, features_technical, preds_bio, preds_tech = self.forward(
            x,
            compute_covariance=True,
            return_logits=True,
            return_features=True,
        )
        indep_value_cov = indep_value
        x_adv, y_adv = self.make_adversarial_batch(bio=features_biological, tech=features_technical)

        if self.hparams.use_predictor:
            g_opt, d_opt = self.optimizers()
            bio_hat = self.predictor_bio(features_technical)
            tech_hat = self.predictor_bio(features_biological)
            loss_predictor_bio = total_R2(bio_hat, features_biological)
            loss_predictor_tech = total_R2(tech_hat, features_technical)
            if self.hparams.cov_and_adv:
                indep_value = indep_value + loss_predictor_tech + loss_predictor_bio
            else:
                indep_value = loss_predictor_bio + loss_predictor_tech

        if self.hparams.use_discriminator:
            g_opt, d_opt = self.optimizers()
            y_adv_hat = self.discriminator(x_adv)
            if self.hparams.cov_and_adv:
                indep_value = indep_value + F.binary_cross_entropy_with_logits(input=y_adv_hat, target=1 - y_adv)
            else:
                indep_value = F.binary_cross_entropy_with_logits(input=y_adv_hat, target=1 - y_adv)

        loss_pred = self.loss_fun(y_hat.squeeze(dim=-1), y)
        loss = loss_pred + self.hparams.coeff_cov * indep_value
        accuracy = self.accuracy(y_hat.squeeze(dim=-1), y)

        if self.hparams.use_discriminator or self.hparams.use_predictor:
            g_opt.zero_grad()
            if batch_idx % self.hparams.num_steps_discriminator == 0:
                self.manual_backward(loss)
                g_opt.step()

        # do not log them for training step to speed up training/reduce memory footprint
        if (batch_idx % self.train_step_log_interval) == 0 and False:
            if self.hparams.ignore_classes is not None:
                y_hat = torch.stack([y_hat[:, i] for i in self.loss_fun.class_idxs], dim=1)
                y = torch.stack([y[:, i] for i in self.loss_fun.class_idxs], dim=1)
            self.training_step_preds.append(y_hat.detach().cpu().numpy())
            self.training_step_targets.append(y.detach().cpu().numpy())

        self.report_train_auroc()

        self.log("train/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/loss_pred", loss_pred, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/loss_indep_value", indep_value, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/loss_cov", indep_value_cov, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if self.hparams.use_predictor:
            bio_hat = self.predictor_bio(features_technical.detach())
            tech_hat = self.predictor_bio(features_biological.detach())
            loss_predictor_bio = total_R2(bio_hat, features_biological.detach())
            loss_predictor_tech = total_R2(tech_hat, features_technical.detach())
            loss_predictor = -(loss_predictor_bio + loss_predictor_tech)
            self.log("train/loss_predictor_bio", loss_predictor_bio, on_step=True, on_epoch=True, prog_bar=True,
                     logger=True)
            self.log("train/loss_predictor_tech", loss_predictor_tech, on_step=True, on_epoch=True, prog_bar=True,
                     logger=True)

            d_opt.zero_grad()
            self.manual_backward(loss_predictor)
            d_opt.step()

        if self.hparams.use_discriminator:
            y_adv_hat = self.discriminator(x_adv.detach())
            loss = F.binary_cross_entropy_with_logits(input=y_adv_hat, target=y_adv)
            self.log("train/loss_discriminator", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            acc = torchmetrics.functional.accuracy(preds=y_adv_hat, target=y_adv, task='binary')
            self.log("train/accuracy_discriminator", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            d_opt.zero_grad()
            self.manual_backward(loss)
            d_opt.step()

        return loss

    def on_validation_epoch_start(self):
        if len(self.val_aucs_agg) > 0:
            self.val_aucs_agg = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x[0]  # TODO: why does the loaded data have an additional dimension in front (?)
        y = y[0]

        if self.hparams.labels_encoder:
            x = y

        y_hat, norm, cov, _, features_biological, features_technical, preds_bio, preds_tech = self.forward(
            x,
            compute_covariance=True,
            return_logits=True,
            return_features=True,
        )
        loss_pred = self.loss_fun(y_hat.squeeze(dim=-1), y)
        loss = loss_pred + self.hparams.coeff_cov * norm
        accuracy = self.accuracy(y_hat.squeeze(dim=-1), y)

        # used for visualizing / measuring independence
        x_adv, y_adv = self.make_adversarial_batch(bio=features_biological, tech=features_technical)
        # TODO: fill up a pre-defined array instead of concatenating a list at the end of validation
        self.indep_batches_val.append((x_adv, y_adv))
        num_indep_batches = 50
        if len(self.indep_batches_val) >= num_indep_batches:
            if batch_idx < num_indep_batches * 2 and wandb.run is not None:
                # extract bio and tech features
                features = torch.cat([b[0][:b[0].shape[0] // 2] for b in self.indep_batches_val], dim=0)
                bio, tech = features[:, :self.subspace_size_biological], features[:, self.subspace_size_biological:]
                fig = pca_cc_plot(bio, tech, num_comp=self.n_pc_plot)
                wandb.log({'PCA_bio_tech_CC': wandb.Image(fig)})
                fig = cca_plot(bio, tech, num_comp=self.n_pc_plot)
                wandb.log({'CCA_bio_tech': wandb.Image(fig)})

            rf_acc = eval_independence_rf(self.indep_batches_val)
            del self.indep_batches_val[:]

            self.log('val/rf_accuracy', rf_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # used for bio / tech variance ratio
        if len(self.validation_step_preds_biological) < self.hparams.num_batches_for_variance_estimation:
            # TODO: fill up a pre-defined array instead of concatenating a list at the end of validation
            self.validation_step_preds_biological.append(preds_bio.detach().cpu())
            self.validation_step_preds_technical.append(preds_tech.detach().cpu())

        # used for AUC
        if (batch_idx % self.val_step_log_interval) == 0:
            if self.hparams.ignore_classes is not None:
                y_hat = torch.stack([y_hat[:, i] for i in self.loss_fun.class_idxs], dim=1)
                y = torch.stack([y[:, i] for i in self.loss_fun.class_idxs], dim=1)
                # y_hat, y = y_hat[:, self.loss_fun.class_idxs], y[:, self.loss_fun.class_idxs]
            # TODO: fill up a pre-defined array instead of concatenating a list at the end of validation
            self.validation_step_preds.append(y_hat.detach().cpu())
            self.validation_step_targets.append(y.detach().cpu())

        if len(self.validation_step_preds) == self.auc_agg_size:
            all_preds = np.concatenate(self.validation_step_preds)
            all_targets = np.concatenate(self.validation_step_targets)

            auroc = metrics.multilabel_auroc_np(targets=all_targets, preds=all_preds)
            self.val_aucs_agg.append(auroc)
            # self.log("val/auroc", auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            del self.validation_step_preds[:]
            del self.validation_step_targets[:]

        self.log("val/accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/loss_pred", loss_pred, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/loss_cov", norm, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx):
        return self.forward(x=batch, return_features=True)

    def variance_ratio_logging(self, variances_biological, variances_technical):

        self.log('ratio_of_mean_variances_bio/tech', variances_biological.mean() / variances_technical.mean(),
                 on_epoch=True, prog_bar=True, logger=True)
        self.log(
            'mean_contribution_of_bio_variance',
            (variances_biological / (variances_technical + variances_biological)).mean(),
            on_epoch=True, prog_bar=True, logger=True,
        )

        variance_values = variances_biological / variances_technical
        self.log('mean_variances_ratio_bio/tech', variance_values.mean(), on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        with torch.no_grad():

            # auroc logging 
            if len(self.validation_step_preds) > 0:
                all_preds = np.concatenate(self.validation_step_preds)
                all_targets = np.concatenate(self.validation_step_targets)
                auroc = metrics.multilabel_auroc_np(targets=all_targets, preds=all_preds)
                del all_preds
                del all_targets
                weights = [1. for _ in self.val_aucs_agg]
                weights.append(len(self.validation_step_preds) / float(self.auc_agg_size))
                self.val_aucs_agg.append(auroc)
                auroc_agg = np.average(self.val_aucs_agg, weights=weights)
            else:
                auroc_agg = np.mean(self.val_aucs_agg)
            self.log("val/auroc", auroc_agg, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            del self.val_aucs_agg[:]

            # apparently that's the memory-safe way to empty arrays
            del self.validation_step_preds[:]
            del self.validation_step_targets[:]

            # log the variances of predictions based on the biological and the technical features
            preds_bio = np.concatenate(self.validation_step_preds_biological)
            preds_tech = np.concatenate(self.validation_step_preds_technical)
            variances_bio = preds_bio.var(axis=0)
            variances_tech = preds_tech.var(axis=0)

            self.variance_ratio_logging(variances_bio, variances_tech)
            self.log('variance_preds_biological', variances_bio.mean(), on_epoch=True, prog_bar=True, logger=True)
            self.log('variance_preds_technical', variances_tech.mean(), on_epoch=True, prog_bar=True, logger=True)

            del self.validation_step_preds_biological[:]
            del self.validation_step_preds_technical[:]

            del variances_bio
            del variances_tech
            del preds_bio
            del preds_tech

            if hasattr(self, 'indep_batches_val'):
                del self.indep_batches_val[:]

            gc.collect()

    def imputation_step(self, x):
        features, features_biological, features_technical = self.get_features(x)
        _, logits, logits_biological, logits_technical = self.get_logits(
            features_biological=features_biological,
            features_technical=features_technical,
        )
        logits = logits - self.bias

        ret = {}
        logits_same_target = (logits * torch.FloatTensor(self.loss_fun.same_target_weights).to(logits.device)).mean(
            dim=-1)
        ret['same_target'] = torch.stack([logits_same_target for _ in self.loss_fun.missing_class_idxs], dim=1)
        logits_same_tissue = (logits * torch.FloatTensor(self.loss_fun.same_tissue_weights).to(logits.device)).mean(
            dim=-1)
        ret['same_tissue'] = torch.stack([logits_same_tissue for _ in self.loss_fun.missing_class_idxs], dim=1)
        ret['average'] = sum(l for l in ret.values())

        logits_same_target = (
                    logits_biological * torch.FloatTensor(self.loss_fun.same_target_weights).to(logits.device)).mean(
            dim=-1)
        ret['same_target_bio'] = torch.stack([logits_same_target for _ in self.loss_fun.missing_class_idxs], dim=1)
        logits_same_target = (
                    logits_technical * torch.FloatTensor(self.loss_fun.same_target_weights).to(logits.device)).mean(
            dim=-1)
        ret['same_target_tech'] = torch.stack([logits_same_target for _ in self.loss_fun.missing_class_idxs], dim=1)

        for k, v in self.loss_fun.lab_aggregation_weights.items():
            ret[f'lab_same_{k}'] = (logits @ torch.FloatTensor(v).to(logits.device))
            ret[f'lab_same_bio_{k}'] = (logits_biological @ torch.FloatTensor(v).to(logits.device))
            ret[f'lab_same_tech_{k}'] = (logits_technical @ torch.FloatTensor(v).to(logits.device))

        ret.update({
            'MFD': logits[:, self.loss_fun.missing_class_idxs],
            'MFD_bio': logits_biological[:, self.loss_fun.missing_class_idxs],
            'MFD_tech': logits_technical[:, self.loss_fun.missing_class_idxs],
        })

        return ret


def random_projections(x, n_components):
    transformer = random_projection.GaussianRandomProjection(n_components=n_components)
    transformer.fit(x.detach().cpu().numpy())
    return x @ torch.FloatTensor(transformer.components_.T).to(x.device)


def eval_independence_rf(batches):
    X, y = torch.cat([b[0] for b in batches], dim=0), torch.cat([b[1] for b in batches], dim=0)
    X, y = X.detach().cpu().numpy(), y.detach().cpu().numpy().squeeze()
    idxs = np.random.permutation(len(X))
    # shuffle examples
    X, y = X[idxs], y[idxs]

    num_train = int(len(X) * .7)

    X_train, y_train = X[:num_train], y[:num_train]
    X_test, y_test = X[num_train:], y[num_train:]

    forest = RandomForestClassifier(max_depth=4).fit(X_train, y_train)
    rf_accuracy = forest.score(X_test, y_test)
    return rf_accuracy


def total_R2(preds, target):
    preds = (preds - preds.mean(0)) / (preds.std(0))
    preds = torch.nan_to_num(preds)
    target = (target - target.mean(0)) / (target.std(0))
    target = torch.nan_to_num(target)

    cc = (preds.T @ target / preds.shape[0]).clip(min=-1, max=1).pow(2)
    cc = torch.tril(torch.triu(cc, diagonal=0), diagonal=0).sum()
    return cc


def pca_cc_plot(x, y, num_comp=10):
    x, y = x.detach().cpu().numpy(), y.detach().cpu().numpy()
    pca_bio = PCA(n_components=num_comp).fit(x)
    pca_tech = PCA(n_components=num_comp).fit(y)

    cc = np.corrcoef(pca_bio.transform(x), pca_tech.transform(y), rowvar=False)[num_comp:, :num_comp]
    fig = plt.figure(figsize=(7, 7))
    sns.heatmap(np.abs(cc), annot=True, fmt=".2f", vmin=0, vmax=1)
    fig.axes[0].set_yticklabels(['{:.2e}'.format(Decimal(float(x))) for x in pca_bio.explained_variance_ratio_])
    fig.axes[0].set_xticklabels(['{:.2e}'.format(Decimal(float(x))) for x in pca_tech.explained_variance_ratio_])
    fig.axes[0].xaxis.set_tick_params(labelsize=6)
    fig.axes[0].yaxis.set_tick_params(labelsize=6)
    plt.close()

    return fig


def cca_plot(x, y, num_comp=10):
    x, y = x.detach().cpu().numpy(), y.detach().cpu().numpy()
    cca_bio, cca_tech = CCA(n_components=num_comp).fit_transform(x, y)

    cc = np.corrcoef(cca_bio, cca_tech, rowvar=False)[num_comp:, :num_comp]
    fig = plt.figure(figsize=(7, 7))
    sns.heatmap(np.abs(cc), annot=True, fmt=".2f", vmin=0, vmax=1)
    plt.close()

    return fig


class IEAquaticDilated(_IndependentEmbeddingsMixin, AquaticDilated):
    pass


class _LatentPredictorMixin:
    def __init__(
            self,
            labels_encoder_checkpoint,
            *args,
            **kwargs,
    ):
        all_args = dict(locals())
        del all_args['self']
        del all_args['args']
        del all_args['kwargs']
        super().__init__(*args, **all_args, **kwargs)

        labels_encoder_checkpoint = Path(labels_encoder_checkpoint)
        if not labels_encoder_checkpoint.name.endswith('.ckpt'):
            # we can pass either the actual checkpoint file or a directory - in the latter case we will search for
            # a .ckpt file inside
            labels_encoder_checkpoint = next(labels_encoder_checkpoint.glob('*.ckpt'))
        self.labels_encoder = IEAquaticDilated.load_from_checkpoint(labels_encoder_checkpoint)

        self.model = self.model[:-2]
        assert self.model[-1].out_channels == self.labels_encoder.latent_size
        self.latent_size = self.labels_encoder.latent_size

        self.bn_features = nn.BatchNorm1d(num_features=self.latent_size)

    def get_features(self, x):
        features = self.model(x)
        features = self.bn_features(features)
        features = features.reshape(features.shape[0], self.latent_size)

        if hasattr(self.labels_encoder.hparams, 'learn_subspace_ratio'):
            # new behavior
            if self.labels_encoder.hparams.learn_subspace_ratio == True:
                raise NotImplementedError()
        else:
            # old behavior
            if self.labels_encoder.hparams.biological_subspace_ratio is None:
                raise NotImplementedError()

        # separate the features into subspaces
        features_technical = features[:, :self.labels_encoder.subspace_size_technical]
        features_biological = features[:, self.labels_encoder.subspace_size_technical:]

        return features, features_biological, features_technical

    def forward(self, x, return_features=False):
        features, features_biological, features_technical = self.get_features(x)

        preds, logits, logits_biological, logits_technical = self.labels_encoder.get_logits(
            features_biological=features_biological,
            features_technical=features_technical,
        )

        if return_features:
            return preds, features, features_biological, features_technical
        return preds

    def _step(self, batch, batch_idx, step_name):
        x, y = batch
        x = x[0]
        y = y[0]

        _, y_hat_bio, y_hat_tech = self.get_features(x)
        # encode target into the labels encoder latent space
        _, _, y_bio, y_tech = self.labels_encoder.forward(y, return_features=True)

        loss_bio = ((y_bio - y_hat_bio) ** 2).mean()
        loss_tech = ((y_tech - y_hat_tech) ** 2).mean()
        loss = (loss_bio + loss_tech) / 2

        if step_name == 'train':
            # compute ROC of predicted features
            if (batch_idx % self.train_step_log_interval) == 0:
                with torch.no_grad():
                    preds, _, _, _ = self.labels_encoder.get_logits(
                        features_biological=y_hat_bio,
                        features_technical=y_hat_tech,
                    )

                    self.training_step_preds.append(preds.detach().cpu().numpy())
                    self.training_step_targets.append(y.detach().cpu().numpy())
            self.report_train_auroc()

        self.log(f"{step_name}/loss_bio", loss_bio, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{step_name}/loss_tech", loss_tech, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{step_name}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_name='train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_name='val')


class LPAquaticDilated(_LatentPredictorMixin, AquaticDilated):
    pass
# endregion
