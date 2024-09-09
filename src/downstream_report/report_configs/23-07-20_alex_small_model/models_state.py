import pytorch_lightning
import torch
import wandb
import yaml
from torch import nn
import numpy as np
from typing import Optional
from torchmetrics.classification import MultilabelAccuracy, MultilabelAUROC
import torch.nn.functional as F
import csv
import datetime
import pandas as pd
import os

import util.metrics as metrics
import dataloading.metadata as data_metadata


class SeqModel(pytorch_lightning.LightningModule):

    def __init__(
            self,
            lr: float = 1e-3,
            seq_len: int = 128,
            n_classes: int = 1,
            class_freq=None,
            metadata_loader=None,
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
        self.trainer.datamodule.data.augment_on()

    def on_validation_model_eval(self):
        self.trainer.datamodule.data.augment_off()


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
        x = x[0]  # TODO: why does the loaded data have an additional dimension in front (?)
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
                 *args,
                 **kwargs
                 ):

        super().__init__(*args, **kwargs)

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
            nn.Conv1d(in_channels=in_channels, out_channels=C // 2, kernel_size=15, padding='same'),
            ConvBlock(in_channels=C // 2, out_channels=C // 2, kernel_size=1, activation=activation, residual=True),
            nn.MaxPool1d(2, 2)
        )

        self.tower = Basenji2ResConvTower(C // 2, C, kernel_size=tower_kernel_size, L=L1, activation=activation)
        self.dilated_residual_block = Basenji2DilatedResidualBlock(C, kernel_size=dilated_residual_kernel_size,
                                                                   dropout=dilated_residual_dropout, D=D,
                                                                   channel_multiplier=1., L=L2, activation=activation)
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
                self.conv1d_block.append(nn.GELU())
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


# region Disentanglement
class MetadataColumnEmbedding(nn.Module):
    def __init__(
            self,
            var_def,
            var_vals,
    ):
        super().__init__()

        self.var_type = var_def['type']
        if self.var_type == 'category':
            var_vals = np.stack(data_metadata.categorize_multilabel_column(var_vals))
        elif self.var_type in ('float', 'int'):
            var_vals = np.stack(var_vals.astype(float)).reshape((len(var_vals), 1))
            var_vals = (var_vals - var_vals.mean()) / var_vals.std()
        elif self.var_type == 'date.year':
            var_vals = np.stack(data_metadata.categorize_multilabel_column(pd.Series(pd.DatetimeIndex(var_vals).year)))
        else:
            raise ValueError(f'Unknown variable type: {self.var_type}')
        self.dim_input = var_vals.shape[1]

        self.intermediate = var_def['intermediate']
        if self.intermediate == 'embed':
            self.dim_intermediate = var_def['intermediate_dim']
            embedding = nn.Sequential(
                # for now hardcode the embedding model as a NN with a single hidden layer of dim = 2*embedd dim
                nn.Linear(in_features=self.dim_input, out_features=self.dim_intermediate * 2),
                nn.ReLU(),
                nn.Linear(in_features=self.dim_intermediate * 2, out_features=self.dim_intermediate * 2),
                nn.ReLU(),
                nn.Linear(in_features=self.dim_intermediate * 2, out_features=self.dim_intermediate)
            )
        elif self.intermediate == 'multihot':
            self.dim_intermediate = self.dim_input
            embedding = nn.Linear(in_features=self.dim_input, out_features=self.dim_intermediate)
        elif self.intermediate is None:
            self.dim_intermediate = self.dim_input
            embedding = nn.Identity()
        else:
            raise ValueError(f'Unknown embedding intermediate type: {self.intermediate}')

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
        self.mappings_direct = nn.ModuleDict({
            var_name: nn.Linear(
                in_features=self.column_embeddings[var_name].dim_intermediate,
                out_features=self.column_embeddings[var_name].dim_intermediate,
            ) for var_name in self.variables_direct
        })
        self.mapping_interact = nn.Sequential(
            nn.Linear(in_features=self.dim_input_interact, out_features=self.dim_embedding_interact * 2),
            nn.ReLU(),
            nn.Linear(in_features=self.dim_embedding_interact * 2, out_features=self.dim_embedding_interact * 2),
            nn.ReLU(),
            nn.Linear(in_features=self.dim_embedding_interact * 2, out_features=self.dim_features_interact)
        )

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
        return sum(mapping.out_features for mapping in self.mappings_direct.values())

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


class MetadataEmbedding(nn.Module):
    def __init__(
            self,
            metadata_loader,
            metadata_mapping_config,
            embedding_size_biological,
            embedding_size_technical,
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
        )
        self.embedding_technical = MetadataGroupEmbedding(
            mapping_config=self.mapping_config,
            group_definitions=self.mapping_config['technical_features'],
            df_metadata=self.metadata_loader.df_raw,
            embedding_size=embedding_size_technical,
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


class _IndependentEmbeddingsMixin:
    """
    This class assumes the original model class defines a 'model' attribute, where the ouput of model[:-2]
    (i.e., of the layer third to last) constitutes the latent features.

    biological_subspace_ratio - defines the ratio of the sizes of the biological and technical embeddings, e.g.,
    with a latent size of 10, and the biological_subspace_ratio=0.8, the biological subspace will have a dimensionality
    of 8, and the technical one of 2.
    If None is passed, the model will instead determine the dimensionalities by itself, by learning mappings of the
    features into two orthogonal spaces.
    """

    def __init__(
            self,
            num_hidden_embedding=0,
            dim_hidden_embedding=256,
            coeff_cov=1,
            biological_subspace_ratio=.5,
            use_intermediate_embedding=False,
            metadata_mapping_config=None,
            num_batches_for_variance_estimation=1000,
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

        assert self.hparams.metadata_loader is not None

        self.latent_size = self.model[-2].in_channels
        assert self.latent_size % 2 == 0
        self.model = self.model[:-2]

        self.metadata_embedding = MetadataEmbedding(
            metadata_loader=self.hparams.metadata_loader,
            embedding_size_biological=self.subspace_size_biological,
            embedding_size_technical=self.subspace_size_technical,
            metadata_mapping_config=metadata_mapping_config,
        )
        self.bias = nn.Parameter(data=torch.zeros((self.hparams.n_classes,)).float())
        self.bn_features = nn.BatchNorm1d(num_features=self.latent_size)
        if self.hparams.biological_subspace_ratio is None:
            self.mapping_biological, self.mapping_technical = (nn.Sequential(
                nn.Linear(in_features=self.latent_size, out_features=self.latent_size),
                nn.BatchNorm1d(num_features=self.latent_size),
            ) for _ in range(2))

        # log norm_ratio
        self.validation_step_preds_biological = []
        self.validation_step_preds_technical = []
        self.variance_ratio_log = []
        now = datetime.datetime.now()
        date_string = now.strftime("%Y-%m-%d_%H-%M")
        base_path = f"norm_ratio_log/{date_string}_"
        if wandb.run is not None:
            base_path = base_path + str(wandb.run.id) + '_'
        self.variance_ratio_file_path = base_path + "variances_ratio_log.csv"
        self.variances_file_path = base_path + "variances_log.csv"
        self.variance_ratio_group_file_path = base_path + "variances_ratio_group_log.csv"

    @property
    def subspace_size_biological(self):
        if self.hparams.biological_subspace_ratio is None:
            return self.latent_size
        return int(self.latent_size * self.hparams.biological_subspace_ratio)

    @property
    def subspace_size_technical(self):
        if self.hparams.biological_subspace_ratio is None:
            return self.latent_size
        return self.latent_size - self.subspace_size_biological

    def get_features(self, x):
        features = self.model(x)
        # batch_norm is useful here to constrain the features to a variance of 1
        # this makes it easier to penalize the covariance matrix
        # (otherwise the model can just scale the features to an arbitrarily low magnitude)
        features = self.bn_features(features)
        features = features.reshape(features.shape[0], self.latent_size)

        if self.hparams.biological_subspace_ratio is None:
            # map the features to new spaces via learned mappings
            features_biological = self.mapping_biological(features)
            features_technical = self.mapping_technical(features)
        else:
            # separate the features into subspaces
            features_technical = features[:, :self.subspace_size_biological]
            features_biological = features[:, self.subspace_size_biological:]

        return features, features_biological, features_technical

    def forward(self, x, compute_covariance=False, return_features=False, return_logits=False):
        features, features_biological, features_technical = self.get_features(x)

        logits_biological = self.metadata_embedding.features_to_logits_biological(features_biological)
        logits_technical = self.metadata_embedding.features_to_logits_technical(features_technical)
        logits = logits_biological + logits_technical + self.bias
        preds = F.sigmoid(logits)

        # TODO refactor this into returning preds + a dict of all possible outputs, instead of having multiple if/else
        # paths
        if compute_covariance:
            # happens during model fitting
            if self.hparams.biological_subspace_ratio is None:
                # enforce orthogonality of the biological and technical mappings
                cov = self.mapping_biological[0].weight @ self.mapping_technical[0].weight.T
            else:
                # penalize the cross-covariance matrix between the two subspaces
                cov = features_technical.T @ features_biological
            norm = torch.linalg.matrix_norm(cov)
            if return_logits:
                return preds, norm, cov, F.sigmoid(logits_biological), F.sigmoid(logits_technical)
            return preds, norm, cov
        if return_features:
            return preds, features, features_biological, features_technical
        return preds

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x[0]  # TODO: why does the loaded data have an additional dimension in front (?)
        y = y[0]
        y_hat, norm, cov = self.forward(x, compute_covariance=True)
        loss_pred = self.loss_fun(y_hat.squeeze(dim=-1), y)
        loss = loss_pred + self.hparams.coeff_cov * norm
        accuracy = self.accuracy(y_hat.squeeze(dim=-1), y)

        self.training_step_preds.append(y_hat.detach().cpu().numpy())
        self.training_step_targets.append(y.detach().cpu().numpy())

        if len(self.training_step_preds) == 100:
            with torch.no_grad():
                all_preds = np.concatenate(self.training_step_preds)
                all_targets = np.concatenate(self.training_step_targets)

                auroc = metrics.multilabel_auroc_np(targets=all_targets, preds=all_preds)
                self.log("train/auroc", auroc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

                # apparently that's the memory-safe way to empty arrays
                del self.training_step_preds[:]
                del self.training_step_targets[:]

        self.log("train/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/loss_pred", loss_pred, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/loss_cov", norm, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x[0]  # TODO: why does the loaded data have an additional dimension in front (?)
        y = y[0]
        y_hat, norm, cov, preds_bio, preds_tech = self.forward(x, compute_covariance=True, return_logits=True)
        loss_pred = self.loss_fun(y_hat.squeeze(dim=-1), y)
        loss = loss_pred + self.hparams.coeff_cov * norm
        accuracy = self.accuracy(y_hat.squeeze(dim=-1), y)

        if len(self.validation_step_preds_biological) < self.hparams.num_batches_for_variance_estimation:
            self.validation_step_preds_biological.append(preds_bio.detach().cpu())
            self.validation_step_preds_technical.append(preds_tech.detach().cpu())
        self.validation_step_preds.append(y_hat.detach().cpu())
        self.validation_step_targets.append(y.detach().cpu())

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
        variance_values = list(variance_values)

        # add experiment identifiers and groups to variance_ratio_log initially
        if not self.variance_ratio_log:
            md = self.metadata_embedding.metadata_loader.df_raw
            n_id_groups = []
            try:
                for i in range(len(md)):
                    if md['proc_Assay_lvl1'][i] == 'ATAC-seq':
                        n_id_groups.append([i, md['File_accession'][i], 'ATAC-seq'])
                    elif md['proc_Assay_lvl1'][i] == 'DNase-seq':
                        n_id_groups.append([i, md['File_accession'][i], 'DNase-seq'])
                    else:
                        n_id_groups.append([i, md['File_accession'][i],
                                            md['proc_Assay_lvl1'][i] + ' ' + md['proc_target'][i]])
            except Exception as e:
                print(
                    "!!!!! The problem is probably with differnt Metadata column naming: Some use dots, some underscores.")
                print("Your column names: ", md.columns)
                print(
                    "Change your colum names to: Index(['File.accession', 'Biosample.term_name', 'Biosample.organ_slims','Biosample.system_slims', 'Biosample.developmental_slims', 'Biosample.organism', 'proc_Assay.lvl1', 'proc_target', 'Experiment.date.released', 'Library.lab', 'spot2_score', 'five_percent_narrowpeaks_count', 'frip', 'reproducible_peaks', 'proc_age.bin', 'proc_age.bin.units', 'proc_Biosample.life_stage'], dtype='object')")
                raise e

            self.variance_ratio_log = [
                ["n_experiment", "experiment_id", "experiment_group", f"epoch_-1_variance_ratio_bio/tech"]]
            self.variance_ratio_log += [n_id_group + [variance_value] for n_id_group, variance_value in
                                        zip(n_id_groups, variance_values)]

        else:
            new_variance_vector = [f"epoch_{self.current_epoch}_variance_ratio_bio/tech"] + variance_values
            self.variance_ratio_log = [variance_ratio_row + [new_col_value] for variance_ratio_row, new_col_value in
                                       zip(self.variance_ratio_log, new_variance_vector)]

        rows = []
        md = self.metadata_embedding.metadata_loader.df_raw
        for i in range(len(md)):
            rows.append({
                'Experiment': md['File_accession'][i],
                'Variance Bio': variances_biological[i],
                'Variance Tech': variances_technical[i],
                'Epoch': self.current_epoch,
                'Model Class IDX': i,
            })
        pd.DataFrame(rows).to_csv(self.variances_file_path, mode='a', index=False)

        # variance_ratio_log exists?
        variance_ratio_log_folder = "variance_ratio_log"
        if not os.path.exists(variance_ratio_log_folder):
            os.makedirs(variance_ratio_log_folder)

        # all data
        with open(self.variance_ratio_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.variance_ratio_log)

        # grouped
        columns = self.variance_ratio_log[0]
        df = pd.DataFrame(self.variance_ratio_log[1:], columns=columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        grouped = df.groupby('experiment_group')[numeric_cols].mean()
        del grouped['n_experiment']
        grouped = grouped.reset_index()
        variance_ratio_log_grouped = [grouped.columns.tolist()] + grouped.values.tolist()

        with open(self.variance_ratio_group_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(variance_ratio_log_grouped)

        # charts on wandb for every group
        for group in variance_ratio_log_grouped[1:]:
            self.log(f'mean_variance_ratio_{group[0]}', group[-1], on_epoch=True, prog_bar=True, logger=True)

        if wandb.run is not None:
            table = wandb.Table(
                data=self.variance_ratio_log[1:],
                columns=self.variance_ratio_log[0],
            )
            wandb.log({f'variance_ratio': table})

            # group
            table_grouped = wandb.Table(
                data=variance_ratio_log_grouped[1:],
                columns=variance_ratio_log_grouped[0],
            )
            wandb.log({f'variance_ratio_grouped': table_grouped})

    def on_validation_epoch_end(self):
        with torch.no_grad():
            all_preds = np.concatenate(self.validation_step_preds)
            all_targets = np.concatenate(self.validation_step_targets)
            auroc = metrics.multilabel_auroc_np(targets=all_targets, preds=all_preds)

            self.log("val/auroc", auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

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


class IEAquaticDilated(_IndependentEmbeddingsMixin, AquaticDilated):
    pass
# endregion