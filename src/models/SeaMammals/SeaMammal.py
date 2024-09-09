import pytorch_lightning
import torch
from torch import nn
import numpy as np
from typing import Optional
from torchmetrics.classification import MultilabelAccuracy

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


class SeaMammal(pytorch_lightning.LightningModule):
    model_dict = None

    def __init__(
            self,
            input_sizes: dict = {"seq_len": 128, "n_channels": 16},
            output_sizes: dict = {"n_classes": 2106},
            lr=1e-3,
            log_hyperparams=False,  # why set this to False?
            init_bias=None,
            class_freq=None,
            **kwargs
    ):
        super().__init__()
        self.input_sizes = input_sizes
        self.output_sizes = output_sizes
        self.save_hyperparameters(logger=log_hyperparams)

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
        self.accuracy = MultilabelAccuracy(num_labels= self.output_sizes["n_classes"], average='macro')

        # build model
        self.model = self._build_model_from_dict(self.model_dict)

    def _linear_input_cnn_pytorch_layer(self, input_sizes_, cnn_list):
        pytorch_layers = []
        print_list = []
        current_size = {"len": input_sizes_["seq_len"], "n_channels": input_sizes_["n_channels"]}
        print_list.append("Current size: {}".format(current_size))
        for cnn_block in cnn_list:

            # add conv layer ----------------------------------------------------------------
            # check if model_dict is correctly defined, add default params
            assert "out_channels" in cnn_block["conv"].keys(), "Conv layer must have 'out_channels' key"
            assert "kernel_size" in cnn_block["conv"].keys(), "Conv layer must have 'kernel_size' key"
            if "stride" not in cnn_block["conv"].keys():
                cnn_block["conv"]["stride"] = 1
            if "padding" not in cnn_block["conv"].keys():
                cnn_block["conv"]["padding"] = 0
            print_list.append("Conv layer: {}".format(cnn_block["conv"]))
            
            # create, add conv layer
            pytorch_layers.append(nn.Conv1d(
                in_channels=current_size["n_channels"], 
                out_channels=cnn_block["conv"]["out_channels"], 
                kernel_size=cnn_block["conv"]["kernel_size"], 
                stride=cnn_block["conv"]["stride"], 
                padding=cnn_block["conv"]["padding"]
                )
            )
            print_list.append("Conv layer: {}".format(cnn_block["conv"]))

            # update current_size
            
            current_size = {
                "len": int((current_size["len"] - (cnn_block["conv"]["kernel_size"]-1) + 2 * cnn_block["conv"]["padding"] -1 ) / cnn_block["conv"]["stride"] + 1),
                "n_channels": cnn_block["conv"]["out_channels"]
            } 
            print_list.append("Current size: {}".format(current_size))

            # add activation function --------------------------------------------------------
            print_list.append("ReLU")
            pytorch_layers.append(nn.ReLU())

            # add dropout -------------------------------------------------------------------
            if "dropout" in cnn_block.keys() and cnn_block["dropout"] > 0.0:
                pytorch_layers.append(nn.Dropout(p=cnn_block["dropout"]))
                print_list.append("Dropout: {}".format(cnn_block["dropout"]))

            # add max pooling ---------------------------------------------------------------
            if "max" in cnn_block.keys() and cnn_block["max"]["kernel_size"] > 0:
                
                # check if model_dict is correctly defined, add default params
                assert "kernel_size" in cnn_block["max"].keys(), "Max layer must have 'kernel_size' key"
                if "stride" not in cnn_block["max"].keys():
                    cnn_block["max"]["stride"] = 1
                
                # create, add max layer
                pytorch_layers.append(nn.MaxPool1d(
                    kernel_size=cnn_block["max"]["kernel_size"], 
                    stride=cnn_block["max"]["stride"], 
                    padding=cnn_block["max"]["padding"]
                    )
                )
                print_list.append("Max layer: {}".format(cnn_block["max"]))

                # update current_size
                current_size = {
                    "len": int((current_size["len"] - (cnn_block["max"]["kernel_size"]-1) + 2 * cnn_block["max"]["padding"] -1 ) / cnn_block["max"]["stride"] + 1),
                    "n_channels": current_size["n_channels"]
                }
                print_list.append("Current size: {}".format(current_size))

        for line in print_list:
            print(line)

        return current_size["len"] * current_size["n_channels"], pytorch_layers   
    
    def _linear_pytorch_layer(self, output_sizes, linear_list, linear_input_size):
        pytorch_layers = []
        print_list = []
        print_list.append("Linear input size: {}".format(linear_input_size))
        current_input_size = linear_input_size
        linear_list.append(output_sizes["n_classes"])
        for layer_size in linear_list:
            pytorch_layers.append(nn.Linear(in_features=current_input_size, out_features=layer_size))
            pytorch_layers.append(nn.ReLU())
            print_list.append("Linear layer: {}".format(layer_size))
            print_list.append("ReLU")
            current_input_size = layer_size
            print_list.append("Current size: {}".format(current_input_size))
        # remove last ReLU
        pytorch_layers.pop()
        print_list.pop(-2)

        for line in print_list:
            print(line)

        return pytorch_layers

    
    def _build_model_from_dict(self, model_dict):
        """
            model_dict = {
                "cnn": [
                    {"conv":{"out_channels": 64, "stride": 1, "kernel_size": 3, "padding": 0},  "dropout": 0.0, "max": {"kernel_size": 4, "stride": 1, "padding": 0}},
                    {"conv":{"out_channels": 64, "stride": 1, "kernel_size": 3, "padding": 0},  "dropout": 0.0, "max": {"kernel_size": 4, "stride": 1, "padding": 0}},
                ],
                "add_linears": [4000]
            }
        """
        # check model_dict
        assert isinstance(model_dict, dict)
        assert "cnn" in model_dict.keys() and "add_linears" in model_dict.keys()
        assert isinstance(model_dict["cnn"], list)
        assert isinstance(model_dict["add_linears"], list)

        for layer_dict in model_dict["cnn"]:
            assert isinstance(layer_dict, dict)
            assert 1 <= len(layer_dict.keys()) <= 3
            assert "conv" in layer_dict.keys()
        
        linear_input_size, cnn_pytorch_layer = self._linear_input_cnn_pytorch_layer(input_sizes_=self.input_sizes, cnn_list=model_dict["cnn"])
        return nn.Sequential(
            nn.Sequential(
                *cnn_pytorch_layer,
            ),
            nn.Sequential(
                Lambda(lambda x: x.view(x.size(0), -1)),
                *self._linear_pytorch_layer(output_sizes=self.output_sizes, linear_list=model_dict["add_linears"], linear_input_size=linear_input_size),
            ),
            nn.Sigmoid(),
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x[0]  # TODO: why does the loaded data have an additional dimension in front (?)
        y = y[0]
        y_hat = self(x)
        loss = self.loss_fun(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        accuracy = self.accuracy(y_hat, y)
        self.log("train_accuracy", accuracy, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x[0]  # TODO: why does the loaded data have an additional dimension in front (?)
        y = y[0]
        y_hat = self(x)
        loss = self.loss_fun(y_hat, y)
        self.log("val_loss", loss, on_step=False, prog_bar=True, on_epoch=True, logger=True)
        accuracy = self.accuracy(y_hat, y)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, logger=True)

    def configure_optimizers(self, lr: Optional[float] = None):

        if lr is None:
            lr = float(self.hparams.lr)

        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

        return optimizer
    
    def forward(self, x):
        return self.model(x)

    def train_dataloader(self):
        raise NotImplementedError()

    def val_dataloader(self):
        raise NotImplementedError()

    def test_dataloader(self):
        raise NotImplementedError()

    def test_step(self, batch, batch_idx, *args, **kwargs):
        raise NotImplementedError()



