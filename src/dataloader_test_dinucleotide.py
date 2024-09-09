
import torch
import pytorch_lightning as pl
from models.models import SeqModel, OligonucleotideFreqLinear
import gc
import os
import pickle
from util.parse_arguments import parse_arguments
from dataloading.dataloaders import LitCoverageDatasetHDF5
from dataloading.dataloaders import NUM_WORKERS_ENV
from torch import sigmoid
import numpy as np

""" 
This script can be used to debug dataloaders and export the validation set for inspection.

will place output files into debug_outputs/{name}/ where {name} is passed on the command-line

will create files 

    debug_outputs/{name}/e{epoch}_batches.pckl
        - contains the batches and the indices of the samples in the batches
    debug_outputs/{name}/e{epoch}_idx.npz
        - contains just the sample indices
    debug_outputs/{name}/e{epoch}_frq.npz
        - contains the oligonucleotide frequencies
    debug_outputs/{name}/e{epoch}_y.npz
        - contains the labels in multihot representation

these files can be used between different runs to check consistency

"""

class OligonucleotideModel(SeqModel):

    """
    Model that uses just oligonucleotide frequencies,

    when forward is called with return_freq = True, will return predictions and frequencies

    """

    def __init__(self, seq_order, n_classes, debug = True, name=None):
        super().__init__()
        self.loss_fun = torch.nn.BCELoss(reduction='mean')
        self.onflinear = OligonucleotideFreqLinear(seq_order=seq_order, out_features=n_classes, bias = True)
        self.debug = debug
        if debug:
            assert name is not None, 'have to specifiy a name for the debug run if debug == True'
        self.name = name
        self.val_x_frq = []
        self.val_y = []
        self.val_pred = []
        self.val_idx = {}
        self._validation_epoch = 0

    def configure_optimizers(self, lr=0.001, weight_decay=0.):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
        return optimizer

    def forward(self, x, return_freq = False):
        if return_freq:
            y_logit, frq = self.onflinear(x, return_freq)
            return sigmoid(y_logit), frq
        else:
            return sigmoid(self.onflinear(x, return_freq))
    

    def training_step(self, batch, batch_idx):
        
        if self.debug:
            x, y, idx = batch
        else:
            x, y = batch
        x = x[0]
        y = y[0]

        y_hat = self.forward(x)
        loss = self.loss_fun(y_hat, y)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar = True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        if self.debug:
            x, y, idx = batch
        else:
            x, y = batch
        x = x[0]
        y = y[0]

        if self.debug:
            assert len(idx) == y.shape[0]

        y_hat, freq = self.forward(x, return_freq=True)
        loss = self.loss_fun(y_hat, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar = True)


        if self.debug:
            idx = torch.stack(idx).detach().cpu().numpy()
            self.val_idx[batch_idx] = idx
            self.val_x_frq.append(freq.detach().cpu().numpy())
            self.val_y.append(y.detach().cpu().numpy())
            #self.val_pred.append(y_hat.detach().cpu().numpy())

    def on_validation_epoch_end(self):
        if self.debug:

            os.makedirs(f'debug_outputs/{self.name}', exist_ok=True)
            out_batches = os.path.join(f'debug_outputs/{self.name}', f'e{self._validation_epoch}_batches.pckl')
            out_idx = os.path.join(f'debug_outputs/{self.name}', f'e{self._validation_epoch}_idx.npz')
            out_frq = os.path.join(f'debug_outputs/{self.name}', f'e{self._validation_epoch}_frq.npz')
            out_y = os.path.join(f'debug_outputs/{self.name}', f'e{self._validation_epoch}_y.npz')
            #out_pred = os.path.join(f'debug_outputs/{self.name}', f'e{self._validation_epoch}_pred.npz')
            
            #for o in [out_batches, out_idx, out_frq, out_y, out_pred]:
            #    assert not os.path.isfile(o), f'File exists: {o}'

            # save index
            with open(out_batches, 'wb') as outfile:
                pickle.dump(self.val_idx, outfile)
            np.savez_compressed(out_idx, np.concatenate([b for b in self.val_idx.values()], axis = 0))
            del self.val_idx
            self.val_idx = {}
            gc.collect()
            
            # save x nucleotide frequencies
            np.savez_compressed(out_frq, np.concatenate(self.val_x_frq, axis=0))
            del self.val_x_frq[:]
            gc.collect()

            # save y
            np.savez_compressed(out_y, np.concatenate(self.val_y, axis=0))
            del self.val_y[:]
            gc.collect()

            # save y_hat
            #np.savez_compressed(out_pred, np.concatenate(self.val_pred, axis=0))
            #del self.val_pred[:]
            #gc.collect()
        self._validation_epoch += 1



if __name__ == '__main__':
    
    print(f"{NUM_WORKERS_ENV} workers will be tested")

    arguments = [
        ('--dataset', str, 'toy or full'),
        ('--num_workers', str, 'auto or integer'),
        ('--name', str, 'name of the run (used for debugging)')
        ]
    
    flags = []

    args  = parse_arguments(arguments=arguments, flags=flags)

    if args.dataset is None:
        basepath = "data/processed/GRCh38/toydata"
        args.dataset = 'toy'
    elif args.dataset == 'toy':
        basepath = "data/processed/GRCh38/toydata"
    elif args.dataset == 'full':
        basepath = "data/processed/GRCh38/221111_128bp_minoverlap64_mincov2_nc10_tissues"
    else:
        raise ValueError("--dataset has to be either 'toydata' or 'full'")
    
    if args.num_workers is None:
        args.num_workers = NUM_WORKERS_ENV
    elif args.num_workers == 'auto':
        args.num_workers = NUM_WORKERS_ENV
    else:
        args.num_workers = int(args.num_workers)
        assert args.num_workers >= 0

    print(f"initializing dataset with {args.num_workers} workers and {args.dataset} data")

    datamodule = LitCoverageDatasetHDF5(
        seq_order = 2,
        seq_len = 128,
        batch_size = 4096,
        random_reverse_complement=True, # note: this should not affect the validation loader
        random_shift=8, # note: this should not affect the validation loader
        basepath = basepath,
        num_workers=args.num_workers,
        debug = True # make the dataloader return the sample index
    )

    model = OligonucleotideModel(seq_order=2,n_classes=datamodule.n_classes,name=args.name)

    if torch.cuda.is_available():
        trainer_args = {
            'accelerator': 'gpu',
            'devices': 1,
        }
    else:
        trainer_args = {
            'accelerator': 'cpu',
            'devices': 'auto',
        }
    trainer_args['reload_dataloaders_every_n_epochs'] = 1
    trainer_args['max_epochs'] = 3
    trainer_args['limit_train_batches'] = 10 # not interested in this
    trainer_args['enable_progress_bar'] = True

    trainer = pl.Trainer(**trainer_args)

    trainer.fit(model, datamodule)






