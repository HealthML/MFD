
import torch
import pytorch_lightning as pl
from models import models
from models.models import IEAquaticDilated, OligonucleotideModel
import gc
import os
from torch import tensor
import pickle
from util.parse_arguments import parse_arguments
from dataloading.dataloaders import LitCoverageDatasetHDF5
from dataloading.dataloaders import NUM_WORKERS_ENV
from util.load_config import config
from torch import sigmoid
import numpy as np
import pandas as pd
from torchmetrics.classification import MultilabelAveragePrecision, MultilabelAUROC
import h5py

from sklearn.metrics import roc_auc_score, average_precision_score

torch.manual_seed(1)

""" 

TODO: documentation

"""

METRIC_AGG_BATCHES = 100 # this value should be as large as possible
BATCH_SIZE = 4096


def roc_auc_np(y_true, y_score):

    if y_true.ptp() == 0.:
        return np.nan
    else:
        return roc_auc_score(y_true = y_true, y_score = y_score)
    
def pr_auc_np(y_true, y_score):

    if y_true.ptp() == 0.:
        return np.nan
    else:
        return average_precision_score(y_true = y_true, y_score = y_score)


class ReverseComplementWrapper(pl.LightningModule):
    """
    Symmetric cropping along position

        crop: number of positions to crop on either side
    """

    def __init__(self, model, rc_fn, use_torchmetrics = False):
        super().__init__()
        self.model = model
        self.loss_fn = torch.nn.BCELoss(reduction='mean')
        self.metric_agg_batches = METRIC_AGG_BATCHES
        self.roc = MultilabelAUROC(model.hparams.n_classes, average = None, compute_on_cpu = True)
        self.pr = MultilabelAveragePrecision(model.hparams.n_classes, average = None, compute_on_cpu = True)
        self.roc_values = []
        self.pr_values = []
        self.y = []
        self.y_hat = []
        self.batches = 0 # counts number of batches that are currently saved
        self.rc_fn = rc_fn
        self.use_torchmetrics = use_torchmetrics        
    
    def stack_mean(self, x):
        return torch.mean(torch.stack(x), dim=0)

    def forward(self, x):
                
        forward = x      
        reverse_complement = self.rc_fn(forward)

        predictions_l = []

        for s in [forward, reverse_complement]:
            predictions = self.model.forward(s)
            predictions_l.append(predictions)

        output = self.stack_mean(predictions_l)
        return output
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        
        x, y, idx = batch
        
        x = x[0]
        y = y[0]
        
        if x.sum(dim=[1, 2], keepdims=True).min() < 1:
            for i, v in enumerate(x.sum(dim=[1, 2])):
                if v < 1:
                    print(idx[i])
                    raise RuntimeError(f'encountered all zero seq with index {int(idx[i])}')

        y_hat = self.forward(x)
                
        loss = self.loss_fn(y_hat, y)

        self.log('loss', loss)
        
        y = y.int()
        
        self.batches += 1

        if self.use_torchmetrics:

            # use torchmetrics
            # this works, but is somewhat slow and not well configurable

            self.roc(y_hat.detach(), y)
            self.pr(y_hat.detach(), y)
        
            if (self.batches % self.metric_agg_batches) == 0:
                self.roc_values.append(self.roc.compute().cpu().numpy())
                self.pr_values.append(self.pr.compute().cpu().numpy())
                self.roc.reset()
                self.pr.reset()
                self.batches = 0

        else:

            # use sklearn metrics

            self.y.append(y.detach().cpu().numpy().astype(int))
            self.y_hat.append(y_hat.detach().cpu().numpy().astype(np.float32))

            if (self.batches % self.metric_agg_batches) == 0:

                y_true = np.concatenate(self.y)
                y_score = np.concatenate(self.y_hat)

                roc_auc = np.array([roc_auc_np(y_true=y_true[:,i],y_score=y_score[:,i]) for i in range(y_true.shape[1])])
                pr_auc = np.array([pr_auc_np(y_true=y_true[:,i],y_score=y_score[:,i]) for i in range(y_true.shape[1])])

                self.roc_values.append(roc_auc)
                self.pr_values.append(pr_auc)

                del self.y[:]
                del self.y_hat[:]

                self.batches = 0


    def on_test_epoch_end(self):

        if self.batches > 0:

            if self.use_torchmetrics:
                self.roc_values.append(self.roc.compute().cpu().numpy())
                self.pr_values.append(self.pr.compute().cpu().numpy())
                self.roc.reset()
                self.pr.reset()
            else:
                y_true = np.concatenate(self.y)
                y_score = np.concatenate(self.y_hat)
                roc_auc = np.array([roc_auc_np(y_true=y_true[:,i],y_score=y_score[:,i]) for i in range(y_true.shape[1])])
                pr_auc = np.array([pr_auc_np(y_true=y_true[:,i],y_score=y_score[:,i]) for i in range(y_true.shape[1])])
                self.roc_values.append(roc_auc)
                self.pr_values.append(pr_auc)
                del self.y[:]
                del self.y_hat[:]

            # we don't clear the number of batches here


def get_args():

    #TODO: have defaults be handled with the argparse package directly instead of all the manual stuff below...

    arguments = [
        ('--dataset', str, 'toy or full'),
        ('--num_workers', str, 'auto or integer'),
        ('--genome', str, 'GRCh38 or mm10'),
        ('--model_cls', str, 'model class'),
        ('--checkpoint', str, 'checkpoint path'),
        ('--set', str, 'val or test')
        #('--save_predictions', str, 'path to a file in which predictions should be saved (default: dont save predictions)')
        ]
    
    flags = []

    args  = parse_arguments(arguments=arguments, flags=flags)

    if args.genome is None:
        args.genome = 'GRCh38'
    else:
        assert args.genome in ['GRCh38','mm10']
    
    if args.dataset is None:
        basepath = f"data/processed/{args.genome}/toydata"
        args.dataset = 'toy'
    elif args.dataset == 'toy':
        basepath = f"data/processed/{args.genome}/toydata"
    elif args.dataset == 'full':
        basepath = f"data/processed/{args.genome}/221111_128bp_minoverlap64_mincov2_nc10_tissues"
    else:
        raise ValueError("--dataset has to be either 'toydata' or 'full'")
    args.basepath = basepath
    
    if args.num_workers is None:
        args.num_workers = NUM_WORKERS_ENV
    elif args.num_workers == 'auto':
        args.num_workers = NUM_WORKERS_ENV
    else:
        args.num_workers = int(args.num_workers)
        assert args.num_workers >= 0

    if args.model_cls is None:
        args.model_cls = 'IEAquaticDilated'

    if args.set is None:
        args.set = 'val'
    else:
        assert args.set in ['val','test','train']

    assert args.checkpoint is not None

    return args

if __name__ == '__main__':
    
    print(f"{NUM_WORKERS_ENV} workers will be used")

    args = get_args()

    model_cls = getattr(models, args.model_cls)
    try:
        model = model_cls.load_from_checkpoint(args.checkpoint)
    except RuntimeError as e:
        model = model_cls.load_from_checkpoint(args.checkpoint, map_location=torch.device('cpu'))

    print(f"initializing dataset with {args.num_workers} workers and {args.genome} {args.dataset} data")

    default_seq_len = 128
    default_seq_order = 2
    if 'seq_len' not in model.hparams:
        print(f'Model has no seq_len attribute, assuming {default_seq_len}')
    if 'seq_order' not in model.hparams:
        print(f'Model has no seq_order attribute, assuming {default_seq_order}')

    datamodule = LitCoverageDatasetHDF5(
        seq_order = model.hparams.get('seq_order', default_seq_order),
        seq_len = model.hparams.get('seq_len', default_seq_len),
        batch_size = BATCH_SIZE,
        random_reverse_complement=False,
        random_shift=0,
        basepath = args.basepath,
        num_workers=args.num_workers,
        debug = True
    )

    rc_model = ReverseComplementWrapper(model, rc_fn=datamodule.tokenizer.onehot_reverse_complement_func(use_numpy=False))

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

    trainer = pl.Trainer(reload_dataloaders_every_n_epochs = 1,
                         enable_progress_bar = True,
                         deterministic=True,
                         **trainer_args)


    if args.set in ['val','train']:
        datamodule.setup('fit')
        datamodule.data.augment_off()
        if args.set == 'val':
            i = datamodule.i_val
        else:
            i = datamodule.i_train
    else:
        datamodule.setup('test')
        i  = datamodule.i_test

    sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.SubsetRandomSampler(i),
        batch_size = BATCH_SIZE,
        drop_last = False
    )
    dataloader = torch.utils.data.DataLoader(
        datamodule.data,
        sampler = sampler,
        num_workers = args.num_workers,
        worker_init_fn = datamodule.worker_init_fn # this is necessary to prevent scrambled data due to many workers using the same fasta file
    )

    labels = datamodule.data.labelloader.get_label_ids()

    trainer.test(rc_model, dataloader)

    roc_values = np.stack(rc_model.roc_values)
    pr_values = np.stack(rc_model.pr_values)

    weights = np.ones(roc_values.shape[0])
    if rc_model.batches > 0:
        last_chunk_weight = rc_model.batches / METRIC_AGG_BATCHES # reduce the weight of the last chunk
        weights[-1] = last_chunk_weight

    def nan_average(x):
        nan_idx = np.isnan(x)
        if nan_idx.all():
            return np.nan
        x = np.where(nan_idx, 0., x) # replace nans with 0 to avoid errors
        w = np.where(nan_idx, 0., weights) # assign 0 weight to nans
        return np.average(x, weights = w)

    roc = np.apply_along_axis(nan_average,axis=0,arr=roc_values)
    pr = np.apply_along_axis(nan_average,axis=0,arr=pr_values)

    roc_avg = np.nanmean(roc)
    roc_sd = np.nanstd(roc)
    pr_avg = np.nanmean(pr)
    pr_sd = np.nanstd(pr)

    print(f'average ROC AUC: {roc_avg} ({roc_sd} sd)')
    print(f'average PRC AUC: {pr_avg} ({pr_sd} sd)')

    result_df = pd.DataFrame({'roc':roc, 'pr':pr}, index=labels)

    out_file = args.checkpoint.replace('.ckpt','') + f'.{args.set}.roc_pr.tsv.gz'

    result_df.to_csv(out_file, sep='\t', index_label='File.accession')























