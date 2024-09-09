import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.profilers import PyTorchProfiler
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.tuner import Tuner

import dataloading.metadata
from models import models
from util.load_config import config
from util.parse_arguments import parse_arguments
from util.wandb import create_sweep, current_sweep_id, read_config
import os

import datetime

from dataloading.dataloaders import LitCoverageDatasetHDF5

import gc

torch.cuda.empty_cache()
gc.collect()

import inspect

def filter_args(target, args):
    # Get the list of argument names that the target accepts.
    valid_args = inspect.signature(target).parameters.keys()
    # Keep only entries with keys that the target accepts.
    return {k: v for k, v in args.items() if k in valid_args}

# RUN ---------------------------------------------------------------------------

def single_model_training(config_file=None):
    if config_file is None:
        run = wandb.init(mode=mode)
        wandb_args = wandb.config

        log_id = run.id

        print('passing wandb_args to datamodule:', wandb_args)

        wandb_logger = WandbLogger(
            project= config['wandb']['project'],
            entity = config['wandb'].get('entity', None)
        )
        config_args = wandb_args

    else:

        args = read_config(config_file)
        args = args['parameters']
        config_args = {}
        for k, v in args.items():
            config_args[k] = v['values'][0]
        wandb_logger = None
        
        prefix = '.'.join(os.path.basename(config_file).split('.')[:-1])
        date_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_id = f'{date_time}_{prefix}'

        print('=> passing config_args to datamodule:', config_args)

    if 'seed_everything' in config_args:
        try:
            from pytorch_lightning.utilities.seed import seed_everything
            seed_everything(config_args['seed_everything'], workers=True)
        except ImportError:
            print('Could not import seed everything')

    if debug_active:
        if no_wandb:
            def worker_init_fn(worker_id):
                worker_info = torch.utils.data.get_worker_info()
                try:
                    worker_info.dataset.fasta_worker_init()
                except AttributeError:
                    worker_info.dataset.dataset.fasta_worker_init()
                print(f'worker {worker_info.id} initialized (num_workers: {worker_info.num_workers})')
            config_args['worker_init_fn'] = worker_init_fn

    if 'random_shift' not in config_args:
        config_args['random_shift'] = 3

    datamodule = LitCoverageDatasetHDF5(
            random_reverse_complement = True,
            **config_args
        )
    

    model_cls = config_args.get('model_class', 'OligonucleotideModel') # get the model class or the default (IEAquaticDilated)
    model_cls = getattr(models, model_cls)

    model_args = filter_args(model_cls.__init__, config_args)

    model = model_cls(n_classes=datamodule.n_classes, **model_args)

    if torch.cuda.is_available():
        trainer_args = {
            'max_epochs': config_args['max_epochs'],
            'accelerator': 'gpu',
            'devices': 1,
            'reload_dataloaders_every_n_epochs': 1
        }
    else:
        trainer_args = {
            'max_epochs': config_args['max_epochs'],
            'accelerator': 'cpu',
            'devices': 'auto',
            'reload_dataloaders_every_n_epochs': 1
        }
    if debug_active:
        trainer_args['limit_train_batches'] = 10
        trainer_args['limit_val_batches'] = 10
        if no_wandb & (not tune_batch_size):
            os.makedirs('profiles', exist_ok=True)
            trainer_args['profiler'] = PyTorchProfiler(dirpath='profiles')

        print(f'will save model weights to checkpoints/{log_id}')
        trainer_args['callbacks'] = [TQDMProgressBar(refresh_rate=1),
                                     ModelCheckpoint(monitor='val/loss', save_top_k=3, mode='min', dirpath=f'checkpoints/{log_id}')
                                     ]
    else:
        print(f'will save model weights to checkpoints/{log_id}')
        trainer_args['callbacks'] = [TQDMProgressBar(refresh_rate=50), # greatly reduce the size of log files
                                     ModelCheckpoint(monitor='val/loss', save_top_k=3, mode='min', dirpath=f'checkpoints/{log_id}')
                                     ]

    if config_file is None:
        trainer_args['logger'] = wandb_logger
    
    trainer = Trainer(**trainer_args)

    if tune_batch_size:
        tuner = Tuner(trainer)
        tuner.scale_batch_size(model, datamodule=datamodule, mode='power')

    trainer.fit(model, datamodule)


# MAIN ---------------------------------------------------------------------------

if __name__ == '__main__':
    # parse arguments ----------------------------------------------------------
    arguments = [
        ('--sweep_config_file', str, 'e.g.: default_sweep_configuration.json'),
        ('--sweep_id', str, 'e.g.: 8j4z2x3z'),
        ('--n_runs', int, 'e.g.: 10'),
        ('--wandb_mode', str, 'e.g.: --wandb_mode online')
    ]
    flags = [
        ('--current_sweep', 'store_true', 'e.g.: --current_sweep'),
        ('--no_wandb', 'store_true', 'e.g.: --no_wandb'),
        ('--debug', 'store_true', 'e.g.: --debug'),
        ('--tune_batch_size', 'store_true', 'e.g.: --tune_batch_size')
    ]
    args = parse_arguments(arguments, flags)

    debug_active = args.debug if args.debug else False
    no_wandb = args.no_wandb if args.no_wandb else False
    tune_batch_size = args.tune_batch_size if args.tune_batch_size else False

    # check arguments
    smaller_one = int(
        0 + (1 if args.sweep_config_file else 0) + (1 if args.sweep_id else 0) + (1 if args.current_sweep else 0))
    if args.no_wandb and args.sweep_config_file is None:
        sweep_config_file = "sweep_configurations/default_sweep_config.json"
    assert smaller_one <= 1, "only one of sweep_config_file, sweep_id, current_sweep can be set"
    assert args.sweep_id == None or len(args.sweep_id) == 8, "sweep_id must be 8 characters long"
    if not args.wandb_mode:
        mode = 'online'
    else:
        mode = args.wandb_mode
    assert mode in ['offline', 'online', 'disabled']

    if args.no_wandb:
        # run without wandb ---------------------------------------------------------------
        if args.sweep_config_file:
            single_model_training(config_file=args.sweep_config_file)
        else:
            single_model_training(config_file=sweep_config_file)

    else:
        # create sweep, start -------------------------------------------------------------
        if args.sweep_config_file != None:
            sweep_id = create_sweep(args.sweep_config_file)
        elif args.sweep_id != None:
            sweep_id = args.sweep_id
        elif args.current_sweep:
            sweep_id = current_sweep_id()
        else:
            sweep_id = create_sweep()

        n_runs = 1 if args.n_runs == None else args.n_runs
        print(
            f'------------------------\nStarting {n_runs} runs in sweep with id: {sweep_id}\n------------------------')
        wandb.agent(
            sweep_id=sweep_id,
            function=single_model_training,
            entity=config['wandb'].get('entity', None),
            project=config['wandb']['project'],
            count=n_runs
        )
