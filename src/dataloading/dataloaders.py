import os
import numpy as np
import pytorch_lightning as pl

import pandas as pd

import torch
from torch import tensor, permute
from torch.utils.data import Dataset, DataLoader
from torch.nn import Embedding
from pysam import FastaFile

from features.nucleotide import BEDSeqLoader, DNATokenizer
from features.label import LabelLoaderHDF5
import pyranges
import atexit

from util.load_config import config
import sys

# always use os.environ.get(key, default_val) with environmental variables
NUM_WORKERS_ENV = int(os.environ.get('NSLOTS', os.environ.get('SLURM_JOB_CPUS_PER_NODE', 1)))


# config["num_workers_env"]
# not working: os.environ.get(config.num_workers_env, 1)

class CoverageDatasetHDF5(Dataset):

    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

    def __init__(self,
                 bed_path: str,
                 ref_path: str,
                 hdf5_path: str,
                 dna_tokenizer,
                 transform=None,
                 target_transform=None,
                 reverse_complement=False,
                 random_reverse_complement=False,
                 random_shift=0,
                 dtype=torch.float32,
                 debug = False
                 ):

        # TODO: make it possible to load only subsets of the data (?)

        # fetch scratch directory from config if it exists
        scratch_dir = config.get('scratch_dir', None)

        self.bedseqloader = BEDSeqLoader(bed_path, ref_path, random_shift=random_shift, scratch_dir=scratch_dir)
        self.labelloader = LabelLoaderHDF5(hdf5_path, bed_path)

        # sanity check
        assert len(self.bedseqloader) == len(self.labelloader)

        self.transform = transform
        self.target_transform = target_transform

        self.dna_tokenizer = dna_tokenizer

        self.reverse_complement = reverse_complement
        self.random_reverse_complement = random_reverse_complement
        self.random_shift = random_shift
        self.dtype = dtype

        self.chromosomes = self.bedseqloader.bed.Chromosome.cat.categories.values
        
        # this attribute can be overridden to prevent workers from using the same FastaFile
        self.fa = self.bedseqloader.fa 

        if reverse_complement & random_reverse_complement:
            print(
                'Warning, both "reverse_complement" and "random_reverse_complement" are True. Will serve up reverse-complement sequences randomly.')
            self.reverse_complement = False

        self._augment = True
        assert isinstance(debug, bool)
        self.debug = debug

    def augment_on(self):
        self._augment = True
        self.bedseqloader.random_shift = self.random_shift
        if self.debug:
            print('augment_on() called.')

    def augment_off(self):
        self._augment = False
        self.bedseqloader.random_shift = 0
        if self.debug:
            print('augment_off() called.')
        
    def fasta_worker_init(self):
        # used to overwrite fasta file attribute for a worker
        self.fa = FastaFile(self.bedseqloader.ref_path)
        atexit.register(self.cleanup)
        
    def cleanup(self):
        self.fa.close()

    def __len__(self):
        return len(self.bedseqloader)

    def __getitem__(self, idx):
        dnaseq_raw = self.bedseqloader.get_seq(idx, fa=self.fa)
        label = self.labelloader.get_labels_one_hot(idx)

        try:
            # TODO: I realize we should have a way of disabling random augmentations for the test set...
            dnaseq = self.dna_tokenizer.tokenize(dnaseq_raw, reverse_complement=self.reverse_complement, random=self.random_reverse_complement & self._augment)
        except KeyError as e:
            print("!!!! Tokenizer Key Error: see potentially problematic sequences / ranges below:")
            ranges = self.bedseqloader._get_range(idx)
            
            if isinstance(ranges, pd.DataFrame):
                for i, (j, r) in enumerate(ranges.iterrows()):
                    
                    tmp_seq = dnaseq_raw[i].replace('A','').replace('C','').replace('T','').replace('G','')
                    
                    if (len(tmp_seq) > 0):
                        print(f'{r.to_dict()}, sequence: {dnaseq_raw[i]}')
            elif isinstance(ranges, pd.Series):                
                print(f'{ranges.to_dict()}, sequence: {dnaseq_raw}')
                
            raise e

        if self.transform:
            dnaseq = self.transform(dnaseq)

        if self.target_transform:
            label = self.target_transform(label)

        if self.debug:
            return dnaseq.to(self.dtype), label.to(self.dtype), idx
        else:
            return dnaseq.to(self.dtype), label.to(self.dtype)

    def train_test_split_chromosomes(self, test, train=None):

        """
        Returns a train-test split based on chromosomes (indices)
        
        test:  the chromosomes in the test set
        train: the chromosomes in the training set (if None, will return all chromosomes that are not in the test set)
        
        """

        assert len(test) < len(
            self.chromosomes), 'Error: number of chromosomes in test set must be smaller than number of chromosomes'

        if not any([x in self.chromosomes for x in test]):
            print("removing/adding prefix from supplied chromosomes to match the reference genome...")
            if test[0].startswith("chr"):
                test = [x.replace('chr', '') for x in test]
                if train is not None:
                    train = [x.replace('chr', '') for x in train]
            else:
                test = [f'chr{x}' for x in test]
                if train is not None:
                    train = [f'chr{x}' for x in train]

        # todo: adjust chromosome names, add or remove "chr" prefix if necessary to match        

        assert all([x in self.chromosomes for x in
                    test]), 'Error: invalid test chromosomes specified. Available chromosomes are: {}'.format(
            self.chromosomes)

        i_test = self.bedseqloader.bed.Chromosome.isin(test)

        if train is not None:

            for c in train:
                assert c not in test, f'Error: test and train chromosomes overlap! (found "{c}" in both)'

            i_train = self.bedseqloader.bed.index[self.bedseqloader.bed.Chromosome.isin(train)].values
        else:

            i_train = self.bedseqloader.bed.index[~i_test].values
            i_test = self.bedseqloader.bed.index[i_test].values

        return i_train, i_test


def coveragedatasethdf5_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.fasta_worker_init()
    if dataset.debug:
        print(f'dataset has augment set to {dataset._augment}')
    
    
class LitCoverageDatasetHDF5(pl.LightningDataModule):
    """
        Subsetting Options:

        - 'no subsetting' 
            - it will do a normal 80/20 train/test split on the whole dataset
            - pass no arguments to region_subset_all, region_subset_train, region_subset_test

        - 'subset whole dataset'
            - pass a pyranges object to region_subset_all
            - it subsets the data according to your pyranges object
            - it will do a 80/18/2 train/test/val split on the subsetted data
            - e.g: region_subset_all=pyranges.from_dict({
            "Chromosome": [9, 10],
            "Start": [0, 0],
            "End": [100000000, 100000000],
            }), 
        
        - 'specify subset for train/test'
            - pass a pyranges object to region_subset_train and region_subset_test
            - with these you define the train and test set, it checks that they dont overlap
            - it creates a validation dataset from the train set
            - e.g:  region_subset_train=pyranges.from_dict({"Chromosome": [9, 10], "Start": [0, 0], "End": [100000000, 100000000]}), 
                    region_subset_test=pyranges.from_dict({Chromosome": [11, 12], "Start": [0, 0], "End": [100000000, 100000000],})
    """

    def __init__(self,
                 seq_len: int,
                 seq_order: int,
                 region_subset_all=None,
                 region_subset_train=None,
                 region_subset_test=None,
                 stride: int = 1,
                 one_hot: bool = True,
                 reverse_complement: bool = False,
                 random_reverse_complement: bool = False,
                 random_shift: int = 0,
                 basepath: str = 'data/processed/GRCh38/toydata',
                 # toydata or 221111_128bp_minoverlap64_mincov2_nc10_tissues
                 ref: str = 'GRCh38',
                 num_workers: int = NUM_WORKERS_ENV,
                 batch_size: int = 1024,
                 dtype=torch.float32,
                 worker_init_fn = coveragedatasethdf5_worker_init_fn,
                 debug = False,
                 **kwargs
                 ):
        super().__init__()

        # subsetting params
        self.region_subset_all = region_subset_all
        self.region_subset_train = region_subset_train
        self.region_subset_test = region_subset_test
        self.subsetting_method = "no subsetting"
        if self.region_subset_all:
            # check if region_subset_all is pyrange object
            assert isinstance(self.region_subset_all,
                              pyranges.pyranges.PyRanges), "the parameter pyranges needs to be of type PyRanges"
            assert self.region_subset_train is None and self.region_subset_test is None, "Error: region_subset_all is mutually exclusive with region_subset_train and region_subset_test"
            self.subsetting_method = "subset whole dataset"
        if self.region_subset_train or self.region_subset_test:
            assert isinstance(self.region_subset_train, pyranges.pyranges.PyRanges) and isinstance(
                self.region_subset_test,
                pyranges.pyranges.PyRanges), "the parameter pyranges needs to be of type PyRanges"
            assert self.region_subset_test and self.region_subset_train, "Error: region_subset_train and region_subset_test must be specified together"
            assert self.region_subset_all is None, "Error: region_subset_all is mutually exclusive with region_subset_train and region_subset_test"
            self.subsetting_method = "specify subset for train/test"

        self.basepath = basepath
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.seq_order = seq_order

        # data augmentation options
        self.reverse_complement = reverse_complement
        self.random_reverse_complement = random_reverse_complement
        self.random_shift = random_shift

        # set the DNA-tokenizer
        self.tokenizer = DNATokenizer(seq_order=seq_order, stride=stride, allow_N=True)
        
        # worker init function
        self.worker_init_fn = worker_init_fn

        # One-hot encoding of DNA-sequences
        if one_hot:
            # dna_embed will help us convert the token-representation to one-hot representation
            W, mapping = self.tokenizer.get_one_hot_weights_matrix(N_max=0)
            dna_embed = Embedding.from_pretrained(tensor(W), freeze=True)
            self.transform = lambda x: permute(dna_embed(tensor(x)), [0, 2, 1])
        else:
            self.transform = tensor

        if ref in ['GRCh38', 'hg38']:
            self.ref_path = config['reference']['GRCh38']
        elif ref == 'mm10':
            self.ref_path = config['reference']['mm10']

        self.bed_path = f'{self.basepath}/regions.bed' if os.path.exists(
            f'{self.basepath}/regions.bed') else f'{self.basepath}/regions.bed.gz'
        self.h5_path = f'{self.basepath}/overlaps.h5'

        self.data = CoverageDatasetHDF5(
            self.bed_path,
            self.ref_path,
            self.h5_path,
            dna_tokenizer=self.tokenizer,
            random_shift=self.random_shift,
            random_reverse_complement=self.random_reverse_complement,
            reverse_complement=self.reverse_complement,
            transform=self.transform,
            target_transform=tensor,
            dtype=dtype,
            debug=debug
        )

        self.data.bedseqloader.resize(self.seq_len + (self.seq_order - 1))

        # In theory, this could also depend on one or more filters
        self.n_classes = len(self.data.labelloader.label_ids)
        self.class_freq = (self.data.labelloader.label_N + 1) / len(self.data)

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP

        print("\n--------------- Subsetting Info ------------------")
        print("Subsetting Method:", self.subsetting_method, "\n")
        if self.subsetting_method == "subset whole dataset":
            i_subset_all = self.data.bedseqloader.get_indexes_overlapping_with(pyranges_input=self.region_subset_all)
            print("region_subset_all:\n", self.region_subset_all, "\n")
            subsetting_interval = len(i_subset_all) // 10
            i_train_percentage = 0.8
            i_val_percentage = 0.05

            i_train_interval = [0, int(subsetting_interval * i_train_percentage)]
            i_val_interval = [i_train_interval[1] + 1,
                              i_train_interval[1] + 1 + int(subsetting_interval * i_val_percentage)]
            i_test_interval = [i_val_interval[1] + 1, subsetting_interval - 1]

            i_subsetting_interval = np.arange(len(i_subset_all)) % subsetting_interval

            def get_subset_interval(interval):
                return i_subset_all[(i_subsetting_interval >= interval[0]) & (i_subsetting_interval <= interval[1])]

            i_train = get_subset_interval(i_train_interval)
            i_val = get_subset_interval(i_val_interval)
            i_test = get_subset_interval(i_test_interval)

        elif self.subsetting_method == "specify subset for train/test":
            # check if self.regions_subset_train and self.regions_subset_test are overlapping
            assert len(self.region_subset_train.overlap(
                self.region_subset_test)) == 0, "Error: region_subset_train and region_subset_test must not overlap"

            i_train = self.data.bedseqloader.get_indexes_overlapping_with(pyranges_input=self.region_subset_train)
            i_test = self.data.bedseqloader.get_indexes_overlapping_with(pyranges_input=self.region_subset_test)
            print("region_subset_train:\n", self.region_subset_train, "\n")
            print("region_subset_test:\n", self.region_subset_test, "\n")

            validation_percentage = 0.05
            n_intervals = 10
            interval_length = len(i_train) // n_intervals
            n_validation_datapoints_per_interval = ((len(i_train) + len(i_test)) * validation_percentage) // n_intervals
            is_i_val = (np.arange(len(i_train)) % interval_length) < n_validation_datapoints_per_interval
            i_val = i_train[is_i_val]
            i_train = i_train[~is_i_val]

        elif self.subsetting_method == "no subsetting":
            # TODO: make .train_test_split_chromosomes treat "chr9" and "9" the same
            i_train, i_test = self.data.train_test_split_chromosomes(['9', '10'])
            # TODO: perhaps change the way this works?

            validation_percentage = 0.05
            n_intervals = 10
            interval_length = len(i_train) // n_intervals
            n_validation_datapoints_per_interval = ((len(i_train) + len(i_test)) * validation_percentage) // n_intervals
            is_i_val = (np.arange(len(i_train)) % interval_length) < (n_validation_datapoints_per_interval)
            i_val = i_train[is_i_val]
            i_train = i_train[~is_i_val]
        else:
            raise Exception(f"Error: unknown subsetting method {self.subsetting_method}")

        len_i_train = len(i_train)
        len_i_val = len(i_val)
        len_i_test = len(i_test)
        len_subset = len_i_test + len_i_train + len_i_val
        len_all = len(self.data.bedseqloader.bed)
        print('Number of samples:')
        print(f"available = {len_all}")
        print(f"\tafter subsetting = {len_subset} ({round((len_subset / len_all) * 100)}% of available)")
        print(f"\ttraining = {len(i_train)} ({round((len_i_train / len_subset) * 100)}% of subset)")
        print(f"\tvalidation = {len(i_val)} ({round((len_i_val / len_subset) * 100)}% of subset)")
        print(f"\ttest = {len(i_test)} ({round((len_i_test / len_subset) * 100, 2)}% of subset)")
        print(f"for check: missed data in split = {len_subset - len_i_test - len_i_val - len_i_train}")
        print("----------------------------------------------------")

        if stage == "fit":
            # enable data augmentation (random shift and random reverse complement)
            self.data.augment_on()

            self.i_train = i_train
            self.i_val = i_val

        if stage in ["test", "predict"]:
            # disable data augmentation (random shift and random reverse complement)
            self.data.augment_off()
            self.i_test = i_test

    def train_dataloader(self):
        sampler = torch.utils.data.sampler.BatchSampler(
            torch.utils.data.sampler.SubsetRandomSampler(self.i_train),
            batch_size=self.batch_size,
            drop_last=False)
        self.data.augment_on()
        # probably has to be used with reload_dataloaders_every_n_epochs = 1 (?)
        return DataLoader(self.data, sampler=sampler, num_workers = self.num_workers, worker_init_fn = self.worker_init_fn)

    def val_dataloader(self):
        # we use a random sampler because the AUROC is calculated in chunks of batches, and sequential sampling may not be a good in that case
        sampler = torch.utils.data.sampler.BatchSampler(
            #self.i_val,
            torch.utils.data.sampler.SubsetRandomSampler(self.i_val),
            batch_size=self.batch_size,
            drop_last=False)
        self.data.augment_off()
        # probably has to be used with reload_dataloaders_every_n_epochs = 1 (?)
        # there's currently an issue where at least one worker appears to still have random sampling enabled on epoch 1, probably a lightning/pytorch bug.
        return DataLoader(self.data, sampler=sampler, num_workers = self.num_workers, worker_init_fn = self.worker_init_fn)

    def test_dataloader(self):
        sampler = torch.utils.data.sampler.BatchSampler(
            self.i_test,
            batch_size=self.batch_size,
            drop_last=False)
        self.data.augment_off()
        return DataLoader(self.data, sampler=sampler, num_workers=self.num_workers, worker_init_fn = self.worker_init_fn)


    
    # def _get_n_classes(self):

    #     with h5py.File(self.h5_path, 'r') as infile:
    #         n_classes = len(infile['target'].attrs.get('labels'))
    #         N = infile['target'].attrs.get('N') + 1

    #     if self.bed_path.endswith('.gz'):
    #         with gzip.open(self.bed_path, 'rt') as infile:
    #             for i, _ in enumerate(infile):
    #                 pass
    #             bed_len = i+1
    #     else:
    #         with open(self.bed_path, 'r') as infile:
    #             for i, _ in enumerate(infile):
    #                 pass
    #             bed_len = i+1

    #     class_freq = N / bed_len

    #     assert all(class_freq < 1.), 'Error: some classes have frequency 1. or above. Label metadata is likely corrupted. Or wrong BED-file was specified.'

    #     return n_classes, class_freq


class LitCoverageDatasetHDF5Mouse(LitCoverageDatasetHDF5):

    def __init__(self,
                 seq_len: int,
                 seq_order: int,
                 stride: int = 1,
                 one_hot: bool = True,
                 reverse_complement: bool = False,
                 random_reverse_complement: bool = False,
                 random_shift: int = 0,
                 basepath: str = 'data/processed/mm10/toydata',
                 ref: str = 'mm10',
                 num_workers: int = NUM_WORKERS_ENV,
                 batch_size: int = 1024,
                 ):
        super().__init__(basepath=basepath, ref=ref, seq_len=seq_len, seq_order=seq_order)


class LitCoverageDatasetHDF5MouseFull(LitCoverageDatasetHDF5):

    def __init__(self,
                 seq_len: int,
                 seq_order: int,
                 stride: int = 1,
                 one_hot: bool = True,
                 reverse_complement: bool = False,
                 random_reverse_complement: bool = False,
                 random_shift: int = 0,
                 basepath: str = 'data/processed/mm10/221111_128bp_minoverlap64_mincov2_nc10_tissues',
                 ref: str = 'mm10',
                 num_workers: int = NUM_WORKERS_ENV,
                 batch_size: int = 1024,
                 ):
        super().__init__(basepath=basepath, ref=ref, seq_len=seq_len, seq_order=seq_order)
