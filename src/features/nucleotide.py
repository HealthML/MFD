# import pyranges
# import pyfaidx

from numpy.typing import ArrayLike
from typing import Union, Optional

from pysam import FastaFile
import gzip
import pandas as pd
import numpy as np
from itertools import product
import os
import pyranges
import torch
import glob
import shutil


class BEDSeqLoader():
    """
    Loader for regions defined in a UCSC-BED-file 
    """

    def __init__(self, bed_path: str,
                 ref_path: str,
                 random_shift: int = 0,
                 ignore_strand: bool = False,
                 scratch_dir: Optional[str] = None
                 ):

        """
        bed_path: path to UCSC-BED formatted file containing regions of interest
        ref_path: path to indexed FASTA reference genome
        """

        self.bed_path = bed_path

        if scratch_dir is None:
            self.ref_path = ref_path
        elif not os.path.isdir(scratch_dir):
            print(f'Warning: specified scratch directory does not exist! {scratch_dir}')
            self.ref_path = ref_path
        else:
            # option to create a copy of the reference in a scratch directory
            assert os.path.isdir(scratch_dir)
            try:
                idxfile = glob.glob(f'{ref_path}*fai')[0]
            except IndexError as e:
                raise FileNotFoundError(f'Cannot find index for {ref_path} fasta')
            target_file = os.path.join(scratch_dir, os.path.basename(ref_path))
            target_file_idx = os.path.join(scratch_dir, os.path.basename(idxfile))
            if not os.path.isfile(target_file):
                shutil.copy(ref_path, target_file)
            self.ref_path = target_file
            print(f'copied reference {ref_path} to scratch directory {self.ref_path}')
            if not os.path.isfile(target_file_idx):
                shutil.copy(idxfile, target_file_idx)
                
        if random_shift != 0:
            assert random_shift > 0, 'Error: random_shift must be greater than zero'
        self.random_shift = int(random_shift)

        # check the reference
        assert (os.path.isfile(ref_path + '.fai') or os.path.isfile(
            ref_path + '.gz.fai')), 'Error: no index found for Fasta-file: {}'.format(ref_path)
        self.fa = FastaFile(ref_path)

        # load the BED-file into memory
        if bed_path.endswith('.gz'):
            with gzip.open(bed_path, 'rt') as infile:
                first_line = infile.readline().rstrip()
        else:
            with open(bed_path, 'r') as infile:
                first_line = infile.readline().rstrip()

        n_tab, n_space = len(first_line.split('\t')), len(first_line.split(' '))

        sep = '\t' if n_tab > n_space else ' '
        assert max(n_tab,
                   n_space) > 1, 'Error: could not determine field separator of BED-file (file has to be space- or tab-separated)'

        n_col = len(first_line.split(sep))
        assert n_col >= 3, 'Error: need at least three columns in BED-file (Chromosome, Start and End)'
        expected_cols = ['Chromosome', 'Start', 'End', 'Name', 'Score', 'Strand'][0:n_col]

        self.bed = pd.read_csv(bed_path, header=None, usecols=np.arange(len(expected_cols)), names=expected_cols,
                               dtype={'Chromosome': 'category', 'Start': np.int64, 'End': np.int64}, na_values=['.'],
                               sep=sep)

        if 'Strand' in self.bed.columns:
            if ignore_strand:
                self.bed.drop(columns=['Strand'], inplace=True)
                self.stranded = False
            elif np.all(np.isnan(self.bed.Strand)):
                self.bed.drop(columns=['Strand'], inplace=True)
                self.stranded = False
            else:
                assert np.all(self.bed.strand.isin(['+',
                                                    '-'])), 'Error: BED-file contains strand information, but not all entries have strands "+" or "-". Strands can be ignored by setting ignore_strand = True'
                self.stranded = True
        else:
            self.stranded = False

        if np.any(~self.bed.Chromosome.cat.categories.isin(self.fa.references)):
            self.bed.Chromosome = self.bed.Chromosome.cat.rename_categories(
                {x: x.replace('chr', '') if str(x).startswith('chr') else x for x in
                 self.bed.Chromosome.cat.categories})

        assert all(self.bed.Chromosome.cat.categories.isin(
            self.fa.references)), f"Error: can't match all chromosomes in {bed_path} to those in reference {ref_path}"

        # needed to make reverse-complement
        self._rc = str.maketrans("ACGT", "TGCA")

    def __len__(self):
        return len(self.bed)

    def __getitem__(self, i):
        return self.get_seq(i)

    def _get_range(self, i):

        """
        returns the range i 
        """

        ranges = self.bed.iloc[i]

        if self.random_shift != 0:
            shift = np.random.choice(np.arange(-self.random_shift, self.random_shift + 1))
            if shift != 0:
                ranges = ranges.copy()
                ranges.Start += shift
                ranges.End += shift

        return ranges

    def get_seq(self, i, fa=None):

        """
        returns the sequence i (if more than one i are given, the result is ordered by genetic position)
        
        fa:    alternative open FastaFile object to use (used for multi-processing), defaults to self.fa
        """

        if fa is None:
            fa = self.fa

        ranges = self._get_range(i)

        if isinstance(ranges, pd.DataFrame):
            # ranges is a DataFrame
            if self.stranded:
                sequences = np.array([fa.fetch(x['Chromosome'], x['Start'], x['End']).upper() if x['Strand'] == '+' else fa.fetch(
                    x['Chromosome'], x['Start'], x['End']).upper().translate(self._rc)[::-1] for _, x in ranges.iterrows()])
            else:
                sequences = np.array(
                    [fa.fetch(x['Chromosome'], x['Start'], x['End']).upper() for _, x in ranges.iterrows()])
        else:
            # ranges is a Sequence
            seq = fa.fetch(ranges['Chromosome'], ranges['Start'], ranges['End']).upper()
            if self.stranded:
                seq = seq if ranges['Strand'] == '+' else seq.translate(self._rc)[::-1]
            sequences = np.array([seq])

        return sequences

    def unstrand(self):

        """
        remove strand information from regions
        """
        if 'Strand' in self.bed.columns:
            self.bed.drop(columns=['Strand'], inplace=True)
        self.stranded = False

    def resize(self, width):

        """
        resize regions fixed on their centers
        """

        centers = np.floor((self.bed.Start + self.bed.End) / 2).astype(int)

        left = np.floor(width / 2).astype(int)
        right = np.ceil(width / 2).astype(int)

        self.bed.Start = centers - left
        self.bed.End = centers + right

    def get_indexes_overlapping_with(self, pyranges_input) -> np.ndarray:
        assert isinstance(pyranges_input,
                          pyranges.pyranges.PyRanges), "the parameter pyranges needs to be of type PyRanges"

        # do we have to do that on every call?
        # because the bed file doesnt change this could be moved to the constructor
        self.bed['i'] = self.bed.index.values
        tmp_bed_pyranges = pyranges.PyRanges(df=self.bed, copy_df=False)

        return tmp_bed_pyranges.overlap(pyranges_input).i.values


class DNATokenizer():
    """
    Class that handles breaking up DNA-sequence-strings into oligo-nucleotide sequences and mapping to integer token IDs 
    """

    def __init__(self, seq_order: int = 1, stride: Optional[int] = None, allow_N: bool = True, N_max: int = 0):

        if allow_N:
            # maps (oligo-) nucleotides to integer token-IDs, e.g., AAA -> 0
            self.mapping = {''.join(x): i for i, x in
                            enumerate(list(product(*[['A', 'C', 'G', 'N', 'T'] for _ in range(seq_order)])))}
            # maps (oligo-) nucleotides to the integer token-IDs of their reverse-complement, e.g., AAA -> 124 (= TTT in the original mapping), or ACG -> 39 (= CGT in the original mapping)
            self.mapping_rc = {''.join(x[::-1]): i for i, x in
                               enumerate(list(product(*[['T', 'G', 'C', 'N', 'A'] for _ in range(seq_order)])))}
        else:
            # simplified version that doesn't have "N" (unknown) nucleotides
            self.mapping = {''.join(x): i for i, x in
                            enumerate(list(product(*[['A', 'C', 'G', 'T'] for _ in range(seq_order)])))}
            self.mapping_rc = {''.join(x[::-1]): i for i, x in
                               enumerate(list(product(*[['T', 'G', 'C', 'A'] for _ in range(seq_order)])))}

        self.stride = stride if stride is not None else 1
        self.seq_order = seq_order
        self.allow_N = allow_N
        self.N_max = N_max

        # sanity check
        # self._validate_rc_mapping()

    def _validate_rc_mapping(self):
        # validate the mapping is working as expected...
        rc_dict = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
        for k, v in self.mapping_rc.items():
            rc = k[::-1]
            rc = ''.join(rc_dict[o] for o in rc)
            print('{} -> {}, {} -> {}'.format(rc, self.mapping[rc], k, self.mapping_rc[k]))
            assert self.mapping[rc] == self.mapping_rc[
                k], 'Error: invalid reverse-complement mapping. please report this bug'

    def get_one_hot_weights_matrix(self, N_max: int = 0, reverse_complement=False):

        '''
        Returns a weight matrix which corresponds to 1-hot encoding of (oligo-)nucleotides, and the corresponding positional mapping (dict)
        
        The weight matrix can be passed to torch.nn.Embedding
        
        N_max controls the max number of unknown nucleotides ("N" in the FASTA) allowed before sequences are mapped to the null-encoding (vector of zeros).
        The default (N_max = 0) excludes (oligo-) nucleotides with any Ns (-> blinding)
        '''

        if not self.allow_N:
            idx = self.mapping if not reverse_complement else self.mapping_rc
            if N_max > 0:
                raise ValueError('N_max > 0, but object was created with allow_N = False !')
        else:
            mapping = self.mapping if not reverse_complement else self.mapping_rc
            idx = {}
            i = 0
            for k in mapping.keys():
                if str(k).count('N') > N_max:
                    continue
                else:
                    idx[k] = i
                    i += 1

        dim_embed = max(idx.values()) + 1

        weights = np.zeros((len(self.mapping), dim_embed))

        for i, (k, v) in enumerate(self.mapping.items()):
            if k in idx.keys():
                weights[i, idx[k]] = 1.

        return weights, idx

    def tokenize(self, sequences: Union[str, ArrayLike], reverse_complement: bool = False, random: bool = False):

        """
        Turns an array of DNA-strings (n,) into an array of (oligo-)nucleotide tokens
        If reverse_complement == True, will return the reverse-complement mapping
        """

        if random:
            reverse_complement = np.random.choice([0, 1]).astype(bool)

        if reverse_complement:
            if isinstance(sequences, str):
                return np.array([self.mapping_rc[sequences[i * self.stride:(i * self.stride + self.seq_order)]] for i in
                                 range((len(sequences) - self.seq_order) // self.stride + 1)])[np.newaxis, ::-1].copy()
            else:
                return np.array([[self.mapping_rc[s[i * self.stride:(i * self.stride + self.seq_order)]] for i in
                                  range((len(s) - self.seq_order) // self.stride + 1)] for s in sequences])[:,
                       ::-1].copy()

        else:
            if isinstance(sequences, str):
                return np.array([self.mapping[sequences[i * self.stride:(i * self.stride + self.seq_order)]] for i in
                                 range((len(sequences) - self.seq_order) // self.stride + 1)])[np.newaxis, :]
            else:
                return np.array([[self.mapping[s[i * self.stride:(i * self.stride + self.seq_order)]] for i in
                                  range((len(s) - self.seq_order) // self.stride + 1)] for s in sequences])


    def onehot_reverse_complement_func(self, use_numpy=False):

        """
        returns a function that peforms reverse complement mapping after one-hot encoding 
        useful for things like variant effect prediction
        """

        # these return different mapping from those stored in self.mapping and self.rc_mapping, because "N" nucleotides can be dropped.
        _, mapping = self.get_one_hot_weights_matrix(N_max = self.N_max, reverse_complement=False)
        _, mapping_rc = self.get_one_hot_weights_matrix(N_max = self.N_max, reverse_complement=True)

        rc_permute = [ mapping_rc[k] for k, _ in mapping.items() ]

        if use_numpy:
            def rcfun(x: np.ndarray):
                # this returns a view
                return x[:,rc_permute,::-1]
            return rcfun
        else:
            def rcfun(x: torch.Tensor):
                # this copies the data
                return torch.flip(x[:,rc_permute,:], (2,))
            return rcfun


# class BEDSeqLoader_BROKEN():
#    
#    """
#    Loader for regions defined in a UCSC-BED-file
#    
#    This loader breaks if the sequences are not ordered the same way pyranges sorts ranges...
#    """
#    
#    def __init__(self, bed_path, ref_path, index_label=None, random_shift=0):
#        
#        """
#        bed_path: path to sorted UCSC-BED file containing regions of interest
#        ref_path: path to indexed FASTA reference genome
#        """
#        
#        self.bed_path = bed_path
#        self.ref_path = ref_path
#        
#        self.bed = pyranges.read_bed(self.bed_path)
#        self.fa = pyfaidx.Fasta(self.ref_path, sequence_always_upper=True)
#        
#        if np.any(~self.bed.Chromosome.cat.categories.isin(self.fa.records.keys())):
#            self.bed.Chromosome = self.bed.Chromosome.cat.rename_categories({ x: x.replace('chr','') if str(x).startswith('chr') else x for x in self.bed.Chromosome.cat.categories})
#            
#        assert all(self.bed.Chromosome.cat.categories.isin(self.fa.records.keys())), f"Error: can't match all chromosomes in {bed_path} to those in reference {ref_path}"
#        
#        if not all(self.bed.lengths() == self.bed.lengths()[0]):
#            print(f'Warning: The ranges in {bed_path} have different lengths.')
#        
#        # store original 0-based position in BED-file
#        self.bed.i = np.arange(len(self.bed))
#        self.set_index_label('i')
#        
#        self.random_shift = int(random_shift)
#    
#    
#    def __len__(self):
#        return len(self.bed)
#    
#    def __getitem__(self,i):
#        return self.get_seq(i)
#    
#    def _get_range(self, i):
#        
#        """
#        returns the range i (if more than one i are given, the result is ordered by genetic position)
#        """
#        
#        tmp = np.zeros(len(self.bed), dtype='bool')
#        
#        try:
#            tmp.itemset(i, True)
#            ranges = self.bed[tmp]
#        except TypeError:
#            # if i was a list or numpy-array
#            tmp[i] = True
#            ranges = self.bed[tmp]
#            
#        if self.random_shift != 0:
#            assert self.random_shift > 0
#            shift = np.random.choice(np.arange(-self.random_shift, self.random_shift+1))
#            # print(shift)
#            ranges.Start += shift
#            ranges.End += shift
#            
#        return ranges
#    
#    def get_seq(self, i):
#        
#        """
#        returns the sequence i (if more than one i are given, the result is ordered by genetic position)
#        """
#        
#        ranges = self._get_range(i)
#        sequences = pyranges.get_fasta(ranges, pyfaidx_fasta=self.fa)
#        sequences.index = ranges.df[self.index_label]
#        
#        return sequences
#    
#    
#    def unstrand(self):
#        
#        """
#        remove strand information from regions
#        """
#        
#        self.bed = self.bed.unstrand()
#        
#        
#    def set_index_label(self, label=None):
#        
#        assert label in self.bed.df.columns, f'Error, requested label "{label}" does not exist in the BED-file'
#        
#        self.index_label = label
#        
#    
#    def resize(self, width):
#        
#        """
#        resize regions fixed on their centers
#        """
#    
#        def _resize_func(df):
#            
#            centers = np.floor((df.Start + df.End)/2).astype(int)
#            
#            left = np.floor(width/2).astype(int)
#            right = np.ceil(width/2).astype(int)
#            
#            df.Start = centers - left
#            df.End = centers + right
#            
#            return df
#            
#    
#        self.bed = self.bed.apply(_resize_func)
#
