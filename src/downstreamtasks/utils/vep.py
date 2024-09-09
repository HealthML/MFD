from pysam import VariantFile, FastaFile
from math import ceil, floor
import os
import torch
from abc import ABC
from torch import permute, tensor
from torch.utils.data import Dataset
from torch.nn import Embedding
from features.nucleotide import DNATokenizer

class Variant():

    def __init__(self, chrom, ref, alts, id = None):
        self.chrom = str(chrom)
        self.ref = ref
        self.alts = [alts] if isinstance(alts, str) else alts
        self.id = id


class VariantParser():

    def __init__(self, ref_fa_path, bin_size = 100, tie = "l"):

        '''
        :param str ref_fa_path: Path to indexed reference fasta
        :param str vcf_path: Path to indexed vcf
        :param int bin_size: Length of the DNA-sequences (centered on the start position of the variant)
        :param str tie: If the bin size is even, where to position the variant ("l" for left of center, "r" for right of center)
        '''
        
        self.ref_fa_path = ref_fa_path
        self.ref = FastaFile(ref_fa_path)
        assert os.path.isfile(ref_fa_path + '.fai'), 'Error: no index found for Fasta-file: {}'.format(ref_fa_path)
        self.bin_size = bin_size
        assert tie in ['l','r']
        self.tie = tie
        
        if (self.bin_size % 2) == 0: 
            self.left_window = int(self.bin_size / 2 - 1) if self.tie == 'l' else int(self.bin_size / 2)
            self.right_window = int(self.bin_size / 2 + 1) if self.tie == 'l' else int(self.bin_size / 2)
        else:
            self.left_window = floor(self.bin_size/2)
            self.right_window = ceil(self.bin_size/2)
            
        self.n_variants, self.records = self._initialize_records()

    def _initialize_records(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self):
        raise NotImplementedError
    
    def get_variant_seq(self, variant):
        '''
        get variant ref and alt sequences
        '''
        pos = variant.pos - 1
        len_ref = len(variant.ref)
        len_alt = len(variant.alts[0])
        d = len_alt - len_ref if len_alt < len_ref else 0

        full_seq = self.ref.fetch(variant.chrom, pos - self.left_window, pos + self.right_window - d).upper()
        
        left_seq = full_seq[:self.left_window]
        
        right_seq_ref, right_seq_alt = full_seq[(self.left_window+len_ref):][:(self.right_window-len_ref)], full_seq[(self.left_window+len_ref):][:(self.right_window-len_alt - d)]
            
        return f'{left_seq}{variant.ref}{right_seq_ref}'.upper(), f'{left_seq}{variant.alts[0]}{right_seq_alt}'.upper()

    def worker_init(self):
        # have to open new file handles when doing multiprocessing to avoid nasty bugs
        self.ref = FastaFile(self.ref_fa_path)
        
    def get_ref(self, variant):
        '''
        get reference sequence for a variant
        '''
        pos = variant.pos - 1

        full_seq = self.ref.fetch(variant.chrom, pos - self.left_window, pos + self.right_window).upper()
        return full_seq
        
    
    def is_compatible(self, variant):
        '''
        simple test for compatibility 
        '''
        if len(variant.alts) > 1:
            # exclude variants with multiple alternative alleles
            return False
        if (len(variant.ref) > 32) | (len(variant.alts[0]) > 32):
            # exclude long structural variants
            return False
        else:
            return True
    

class DFVariantParser(VariantParser):
    
    def __init__(self, ref_fa_path, variant_df, bin_size = 100, tie = "l"):

        '''
        :param str ref_fa_path: Path to indexed reference fasta
        :param str variant_df: A data frame with the variants to parse (expected columns: ['chrom','id','ref','alts'])
        :param int bin_size: Length of the DNA-sequences (centered on the start position of the variant)
        :param str tie: If the bin size is even, where to position the variant ("l" for left of center, "r" for right of center)
        '''
        
        self.ref_fa_path = ref_fa_path
        self.ref = FastaFile(ref_fa_path)
        assert os.path.isfile(ref_fa_path + '.fai'), 'Error: no index found for Fasta-file: {}'.format(ref_fa_path)
        self.bin_size = bin_size
        assert tie in ['l','r']
        self.tie = tie
        
        if (self.bin_size % 2) == 0: 
            self.left_window = int(self.bin_size / 2 - 1) if self.tie == 'l' else int(self.bin_size / 2)
            self.right_window = int(self.bin_size / 2 + 1) if self.tie == 'l' else int(self.bin_size / 2)
        else:
            self.left_window = floor(self.bin_size/2)
            self.right_window = ceil(self.bin_size/2)
        
        self.variant_df = variant_df
        self.n_variants, self.records = self._initialize_records()
        
    def __getitem__(self, i):
        '''
        returns variant_id, (ref_sequence, alt_sequence), code
        '''
        return self.records[i][0].id, self.get_variant_seq(self.records[i][0]), self.records[i][1]
    
    def __len__(self):
        return self.n_variants
        
    def _initialize_records(self):
        
        records = {}
        
        n_all = 0
        n = 0
        incompatible_count = 0
        flipped_count = 0
        bad_count = 0
            
        for _, record in self.variant_df.iterrows():
                    
            """
            after calling _initialize_records(), self.records will contain a dictionary of all variants that pass filtering, with an integer index
    
            variant sequences can be loaded with __getitem__:
    
                variant_id, (ref_sequence, alt_sequence), code = dfvariantparser[i]
    
            code will contain the information if the variant was potentially flipped (the alternative allele matched the reference sequence)
    
            """
                
            n_all += 1
            code = 0
            
            record['alts'] = [record['alts']]
            
            if not self.is_compatible(record):
                incompatible_count += 1
                print(record)
                continue
                
            ref_seq = self.ref.fetch(record.chrom, record.pos - 1, record.pos - 1 + len(record.ref))
            
            if ref_seq != record.ref:
                if ref_seq == record.alts[0]:
                    flipped_count += 1
                    code = 1
                else:
                    bad_count += 1
                    continue
                    
            records[n] = (record, code)
            n+=1
                
        print(f'Parsed {n_all} variants.\n{incompatible_count} were excluded because they were multi-allelic or longer than 32bp.\n{flipped_count} were flipped.\n{bad_count} were excluded because they could not be matched to the reference.\n{n} variants remain.')
                    
        return n, records


class VcfParser(VariantParser):
    
    '''
    Class to retrieve variants inside a VCF
        
    :ivar pysam.VariantFile vcf: pysam VariantFile, the variant calls
    :ivar pysam.FastaFile ref: pysam FastaFile, the reference sequence
    :ivar int bin_size: size of the surrounding DNA-sequences to load
    :ivar int n_variants: number of variants after filtering
    '''
    
    def __init__(self, ref_fa_path, vcf_path=None, bin_size=100, tie='l'):
        '''
        :param str ref_fa_path: Path to indexed reference fasta
        :param str vcf_path: Path to indexed vcf
        :param int bin_size: Length of the DNA-sequences (centered on the start position of the variant)
        :param str tie: If the bin size is even, where to position the variant ("l" for left of center, "r" for right of center)
        '''
        self.vcf_path = vcf_path
        self.vcf = VariantFile(vcf_path, drop_samples=True) if vcf_path is not None else None
        self.ref_fa_path = ref_fa_path
        self.ref = FastaFile(ref_fa_path)
        assert os.path.isfile(ref_fa_path + '.fai'), 'Error: no index found for Fasta-file: {}'.format(ref_fa_path)
        self.bin_size = bin_size
        assert tie in ['l','r']
        self.tie = tie
        
        if (self.bin_size % 2) == 0: 
            self.left_window = int(self.bin_size / 2 - 1) if self.tie == 'l' else int(self.bin_size / 2)
            self.right_window = int(self.bin_size / 2 + 1) if self.tie == 'l' else int(self.bin_size / 2)
        else:
            self.left_window = floor(self.bin_size/2)
            self.right_window = ceil(self.bin_size/2)
            
        self.n_variants, self.records = self._initialize_records()
        
    def worker_init(self):
        # have to open new file handles when doing multiprocessing to avoid nasty bugs
        self.ref = FastaFile(self.ref_fa_path)
        self.vcf = VariantFile(self.vcf_path, drop_samples=True) if self.vcf_path is not None else None

    def __len__(self):
        return self.n_variants
    
    def __getitem__(self, i):
        '''
        returns variant_id, (ref_sequence, alt_sequence), code
        '''
        return self.records[i][0].id, self.get_variant_seq(self.records[i][0]), self.records[i][1]
        
    def _initialize_records(self):
        
        """
        after calling _initialize_records(), self.records will contain a dictionary of all variants that pass filtering, with an integer index

        variant sequences can be loaded with __getitem__:

            variant_id, (ref_sequence, alt_sequence), code = vcfparser[i]

        code will contain the information if the variant was potentially flipped (the alternative allele matched the reference sequence)

        """

        records = {}
        
        if self.vcf is None:
            print('Warning: VcfAltParser initialized without a VCF file. This is typically only useful for debugging...')
            return 0, None
        
        n_all = 0
        n = 0
        incompatible_count = 0
        flipped_count = 0
        bad_count = 0
        
        for record in self.vcf.fetch(reopen=True):
            
            n_all += 1
            code = 0
            
            if not self.is_compatible(record):
                incompatible_count += 1
                continue
                
            ref_seq = self.ref.fetch(record.chrom, record.pos - 1, record.pos - 1 + len(record.ref))
            
            if ref_seq != record.ref:
                if ref_seq == record.alts[0]:
                    flipped_count += 1
                    code = 1
                else:
                    bad_count += 1
                    continue
                    
            records[n] = (record, code)
            n+=1
            
        print(f'Parsed {n_all} variants.\n{incompatible_count} were excluded because they were multi-allelic or longer than 32bp.\n{flipped_count} were flipped.\n{bad_count} were excluded because they could not be matched to the reference.\n{n} variants remain.')
                
        return n, records


class DFVariantDataset(Dataset):
    
    def __init__(self,
                 ref_fa_path,
                 variant_df,
                 seq_len,
                 seq_order,
                 stride = 1,
                 one_hot = True,
                 dtype=torch.float32,
                 load = 'ref',
                ):
        
        '''
        Wraps a pytorch Dataset around a VcfParser and performs the appropriate data transformations (e.g., tokenization)

        ref_fa_path: the path to the indexed FASTA reference sequence
        variant_df: A data frame with the variants to parse (expected columns: ['chrom','id','ref','alts'])
        seq_len: the sequence length to load around the variants. This should match the model input length
        seq_order: the sequence order (passed to the tokenizer)
        stride: the stride (passed to the tokenizer)
        one_hot: if sequences should be one-hot encoded
        dtype: the torch dtype to cast all data to
        load: wether to load the reference ("ref") or alternative sequences ("alt")

        the load-attribute can be modified after instantiation to switch between loading of reference or alternative sequences
        this is preferable over re-instantiating the object, because the variant compatibility checks etc can be skipped.
        
        '''
        
        assert load in ['ref','alt'], 'load has to be either "ref" or "alt"'
        
        self.data = DFVariantParser(ref_fa_path=ref_fa_path, variant_df=variant_df, bin_size = seq_len + (seq_order - 1))
        self.tokenizer = DNATokenizer(seq_order = seq_order, allow_N = True, stride = stride)
        self.dtype = dtype
        
        # One-hot encoding of DNA-sequences
        if one_hot:
            # dna_embed will help us convert the token-representation to one-hot representation
            W, mapping = self.tokenizer.get_one_hot_weights_matrix(N_max=0)
            dna_embed = Embedding.from_pretrained(tensor(W), freeze=True)
            self.transform = lambda x: permute(dna_embed(tensor(x)), [0, 2, 1])
        else:
            self.transform = tensor
            
        self.load = load
            
    def __getitem__(self, i):
        
        varid, (ref, alt), code = self.data[i]
        
        if self.load == 'ref':
            dnaseq = self.tokenizer.tokenize(ref)
        else:
            dnaseq = self.tokenizer.tokenize(alt)
            
        dnaseq = self.transform(dnaseq)
        
        return dnaseq.to(self.dtype)
    
    def __len__(self):
        return len(self.data)


class VcfDataset(Dataset):
    
    def __init__(self,
                 ref_fa_path,
                 vcf_path,
                 seq_len,
                 seq_order,
                 stride = 1,
                 one_hot = True,
                 dtype=torch.float32,
                 load = 'ref',
                ):
        
        '''
        Wraps a pytorch Dataset around a VcfParser and performs the appropriate data transformations (e.g., tokenization)

        ref_fa_path: the path to the indexed FASTA reference sequence
        vcf_path: the path to the VCF file
        seq_len: the sequence length to load around the variants. This should match the model input length
        seq_order: the sequence order (passed to the tokenizer)
        stride: the stride (passed to the tokenizer)
        one_hot: if sequences should be one-hot encoded
        dtype: the torch dtype to cast all data to
        load: wether to load the reference ("ref") or alternative sequences ("alt")

        the load-attribute can be modified after instantiation to switch between loading of reference or alternative sequences
        this is preferable over re-instantiating the object, because the variant compatibility checks etc can be skipped.
        
        '''
        
        assert load in ['ref','alt'], 'load has to be either "ref" or "alt"'
        
        self.data = VcfParser(ref_fa_path=ref_fa_path, vcf_path=vcf_path, bin_size = seq_len + (seq_order - 1))
        self.tokenizer = DNATokenizer(seq_order = seq_order, allow_N = True, stride = stride)
        self.dtype = dtype
        
        # One-hot encoding of DNA-sequences
        if one_hot:
            # dna_embed will help us convert the token-representation to one-hot representation
            W, mapping = self.tokenizer.get_one_hot_weights_matrix(N_max=0)
            dna_embed = Embedding.from_pretrained(tensor(W), freeze=True)
            self.transform = lambda x: permute(dna_embed(tensor(x)), [0, 2, 1])
        else:
            self.transform = tensor
            
        self.load = load
            
    def __getitem__(self, i):
        
        varid, (ref, alt), code = self.data[i]
        
        if self.load == 'ref':
            dnaseq = self.tokenizer.tokenize(ref)
        else:
            dnaseq = self.tokenizer.tokenize(alt)
            
        dnaseq = self.transform(dnaseq)
        
        return dnaseq.to(self.dtype)
    
    def __len__(self):
        return len(self.data)


def variantparser_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.data.worker_init()
    


