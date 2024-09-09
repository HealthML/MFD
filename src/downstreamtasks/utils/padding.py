from features.nucleotide import DNATokenizer
import numpy as np
import random

def get_padding_func(dna_df):
    # Define Padding function
    tokenizer = DNATokenizer(seq_order=2, stride=1, allow_N=True)
    forward_tokens = [tokenizer.tokenize(x) for x in dna_df.seq]
    rc_tokens = [tokenizer.tokenize(x, reverse_complement = True) for x in dna_df.seq]

    forward_tokens = np.concatenate(forward_tokens, axis=1)
    rc_tokens = np.concatenate(rc_tokens, axis = 1)

    tokens = np.concatenate([forward_tokens, rc_tokens])
    counts = np.unique(tokens, return_counts=True)
    frequencies = counts[1] / counts[1].sum()

    revert_mapping = {v: k for k, v in tokenizer.mapping.items()}
    dinucleotide_freq = {revert_mapping[j]: frequencies[i] for i, j in enumerate(counts[0])}
    dinucleotide_freq_no_cg = {k: v for k, v in dinucleotide_freq.items() if 'CG' not in k}

    def sample_sequence(length, n, freq):    
        return ''.join(np.random.choice(n, size = length//2, replace = True, p = freq))

    def add_frequency_dna_n(length):
        assert length > 0
        if length % 2 != 0:
            return sample_sequence(length-1, list(dinucleotide_freq.keys()), list(dinucleotide_freq.values()) ) + 'N'
        return sample_sequence(length, list(dinucleotide_freq.keys()), list(dinucleotide_freq.values()) )

    # for padding testing
    def get_random_dna_seq(length):
        return ''.join(random.choice(['A', 'C', 'G', 'T']) for _ in range(length))
    
    return add_frequency_dna_n