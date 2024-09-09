import torch
from torch.nn import Embedding
from torch import tensor

from features.nucleotide import DNATokenizer


def tokenize_extract(dna_df):
    # the tokenizer defines the token length, and the stride
    dnatokenizer = DNATokenizer(seq_order=2, stride=1, allow_N=True)

    # dna_embed will help us convert the token-representation to one-hot representation
    W, mapping = dnatokenizer.get_one_hot_weights_matrix(N_max=0)
    dna_embed = Embedding.from_pretrained(tensor(W), freeze=True)

    # this takes a little longer (30 sec)
    # tokenize
    tokenizeFunc = lambda list_str_seq, reverse: dna_embed(
        tensor(dnatokenizer.tokenize(list_str_seq, random=False, reverse_complement=reverse)))
    tokenizeNormal = lambda list_str_seq: tokenizeFunc(list_str_seq, reverse=False)
    tokenizeReverse = lambda list_str_seq: tokenizeFunc(list_str_seq, reverse=True)

    Xnormal = tokenizeNormal(dna_df['window'].tolist())
    Xreverse = tokenizeReverse(dna_df['window'].tolist())
    X = torch.cat((Xnormal, Xreverse), dim=0)

    return X, Xnormal, Xreverse
