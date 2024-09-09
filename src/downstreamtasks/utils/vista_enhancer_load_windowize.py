import sys
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import random

from features.nucleotide import BEDSeqLoader
from typing import List

from collections import Counter

sort_dict = lambda d: dict(sorted(d.items()))

from util.load_config import config

from utils.windowize import windowize
from utils.padding import get_padding_func


def vista_enhancer_load_windowize(bedfile_path, model_input_len, model_center_bin_len, shift_len, log_func=print):
    ref_path = config['reference']['GRCh38']

    # log_func(ref_path)

    def log_dataframe(d, comment):
        column_to_check = "enhancer_number" if "enhancer_number" in d.columns else "Name"
        log_func(f'----- {comment}: {len(d)} unique sequences: {d[column_to_check].nunique()}')

    # load bed file
    bed_file_path = bedfile_path
    bed_seq_vista = BEDSeqLoader(bed_path=bed_file_path, ref_path=ref_path,
                                 random_shift=0, ignore_strand=True)

    # extract dataframe
    dna_df = bed_seq_vista.bed[['Name', 'Score']]
    dna_df['seq'] = list(bed_seq_vista.get_seq(np.arange(len(bed_seq_vista.bed))))
    dna_df['seq_len'] = dna_df['seq'].apply(len)
    log_dataframe(dna_df, 'length of dataframe after loading bed file')

    # extract labels
    bed_data = pd.read_csv(bed_file_path, sep="\t", header=None)

    def extract_labels(label_str):
        if label_str == ".":
            return []
        else:
            # Split the string on commas to get individual labels
            labels = label_str.split(",")
            # Remove score information within square brackets for each label
            labels = [label.split('[')[0] for label in labels]
            return labels

    label_lists = bed_data[5].apply(extract_labels)
    label_lists = label_lists.tolist()

    dna_df['tissues'] = label_lists

    def calculate_gc_content(seq):
        g_count = seq.count('G')
        c_count = seq.count('C')
        total_count = len(seq)
        return (g_count + c_count) / total_count * 100

    dna_df['gc_content_enhancer'] = dna_df['seq'].apply(calculate_gc_content).tolist()

    dna_df['is_enhancer'] = dna_df['Score'].apply(lambda x: 1 if x == 'positive' else 0)
    dna_df.drop('Score', axis=1, inplace=True)

    dna_df['enhancer_number'] = dna_df['Name'].apply(lambda elementString: int(elementString.split("element")[-1]))
    dna_df.drop('Name', axis=1, inplace=True)

    # at this point: dna_df contains the following columns:
    # seq (AGCGT, ...), seq_len (640, ...), is_enhancer (0, 1, ...), enhancer_number (0, 1, 2, ...), tissues ("hindbrain(rhombencephalon),forebrain", "other", ...)

    # resize all sequence to 2176 + 6 with padding
    # if bigger: cut into pieces of 128 and pad with "N" to 2176
    # if smaller: pad with "N"
    # model_input_len = 2176 # IEA model
    model_input_len = model_input_len
    shift_len = shift_len

    window_len = model_input_len + shift_len
    # model_centre_focus_len = 128
    model_centre_focus_len = model_center_bin_len

    padding_func = get_padding_func(dna_df)
    dna_df = windowize(dna_df, window_len, model_centre_focus_len, padding_func)

    if not all(dna_df['window'].apply(len) == window_len):
        raise ValueError(f"Not all window strings have length {window_len}")
    log_func(f"window_len: {window_len}")

    dna_df['window_gc_content'] = dna_df['window'].apply(calculate_gc_content).tolist()

    log_dataframe(dna_df, 'length of dataframe after windowize')
    # dna_df = dna_df.sort_values(by=['Name', 'window_index']).reset_index(drop=True)
    return dna_df
