
import pandas as pd
import numpy as np
import h5py
import argparse
import os
import gzip


def get_args():
    default_pv_thresh = 1e-5
    quantiles = ','.join(['0.001','0.999'])
    p = argparse.ArgumentParser()
    p.add_argument('--varid', help='path to *_varID.txt.gz file containing variant IDs', required=True)
    p.add_argument('--vep', help='path to *_VEP.h5 file containing variant effect predictions', required=True)
    p.add_argument('--result_file', help='path to *_results.tsv.gz file containing rare-variant enrichments at different quantiles', required=True)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--pred_type',choices=[
        'predictions',
        'features_biological', 'features_technical',
        'logits_biological', 'logits_technical',
        'logits_biological_ver2', 'logits_technical_ver2',
        'predictions_biological', 'predictions_technical',
        'predictions_biological_ver2', 'predictions_technical_ver2',
    ], required=True, help='which prediction type to evaluate')
    p.add_argument('--pvalue_thresh', help=f'p-value threshold used to filter features (default: {default_pv_thresh})', type=float, default=default_pv_thresh)
    p.add_argument('--quantiles', help=f'comma-separated list of quantiles to investigate (default = {quantiles})', default=quantiles)
    args = p.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()

    # set up
    quantiles_of_interest = args.quantiles.split(',')
    for quant_str in quantiles_of_interest:
        try:
            float(quant_str)
        except ValueError as e:
            print(f'Error casting quantile "{quant_str}" to float!')
            raise e
    variants_all = pd.read_csv(args.varid, header=None)
    if args.debug:
        print('--debug active, subsetting to 10000 variants.')
        variants_all = variants_all.iloc[0:10000].copy()
    result_file = args.result_file
    vep_file = args.vep
    setname = args.pred_type
    pval_thresh = args.pvalue_thresh

    if setname not in os.path.basename(result_file):
        print('Warning: normally the input result file name should contain the setname...')

    quantiles_all = pd.read_csv(result_file, sep='\t')

    with h5py.File(vep_file,mode='r') as h5:
        if args.debug:
            vep =  h5[setname][0:10000]
        else:
            vep =  h5[setname][:]

    assert len(vep) == len(variants_all)
    # set up done.

    # messages
    print(f'found predictions for {vep.shape[1]} features (setname = "{setname}")')
    print(f'investigating variant effect predictions at {len(quantiles_of_interest)} quantiles.')
    print(f'the p-value threshold for feature selection is {pval_thresh}')

    for q in quantiles_of_interest:
        
        print(f'processing quantile {q}')
        
        # prepare output file
        q_str = q.replace('.','_')
        out_prefix = result_file.replace('.tsv.gz','')
        out_file = f'{out_prefix}.q{q_str}_variants.tsv.gz'

        # select features at the quantile of interest
        quantiles = quantiles_all.query(f'q == {q}')
        # select only those features that significantly enrich for rare variants at the given quantile
        quantiles = quantiles.query(f'pval < {pval_thresh} & OR > 1') 
        
        if quantiles.shape[0] == 0:
            print(f'Warning: No features enriched for rare variants at quantile {q}, p-value threshold {pval_thresh}. Results table will be empty.')
            with gzip.open(out_file, 'wt') as outfile:
                outfile.write('varID\tN\ti_select\n')
            continue

        
        print(f'selected {quantiles.shape[0]} features.')

        # subset the variant effect predictions to those with significant enrichment of rare variants
        vep_select = vep[:,quantiles.idx.values]

        # this has shape (n_variants, n_selected_features) and indicates if the variant reached the quantile
        if float(q) >= 0.5:
            bool_matrix = np.apply_along_axis(arr=vep_select, axis=0, func1d=lambda x: x >= np.quantile(x, float(q)))
        else:
            bool_matrix = np.apply_along_axis(arr=vep_select, axis=0, func1d=lambda x: x <= np.quantile(x, float(q)))

        # this is a vector containing how many times the variant was selected
        n_select = bool_matrix.sum(axis=1)
        # subset to those selected at least once
        selected_variants = variants_all.loc[n_select > 0,:].copy()
        print(f'selected {selected_variants.shape[0]} variants at quantile {q} from {quantiles.shape[0]} features')

        selected_variants['N'] = n_select[n_select > 0]
        bool_matrix = bool_matrix[n_select > 0,:]

        # store which features selected each variant
        idx_str = quantiles.idx.astype('str').values
        selected_variants['i_select'] = [ ','.join(idx_str[bool_matrix[i,:]]) for i in range(bool_matrix.shape[0])] # has to be done this way to avoid a type casting issue that truncates the resulting strings in numpy
        selected_variants.rename(columns={0:'varID'}, inplace=True)

        # save to tsv
        print(f'saving results to {out_file}')
        selected_variants.to_csv(out_file, sep='\t', index=False)


