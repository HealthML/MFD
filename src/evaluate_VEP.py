import os
import numpy as np
import h5py
import gzip

from util.load_config import config
from scipy.stats import fisher_exact
import pandas as pd
import argparse
from dataloading.dataloaders import NUM_WORKERS_ENV


def get_args():
    
    p = argparse.ArgumentParser()

    p.add_argument('--vep', type=str, required=True, help='path to *_VEP.h5 file containing variant effect predictions')
    p.add_argument('--varid', type=str, required=True, help='path to *_varID.txt.gz file containing variant IDs')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--method', choices=['absolute','bidirectional'], default='absolute')
    p.add_argument('--pred_type',choices=[
        'predictions',
        'features_biological', 'features_technical',
        'logits_biological', 'logits_technical',
        'logits_biological_ver2', 'logits_technical_ver2',
        'predictions_biological', 'predictions_technical',
        'predictions_biological_ver2', 'predictions_technical_ver2',
    ], default='predictions', help='which prediction type to evaluate')

    args = p.parse_args()
    return args


def main():

    args = get_args()

    out_dir = os.path.dirname(args.vep)
    out_prefix = os.path.basename(args.vep).replace('.h5','')

    with gzip.open(args.varid, 'rt') as infile:
        varid = [x.rstrip() for x in infile]
    print(f'variant file contains {len(varid)} variants')

    variants = pd.read_csv('data/external/gnomAD/gnomad_intersected_with_encode_regulatory.tsv.gz', sep='\t', index_col=0)
    variants.set_index('id', drop=False, inplace=True)
    variants = variants.loc[varid]

    if args.debug:
        variants = variants.iloc[0:10000]

    counts_all = variants.label.value_counts()

    c_all = counts_all[['Rare','Common']].values
    c_rare_total = c_all[0]
    c_common_total = c_all[1]

    if args.method == 'absolute':
        Q = [0.1, 0.9, 0.99, 0.999, 0.9999]
    else:
        Q = [0.0001,0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999,0.9999]

    setnames = [args.pred_type] # could expand this to process multiple sets

    with h5py.File(args.vep, 'r') as infile:

        for i, setname in enumerate(setnames):

            if setname not in infile.keys():
                print(f'VEP file does not contain "{setname}", skipping...')
                continue

            print(f'processing {setname}')
            
            if args.debug:
                # load only the first 10000 if debugging
                pdiff = infile[setname][0:10000]
            else:
                # load only the first 10000 if debugging
                pdiff = infile[setname][:]

            # TODO: could also do PCA on variant effect predictions...
            # from sklearn.decomposition import PCA
            # pca = PCA()
            # X_pca = pca.fit_transform(pdiff)
            # # Calculate Variance Explained
            # explained_var = pca.explained_variance_ratio_
            # sns.lineplot(pca.explained_variance_ratio_[0:20])
            # # Find Components for 90% Variance:
            # cum_var = np.cumsum(explained_var)
            # n_90 = np.argmax(cum_var >= 0.99) + 1
            # n_90
            # ...
            
            def prep_cmat(v, q, direction='greater_equal'):
                
                # construct contingency matrix
                # v: an array of scores
                # q: the value by which to subset the array
                
                r = np.zeros(2, dtype=int)

                if direction == 'greater_equal':
                    c = variants.iloc[v >= q].label.value_counts()
                elif direction == 'less_equal':
                    c = variants.iloc[v <= q].label.value_counts()
                elif direction == 'less':
                    c = variants.iloc[v < q].label.value_counts()
                elif direction == 'greater':
                    c = variants.iloc[v > q].label.value_counts()
                else:
                    raise NotImplementedError(f'direction must be "greater_equal", "less_equal", "less" or "greater", got {direction}')
                
                try:
                    r[0] = c['Rare']
                except KeyError:
                    pass
                try:
                    r[1] = c['Common']
                except KeyError:
                    pass
                
                
                return np.array([[c_common_total - r[1], r[1]],[c_rare_total - r[0], r[0]]])
                    
            
            def calculate_enrichments(score):
                
                # perform statistical tests and calculate enrichments for a single or a set of scores
                # if there is more than one score, use the max(abs(score)) to collapse to a single value
                
                if args.method == 'absolute':
                    if score.ndim > 1:
                        max_abs = np.abs(score).max(axis=1)
                    else:
                        max_abs = np.abs(score)
                else:
                    assert score.ndim == 1
                    max_abs = score
                    
                quants = np.quantile(max_abs, Q)
                direction = ['greater_equal' if q >= 0.5 else 'less_equal' for q in Q]
                
                mat = [ prep_cmat(max_abs, q, direction=direction[i]) for i, q in enumerate(quants) ]
                tests = [ fisher_exact(m) for m in mat ]
                counts = [ c[:,1] for c in mat ]
                    
                return  tests, quants, counts
                
            results = []
            
            for i in range(pdiff.shape[1]):
                
                if args.debug:
                    if i > 10:
                        continue

                s = pdiff[:,i]
                
                test, quant, count = calculate_enrichments(s)
                
                # odds ratios
                OR = pd.Series([x.statistic for x in test])
                # pvalues
                pval = pd.Series([x.pvalue for x in test])
                # variant counts
                count = np.stack(count)
                
                df = pd.DataFrame({
                    'q': Q,
                    'cut': quant,
                    'Common': count[:,0],
                    'Rare': count[:,1],
                    'OR': OR,
                    'pval': pval
                })
                
                df['idx'] = i
                
                results.append(df)
            
            results = pd.concat(results, ignore_index=True)    
            
            outfile = f'{out_dir}/{out_prefix}.{setname}.{args.method}_results.tsv.gz'
            
            print(f'Completed. Saving results to {outfile}')
            
            results.to_csv(outfile, sep='\t', index=False)

            outfile_counts = f'{out_dir}/{out_prefix}.{setname}.{args.method}_totalcounts.tsv'
            print(f'writing total counts to {outfile_counts}')

            counts_all.to_csv(outfile_counts, sep='\t')


if __name__ == '__main__':
    main()



