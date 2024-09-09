import os
import torch
import pytorch_lightning as pl
import numpy as np
import h5py
import gzip

from models import models
from util.load_config import config
import pandas as pd
from torch.utils.data import DataLoader
from downstreamtasks.utils.vep import DFVariantDataset
from downstreamtasks.utils.vep import variantparser_worker_init_fn
import argparse
from dataloading.dataloaders import NUM_WORKERS_ENV
from torch import sigmoid


def get_args():

    p = argparse.ArgumentParser()

    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--set', type=str, choices=['ctcf', 'pls', 'els'], default='els')
    p.add_argument('--model_cls', type=str, default='IEAquaticDilated')
    p.add_argument('--ref', type=str, default='GRCh38')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--feature_predict', choices=['logits','predictions'], default='logits', help='for the feature-specific outputs: whether to store logits ("logits") or sigmoid(logit+bias) ("predictions")')

    args = p.parse_args()
    return args


def collate_fn(batch):
    batch = torch.cat([x for x in batch], dim=0)
    return batch


class ReverseComplementShiftWrapper(pl.LightningModule):

    def __init__(self, model, n_shift, reverse_complement_func, model_cls, feature_predict):
        super().__init__()
        self.model = model
        self.n_shift = n_shift -1
        self.reverse_complement_func = reverse_complement_func
        self.model_cls = model_cls
        self.feature_predict = feature_predict
        assert feature_predict in ['predictions','logits']


    def stack_mean(self, li):
        return torch.mean(torch.stack(li), dim=0).to(torch.float16).detach().cpu().numpy()

    def forward(self, x):

        forward = x
        reverse_complement = self.reverse_complement_func(forward)

        predictions_l = []
        logits_l = []
        features_biological_l = []
        features_technical_l = []
        logits_biological_l = []
        logits_technical_l = []

        for i in range(self.n_shift):
            for s in [forward, reverse_complement]:

                input_crop = s[:, :, i: -self.n_shift + i]

                if self.model_cls in ['IEAquaticDilated']:

                    predictions, _, features_biological, features_technical = self.model.forward(input_crop, return_features=True)

                    predictions_l.append(predictions)
                    features_biological_l.append(features_biological)
                    features_technical_l.append(features_technical)

                    predictions, _, logits_biological, logits_technical = self.model.get_logits(features_biological, features_technical)

                    if self.feature_predict == 'predictions':
                        logits_biological = sigmoid(logits_biological + self.model.bias)
                        logits_technical = sigmoid(logits_technical + self.model.bias)

                    logits_biological_l.append(logits_biological)
                    logits_technical_l.append(logits_technical)

                else:
                    features = self.model.features(input_crop)
                    predictions, logits, _, _ = self.model.get_logits(features[:, :, 0], features)
                    predictions_l.append(predictions)
                    logits_l.append(logits)


        if self.model_cls in ['IEAquaticDilated']:
            output = [ self.stack_mean(predictions_l), self.stack_mean(features_biological_l), self.stack_mean(features_technical_l), self.stack_mean(logits_biological_l), self.stack_mean(logits_technical_l) ]
            return output
        else:
            return [self.stack_mean(predictions_l),  self.stack_mean(logits_l) ]

def main():

    args = get_args()

    out_dir = os.path.dirname(args.ckpt)
    out_prefix = os.path.basename(args.ckpt).replace('.ckpt','')

    variants = pd.read_csv('data/external/gnomAD/gnomad_intersected_with_encode_regulatory.tsv.gz', sep='\t', index_col=0)
    variants = variants[variants[args.set].values == 1].copy()

    if args.debug:
        variants = variants.iloc[0:96]

    model_cls_str = torch.load(args.ckpt, map_location='cpu')[
        'hyper_parameters'].get('model_class', 'IEAquaticDilated')
    model_cls = getattr(models, model_cls_str)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model_cls.load_from_checkpoint(args.ckpt, map_location=device).eval()

    def dump_hdf5(ref_pred, alt_pred, path, dtype='f2'):

        assert isinstance(ref_pred, list)
        assert isinstance(alt_pred, list)

        if model_cls_str in ['IEAquaticDilated']:
            setnames = [
                'predictions',
                'features_biological',
                'features_technical',
                f'{args.feature_predict}_biological',
                f'{args.feature_predict}_technical',
                f'{args.feature_predict}_biological_ver2',
                f'{args.feature_predict}_technical_ver2',
            ]
        else:
            setnames = ['predictions', 'logits']

        with h5py.File(path, mode='w') as outfile:

            for i, _ in enumerate(ref_pred):

                diff_scores = alt_pred[i] - ref_pred[i]
                outfile.create_dataset(name=setnames[i], data=diff_scores, dtype=dtype, compression=6)
                outfile.flush()

    variants['Chromosome'] = variants.Chromosome.astype(str)
    variants.rename(columns={'Chromosome':'chrom', 'alt':'alts'}, inplace=True)

    shift_len = 8 # number of shift to apply to every sequence +- the center

    dataset = DFVariantDataset(ref_fa_path = config['reference'][args.ref],
                            variant_df = variants,
                            seq_order = model.hparams.seq_order,
                            seq_len = model.hparams.seq_len + shift_len,
                            load = 'ref')

    outfile = f'{out_dir}/{out_prefix}.gnomAD_{args.set}_varID.txt.gz'
    with gzip.open(outfile, 'wt') as out:
        for i, (record, code) in dataset.data.records.items():
            out.write(f'{record.id}\n')

    reverse_complement_func = dataset.tokenizer.onehot_reverse_complement_func(use_numpy = False)

    shift_rc_model = ReverseComplementShiftWrapper(model, shift_len, reverse_complement_func, model_cls_str, feature_predict=args.feature_predict)

    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1)

    dataloader = DataLoader(dataset, batch_size=32 if args.debug else 1024,
                            num_workers=NUM_WORKERS_ENV, collate_fn=collate_fn,
                            worker_init_fn=variantparser_worker_init_fn)

    ret = trainer.predict(model=shift_rc_model, dataloaders=dataloader)

    def concat_preds(predictions):

        result = []

        for i in range(len(ret[0])):

            result.append(np.concatenate([r[i] for r in predictions]))

        return result

    ref_predictions = concat_preds(ret)
    del ret

    # switch to alt sequences
    dataset.load = 'alt'
    dataloader = DataLoader(dataset, batch_size=64 if args.debug else 1024,
                            num_workers=NUM_WORKERS_ENV, collate_fn=collate_fn,
                            worker_init_fn=variantparser_worker_init_fn)

    ret = trainer.predict(model=shift_rc_model, dataloaders=dataloader)

    alt_predictions = concat_preds(ret)
    del ret

    if model_cls_str in ['IEAquaticDilated']:
        bio_ref, bio_alt = ref_predictions[1], alt_predictions[1]
        tech_ref, tech_alt = ref_predictions[2], alt_predictions[2]
    output_idx = 0 if args.feature_predict == 'predictions' else 1

    def _get_logits(bio, tech):
        with torch.no_grad():
            return shift_rc_model.model.get_logits(
                torch.FloatTensor(bio).to(shift_rc_model.device),
                torch.FloatTensor(tech).to(shift_rc_model.device),
            )[output_idx]

    if model_cls_str in ['IEAquaticDilated']:
        ref_predictions_bio_ver2 = _get_logits(bio_ref, tech_ref)
        ref_predictions.append(ref_predictions_bio_ver2)
        # the ver.2  ref predictions are the same for bio and tech
        ref_predictions.append(ref_predictions_bio_ver2)
        alt_predictions.append(_get_logits(bio_alt, tech_ref))
        alt_predictions.append(_get_logits(bio_ref, tech_alt))

    outfile = f'{out_dir}/{out_prefix}.gnomAD_{args.set}_VEP.h5'
    dump_hdf5(ref_predictions, alt_predictions, outfile)


""" 

    counts_all = variants.label.value_counts()
    freq_all = counts_all / counts_all.sum()

    c_all = counts_all[['Rare','Common']].values

    if args.method == 'absolute':
        Q = [0.1, 0.9, 0.99, 0.999, 0.9999]
    else:
        Q = [0.001, 0.01, 0.1, 0.9, 0.99, 0.999]

    setnames = ['predictions','features_biological', 'features_technical', 'logits_biological', 'logits_technical']


    for i, _ in enumerate(ref_predictions):

        setname = setnames[i]

        print(f'processing {setname}')
        
        pref = ref_predictions[i]
        palt = alt_predictions[i]
        
        pdiff = palt - pref

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
        
        def prep_cmat(v, q):
            
            # construct contingency matrix
            # v: an array of scores
            # q: the value by which to subset the array
            
            r = np.zeros(2, dtype=int)
            c = variants.iloc[v >= q].label.value_counts()
            
            try:
                r[0] = c['Rare']
            except KeyError:
                pass
            try:
                r[1] = c['Common']
            except KeyError:
                pass
            
            
            return np.array([[c_all[1] - r[1], r[1]],[c_all[0] - r[0], r[0]]])
                
        
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
            
            mat = [ prep_cmat(max_abs, q) for q in quants ]
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
        results['Rare_freq'] = results.Rare / (results.Rare + results.Common)
        results['Rare_enrich'] = results.Rare_freq / (freq_all[['Rare']].values)
        
        outfile = f'{out_dir}/{out_prefix}.{setname}.gnomAD_{args.set}_{args.method}_results.tsv.gz'
        
        print(f'Completed. Saving results to {outfile}')
        
        results.to_csv(outfile, sep='\t', index=False)

        outfile_counts = f'{out_dir}/{out_prefix}.{setname}.gnomAD_{args.set}_{args.method}_totalcounts.tsv'
        print(f'writing total counts to {outfile_counts}')

        counts_all.to_csv(outfile_counts, sep='\t')
 """

if __name__ == '__main__':
    main()



