import os
import torch
import pytorch_lightning as pl
import numpy as np

from models import models
from util.load_config import config
import itertools
import pandas as pd
from torch.utils.data import DataLoader
import argparse
from downstreamtasks.utils.vep import DFVariantDataset
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from scipy.stats import fisher_exact
import gzip
from torch import sigmoid
import models.models as src_models


def get_args():
    default_pv_thresh = 1e-5
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',help='checkpoint file',required=True)
    p.add_argument('--pvalue_thresh',help=f'p-value threshold used to filter features (default: {default_pv_thresh})', type=float, default=default_pv_thresh)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--model_cls', type=str, default='IEAquaticDilated')
    p.add_argument('--feature_predict', choices=['logits','predictions'], default='logits', help='for the feature-specific outputs: whether to store logits ("logits") or sigmoid(logit+bias) ("predictions")')
    args = p.parse_args()
    return args



class ReverseComplementShiftWrapper(pl.LightningModule):

    # TODO: this thing is now used in multiple scripts, maybe move it somewhere where it can be re-used instead...

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


if __name__ == '__main__':


    args = get_args()

    # TODO: as these are relatively few variants, we probably don't have to run predictions on gpu
    # this code will not work if we are running on cpu though..
    model_cls_str = torch.load(args.ckpt, map_location='cpu')[
        'hyper_parameters'].get('model_class', 'IEAquaticDilated')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_cls = getattr(src_models, model_cls_str)
    model = model_cls.load_from_checkpoint(args.ckpt, map_location=device).eval()
    model_input_len = int(model.hparams.seq_len) 

    shift_len = 12

    # load the positive set
    variant_df = pd.read_csv('data/external/eQTL_catalog/highPIP_variants_agg.tsv.gz', sep='\t')
    variant_df.rename(columns = {'variant':'id','alt':'alts'}, inplace=True)
    variant_df.drop(columns=['Unnamed: 0'], inplace=True)
    # keep only SNPs and short indels for now
    variant_df['too_long'] = (variant_df.alts.str.len() > 1) | (variant_df.ref.str.len() > 1)
    excluded_lead_variants = variant_df.query('too_long').id.values
    variant_df = variant_df.query('too_long == False').copy()

    # load the negative set
    nvariant_df = pd.read_csv('data/external/eQTL_catalog/lowPIP_variants_agg.tsv.gz', sep='\t')
    nvariant_df.rename(columns = {'variant':'id','alt':'alts'}, inplace=True)
    nvariant_df.drop(columns=['Unnamed: 0'], inplace=True)
    # keep only SNPs and short indels for now
    nvariant_df['too_long'] = (nvariant_df.alts.str.len() > 1) | (nvariant_df.ref.str.len() > 1)
    excluded_lead_variants_2 = nvariant_df.query('too_long').lead_variants
    excluded_lead_variants_2 = np.unique(np.concatenate(excluded_lead_variants_2.str.split(',').values))
    nvariant_df = nvariant_df.query('too_long == False').copy()
    variant_df = variant_df[~variant_df.id.isin(excluded_lead_variants_2)].copy()
    # this code is a bit complicated but it makes sure each positive has a matched negative
    # negatives can match multiple positives...
    lv = nvariant_df.lead_variants.str.split(',')
    select = pd.Series(index = np.repeat(nvariant_df.id.values, lv.apply(len)), data=list(itertools.chain.from_iterable(lv))).isin(variant_df.id)
    select = select[select == True]
    nvariant_df = nvariant_df[nvariant_df.id.isin(select.index)]

    # merge positives and negatives
    df1 = variant_df[['chrom','pos','ref','alts','id','tissues_selected']].copy()
    df1['eQTL'] = 1
    df2 = nvariant_df[['chrom','pos','ref','alts','id','tissues_selected']].copy()
    df2['eQTL'] = 0
    combined_df = pd.concat([df1,df2])
    combined_df.reset_index(inplace=True, drop=True)
    # get rid of chr-prefix
    combined_df['chrom'] = combined_df.chrom.str.replace('chr','')
    combined_df['chrom'] = combined_df.chrom.astype(str)

    if args.debug:
        combined_df = combined_df.sample(n=96)

    # initialize the Dataset
    dataset = DFVariantDataset(ref_fa_path = config['reference']['GRCh38'],
                            variant_df = combined_df,
                            seq_order = model.hparams.seq_order,
                            seq_len = model_input_len + shift_len,
                            load = 'ref'
                            )

    def collate_fn(batch):
        batch = torch.cat([x for x in batch], dim=0)
        return batch

    dataloader = DataLoader(dataset, batch_size=32 if args.debug else 512, num_workers=1, collate_fn=collate_fn)

    reverse_complement_func = dataset.tokenizer.onehot_reverse_complement_func(use_numpy = False)

    shift_rc_model = ReverseComplementShiftWrapper(model=model,
                                                   n_shift=shift_len,
                                                   reverse_complement_func=reverse_complement_func,
                                                   model_cls = model_cls_str,
                                                   feature_predict=args.feature_predict
                                                   )


    # predict reference sequences
    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1)
    ret = trainer.predict(model=shift_rc_model, dataloaders=dataloader)

    def concat_preds(predictions):

        result = []
        
        for i in range(len(ret[0])):
            
            result.append(np.concatenate([r[i] for r in predictions]))
            
        return result
    
    ref_predictions = concat_preds(ret)
    del ret

    # switch to alt predictions
    dataset.load = 'alt'
    dataloader = DataLoader(dataset, batch_size=32 if args.debug else 512, num_workers=1, collate_fn=collate_fn)

    # predict alternative sequences
    ret = trainer.predict(shift_rc_model, dataloaders=dataloader)
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

    def calculate_diff_scores(ref, alt):
        
        diff_scores = alt - ref
        if isinstance(diff_scores, torch.Tensor):
            diff_scores = diff_scores.numpy()
        
        return(diff_scores)


    if model_cls_str in ['IEAquaticDilated']:
        setnames = [
            'predictions',
            'features_biological', 'features_technical',
            f'{args.feature_predict}_biological', f'{args.feature_predict}_technical',
            f'{args.feature_predict}_biological_ver2', f'{args.feature_predict}_technical_ver2',
        ]
    else:
        setnames = ['predictions', 'logits']

    for i, _ in enumerate(ref_predictions):
        setname = setnames[i]

        var_test = dataloader.dataset.data.variant_df.copy().reset_index()

        diff_scores = calculate_diff_scores(ref_predictions[i], alt_predictions[i])
        diff_scores_test = diff_scores[var_test.index.values]

        results = list()

        # quantiles to analyse 
        # note: which quantiles are used for export is defined further below
        Q = [0.9, 0.95, 0.99]

        def prep_cmat(x,y,q):
            
            # prepare the matrix for the fisher's exact test based on quantile q, score x, and labels y
            q_val = np.quantile(x, q)
            select_mask = x >= q_val
            a_total = y.sum()
            b_total =  len(y) - a_total
            a_select = y[select_mask].sum()
            b_select = select_mask.sum() - a_select
            a_remain = a_total - a_select
            b_remain = b_total - b_select
            
            return np.array([[b_remain, b_select],[a_remain, a_select]]), q_val
            

        for j in range(diff_scores_test.shape[1]):
            
            # for every feature j
            if args.debug:
                if j > 9:
                    continue
            
            result_dict = {}
            
            x = np.abs(diff_scores_test[:,j]) # use absolute value for VEPs
            mu = x.mean()
            s = x.std()

            if s == 0:
                # feature has standard deviation 0
                result_dict['mean_diff'] = mu
                result_dict['sd_diff'] = s
                for variable in ['coef','se','stat','pval','ci_low','ci_high','pseudo_r2','converged','N','N_pos','N_neg','AUROC']:
                    result_dict[variable] = np.nan
                result_dict['converged'] = False
                for q in Q:
                    result_dict[f'cut_q{q}'] = np.nan
                    result_dict[f'N_pos_q{q}'] = np.nan
                    result_dict[f'N_neg_q{q}'] = np.nan
                    result_dict[f'OR_q{q}'] = np.nan
                    result_dict[f'pfisher_q{q}'] = np.nan
                results.append(result_dict)
                continue

            x = x / s # scale by the standard deviation

            # Fit the logistic regression model.
            X = sm.add_constant(x)
            y = var_test['eQTL']
            m = sm.Logit(y, X).fit(disp=False)
            # logistic regression stats
            result_dict['mean_diff'] = mu
            result_dict['sd_diff'] = s
            result_dict['coef'] = m.params['x1']
            result_dict['se'] = m.bse['x1']
            result_dict['stat'] = m.tvalues['x1']
            result_dict['pval'] = m.pvalues['x1']
            result_dict['ci_low'], result_dict['ci_high'] = m.conf_int(alpha=0.05).loc['x1']
            result_dict['pseudo_r2'] = m.prsquared
            # result_dict['converged'] = m.converged
            result_dict['N'] = m.nobs
            result_dict['N_pos'] = sum(y)
            result_dict['N_neg'] = result_dict['N'] - result_dict['N_pos']
            
            # calculate the zero-shot AURROC
            result_dict['AUROC'] =  roc_auc_score(y, x)
            
            for q in Q:
                # calculate the OR and Fisher exact test p-values at higher quantiles
                mat, q_val = prep_cmat(x,y,q)
                stat, pval = fisher_exact(mat)
                count = mat[:,1]
                
                result_dict[f'cut_q{q}'] = q_val
                result_dict[f'N_pos_q{q}'] = count[1]
                result_dict[f'N_neg_q{q}'] = count[0]
                result_dict[f'OR_q{q}'] = stat
                result_dict[f'pfisher_q{q}'] = pval
                
            results.append(result_dict)

        results = pd.DataFrame(results)

        out_dir = os.path.dirname(args.ckpt)
        out_prefix = os.path.basename(args.ckpt).replace('.ckpt','')

        out_file = f'{out_dir}/{out_prefix}.gtex_eqtl_VEP.{setname}.results.tsv.gz'

        print(f'writing results for "{setname}" to {out_file}')
        results.to_csv(out_file, sep='\t', index_label = 'idx')

        # the quantiles exported to variant files
        variants_export_quantiles = ['q0.9','q0.95','q0.99']
        EQ = [float(x.replace('q','')) for x in variants_export_quantiles]

        for iq, eq in enumerate(variants_export_quantiles):

            q = EQ[iq]

            # select features that enrich for eQTL variatns at pvalue_thresh and with OR > 1
            quantiles = results[ (results[f'pfisher_{eq}'] < args.pvalue_thresh) & (results[f'OR_{eq}'] > 1) ]

            out_file = f'{out_dir}/{out_prefix}.gtex_eqtl_VEP.{setname}.results.{eq.replace(".","_")}_variants.tsv.gz'

            print(f'exporting variants for quantile {eq} to {out_file}')

            if quantiles.shape[0] == 0:
                print(f'Warning: No features enriched for eQTL variants at quantile {eq}, p-value threshold {args.pvalue_thresh}. Results table will be empty.')
                with gzip.open(out_file, 'wt') as outfile:
                    outfile.write('varID\tN\ti_select\n')
                continue

            print(f'selected {quantiles.shape[0]} features.')

            # subset the variant effect predictions to those with significant enrichment of eQTL variants
            vep_select = diff_scores_test[:,quantiles.index.values]
            bool_matrix = np.apply_along_axis(arr=vep_select, axis=0, func1d=lambda x: x >= np.quantile(x, q))

            # this is a vector containing how many times the variant was selected
            n_select = bool_matrix.sum(axis=1)

            # subset to those selected at least once
            selected_variants = var_test.loc[n_select > 0,:].copy()
            print(f'selected {selected_variants.shape[0]} variants at quantile {eq} from {quantiles.shape[0]} features. {selected_variants.eQTL.sum()} are finemapped eQTL variants')

            selected_variants['N'] = n_select[n_select > 0]
            bool_matrix = bool_matrix[n_select > 0,:]

            # store which features selected each variant
            idx_str = quantiles.index.astype('str').values
            selected_variants['i_select'] = [ ','.join(idx_str[bool_matrix[i,:]]) for i in range(bool_matrix.shape[0])] # has to be done this way to avoid a type casting issue that truncates the resulting strings in numpy
            selected_variants.rename(columns={'id':'varID'}, inplace=True)

            selected_variants.drop(columns='index', inplace=True)
            selected_variants[['varID','N','i_select']].to_csv(out_file, sep='\t', index=False)






























# df_raw.loc[:,'OR'] = results['OR_q0.9'].values
# 
# df_raw
# 
# fisher_test[0][0].pvalue
# 
# fisher_test[0][0].statistic
# 
# counts[0]
# 
# len(diff_scores_test) * 0.01
# 
# 
# 
# 
# 
# pval = [ m.pvalues['x1'] for m in models ]
# 
# coef = [ m.params['x1'] for m in models]
# 
# pval = np.array(pval)
# 
# 
# 
# df_raw['coef'] = coef
# 
# df_raw['roc'] = np.array(rocs)
# 
# df_raw['stdev'] = np.array(stdev)
# 
# df_raw.loc[df_raw.proc_Assay_lvl1 == 'DNase-seq','proc_target'] = 'DNase-seq'
# df_raw.loc[df_raw.proc_Assay_lvl1 == 'ATAC-seq','proc_target'] = 'ATAC-seq'
# 
# import seaborn as sns
# 
# sns.boxplot(data=df_raw, y='proc_target', x='OR',orient='h')
# 
# sns.boxplot(data=df_raw, y='proc_target', x='stdev',orient='h')
# 
# sns.boxplot(data=df_raw, y='proc_target', x='coef',orient='h')
# 
# sns.boxplot(data=df_raw, y='proc_target', x='roc', orient = 'h')


