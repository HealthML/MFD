"""
This script evaluates disentangled models not the downstream fantom enhacer prediction.
Enhancer sequences were taken from fantom5 project: https://fantom.gsc.riken.jp/5/datafiles/reprocessed/hg38_latest/extra/enhancer/
Binary usage matrix was used for labels for enhancers from chromosomes 9 and 10
Filtered for sample IDs from experiments ~ matching ENCODE and having more than 30 positive sequences

All necessary info with labels etc is in data/external/fantom/fantom5_enhancers.csv
Sample IDs and thoer mapping can be found in data/external/fantom/fantom5_tissue_labels_selected_enh.csv

run fron root
"""
import os
# os.chdir('../../')
# sys.path.append('./src')
# os.getcwd()
import pandas as pd
import numpy as np

import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import pickle
import argparse

from copy import deepcopy
from sklearn.feature_selection import VarianceThreshold

from downstreamtasks.utils.regress_out_X import regress_out_X
import models.models as src_models
from features.label import LabelLoader
from features.nucleotide import BEDSeqLoader, DNATokenizer

from torch import tensor
from torch.utils.data import DataLoader, Dataset
from torch.nn import Embedding
from torch import permute
from util.load_config import config

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.preprocessing import StandardScaler

from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=ConvergenceWarning)

p = argparse.ArgumentParser()
p.add_argument('--ckpt', required=True, help='path to the checkpoint file')
p.add_argument('--out', required=True, help='output directory')
p.add_argument('--n_jobs', required=False, default=1, type=int)
p.add_argument('--debug', action='store_true')
args = p.parse_args()
model_path = args.ckpt

max_iter = 1000  # Note: this makes things slow...
if args.debug:
    print('!!! --debug active !!!\n')
    max_iter = 10

bed_path = 'data/external/fantom/fantom5_enhancers_hg38.bed'
ref_path = config['reference']['GRCh38']
label_IDs_path = 'data/external/fantom/fantom5_tissue_labels_selected_enh.csv'

out_dir_path = args.out
if not out_dir_path.endswith('/'):
    out_dir_path += '/'

# Check if the directory exists, and create it if not
if not os.path.exists(out_dir_path):
    os.makedirs(out_dir_path)


# Fantom specific data loading and prediction

class FantomLabelLoader(LabelLoader):
    def __init__(self, bed_path):
        "Loads labels corresponding to the regions from fantom5 bed file with labels in it"

        with open(bed_path, 'r') as infile:
            for i, _ in enumerate(infile):
                pass
            bed_len = i + 1
            print(f'BED-file contains {bed_len} regions.')

        labels = pd.read_csv(bed_path, sep='\t',
                             header=None, )
        self.labels = np.array(labels.iloc[:, 4:len(labels)])
        # print('{:.3f}% of regions have at least 1 label.'.format(int here) / len(self.labels) * 100))

        self.label_ids = range(len(self.labels))
        self.n_labels = len(self.label_ids)

    def __len__(self):
        return len(self.label_ids)

    def __getitem__(self, i):
        return self.labels[i]

    def get_labels(self, i):
        return self.labels[i]

    def get_label_ids(self):
        '''
        Return the label identifiers 
        '''
        return self.label_ids


def calculate_gc_content(seq):
    assert len(seq) == 1
    seq = np.array(list(seq[0]))
    g_count = np.count_nonzero(seq == 'G')
    c_count = np.count_nonzero(seq == 'C')
    total_count = len(seq)
    return (g_count + c_count) / total_count * 100


class Fantom5Dataset(Dataset):
    def __init__(self,
                 bed_path: str,
                 ref_path: str,
                 labelfile_path: str,
                 seq_order: int,
                 seq_len: int,
                 stride=1,
                 target_transform=tensor,
                 reverse_complement=False,
                 random_reverse_complement=False,
                 random_shift=0,
                 one_hot=True,
                 dtype=torch.float32
                 ):

        self.seq_len = seq_len
        self.seq_order = seq_order
        self.dtype = dtype
        self.bedseqloader = BEDSeqLoader(bed_path, ref_path, random_shift=random_shift, ignore_strand=True)
        self.bedseqloader.resize(self.seq_len + (self.seq_order - 1))
        self.labelloader = FantomLabelLoader('data/external/fantom/fantom5_enhancers_hg38.bed')

        # sanity check
        assert len(self.bedseqloader) == len(self.labelloader)

        # self.transform = transform
        self.target_transform = target_transform

        self.tokenizer = DNATokenizer(seq_order=seq_order, allow_N=True, stride=stride)

        self.reverse_complement = reverse_complement
        self.random_reverse_complement = random_reverse_complement
        self.random_shift = random_shift

        # One-hot encoding of DNA-sequences
        if one_hot:
            # dna_embed will help us convert the token-representation to one-hot representation
            W, mapping = self.tokenizer.get_one_hot_weights_matrix(N_max=0)
            dna_embed = Embedding.from_pretrained(tensor(W), freeze=True)
            self.transform = lambda x: permute(dna_embed(tensor(x)), [0, 2, 1])
        else:
            self.transform = tensor

        if reverse_complement & random_reverse_complement:
            print(
                'Warning, both "reverse_complement" and "random_reverse_complement" are True. Will serve up reverse-complement sequences randomly.')
            self.reverse_complement = False

        self._augment = True

    def augment_on(self):
        self._augment = True
        self.bedseqloader.random_shift = self.random_shift

    def augment_off(self):
        self._augment = False
        self.bedseqloader.random_shift = 0

    def __len__(self):
        if args.debug:
            return 60
        return len(self.bedseqloader)

    def __getitem__(self, idx):
        dnaseq_raw = self.bedseqloader.get_seq(idx)
        label = self.labelloader.get_labels(idx)

        dnaseq = self.tokenizer.tokenize(dnaseq_raw,
                                         reverse_complement=self.reverse_complement,
                                         random=self.random_reverse_complement & self._augment)

        if self.transform:
            dnaseq = self.transform(dnaseq)

        if self.target_transform:
            label = self.target_transform(label)

        gc_content = calculate_gc_content(dnaseq_raw)

        return dnaseq.to(self.dtype), label.to(self.dtype), torch.scalar_tensor(gc_content, dtype=self.dtype)

    def train_test_split(self, test, train=None):
        """
         Returns a train-test split 80/20

         """
        dataset_size = len(self.labelloader)
        indices = list(range(dataset_size))
        split = int(np.floor(test * dataset_size))

        np.random.seed(42)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        return train_indices, val_indices


def collate_fn(batch):
    x = torch.cat([x for x, _, _ in batch], dim=0)
    y = torch.stack([y for _, y, _ in batch], dim=0)
    gc = torch.stack([gc for _, _, gc in batch], dim=0)
    return x, y, gc


def get_embeddings(model, dataloader):
    ret = trainer.predict(
        model=model,
        dataloaders=dataloader,
    )

    # Extract and concatenate features
    predictions, features_biological, features_technical, y, gc_content = (
        torch.cat([batch[output_idx] for batch in ret], dim=0)
        for output_idx in range(5)
    )
    predictions_bio, predictions_tech = predictions, predictions
    if isinstance(model, src_models.IEAquaticDilated):
        _, _, logits_biological, logits_technical = model.get_logits(
            features_biological=features_biological,
            features_technical=features_technical,
        )
        predictions_bio = src_models.sigmoid(logits_biological + model.bias)
        predictions_tech = src_models.sigmoid(logits_technical + model.bias)

    return predictions, predictions_bio, predictions_tech, features_biological, features_technical, y, gc_content


class ReverseComplementShiftWrapper(pl.LightningModule):
    """
    Symmetric cropping along position

        crop: number of positions to crop on either side
    """

    def __init__(self, model, n_shift):
        super().__init__()
        self.model = model
        self.n_shift = n_shift - 1

    def forward(self, x):
        forward = x
        reverse_complement = reverse_complement_func(forward)

        predictions_l = []
        features_biological_l = []
        features_technical_l = []

        for i in range(self.n_shift):
            for s in [forward, reverse_complement]:
                input_crop = s[:, :, i: -self.n_shift + i]
                predictions, _, features_biological, features_technical = self.model.forward(input_crop,
                                                                                             return_features=True)
                predictions_l.append(predictions)
                features_biological_l.append(features_biological)
                features_technical_l.append(features_technical)

        stack_mean = lambda li: torch.mean(torch.stack(li), dim=0)
        output = stack_mean(predictions_l), stack_mean(features_biological_l), stack_mean(features_technical_l)
        return output

    def predict_step(self, batch, batch_idx):
        x, y, gc = batch
        pred, bio, tech = self.forward(x)
        return pred, bio, tech, y, gc


## Load model, get features
# trainer = pl.Trainer(accelerator='gpu', devices=1)
trainer = pl.Trainer(accelerator='auto', devices=1)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_cls = torch.load(model_path, map_location='cpu')[
    'hyper_parameters'].get('model_class', 'IEAquaticDilated')
model_cls = getattr(src_models, model_cls)
model = model_cls.load_from_checkpoint(model_path, map_location=device)

shift_len = 6
model_input_len = int(model.hparams.seq_len)
dataset = Fantom5Dataset(bed_path=bed_path,
                         ref_path=ref_path,
                         labelfile_path=bed_path,
                         seq_order=model.hparams.seq_order,
                         seq_len=model_input_len + shift_len,
                         reverse_complement=False,
                         random_reverse_complement=False,
                         random_shift=0
                         )
# =========== Get features and predictions from the model =============

dataloader = DataLoader(dataset, batch_size=32, num_workers=1, collate_fn=collate_fn, shuffle=False)
reverse_complement_func = dataset.tokenizer.onehot_reverse_complement_func(use_numpy=False)
shift_rc_model = ReverseComplementShiftWrapper(model, shift_len)
(disent_predictions, predictions_bio, predictions_tech, features_biological, features_technical,
 y_true, gc_content) = get_embeddings(shift_rc_model, dataloader)

# ==== Downstream classification ====

# _____Map same tissue samples to one class_____
mapping = [
    ("adipose", [3, 4, 5, 6, 7]),
    # ("adrenal gland", [8]),
    ("aorta", [9]),
    ("brain", [0, 1, 2, 10, 11, 12, 25, 26, 29]),
    ("colon", [13, 14, 15]),
    ("heart", [16, 17, 18, 19, 20]),
    ("kidney", [21, 22]),
    ("liver", [23, 24]),
    ("ovary", [27]),
    ("pancreas", [28]),
    ("skeletal muscule", [30, 31, 32]),
    ("spleen", [33, 34]),
    ("testis", [35, 36]),
    ("uterus", [37, 38])
]
new_class_names = []
for term in mapping:
    idx = term[1]
    new_class_names.append(term[0])

# _____Create new y_true with combined classes of the same tissue (e.g. 2 brain classes and cortex into 1)_____

new_y_true = []
for tissue_sample in mapping:
    name = tissue_sample[0]
    idx = tissue_sample[1]
    previous_labels = y_true[:, idx]
    summ = previous_labels.sum(axis=1) > 0
    new_labels = [int(statement) for statement in summ]
    new_y_true.append(new_labels)

new_y_true = np.array(new_y_true).T

# _____Fit classifiers OvR_____
embeddings = [
    ('bio', features_biological),
    ('tech', features_technical),
    ('full', torch.cat((features_biological, features_technical), axis=1)),
]


def regress_out_gc(embedding, gc_content_enhancer_agg):
    gc_content_enhancer_agg = gc_content_enhancer_agg.detach().cpu().numpy()
    embedding_gc_removed, _ = regress_out_X(embedding, gc_content_enhancer_agg.reshape(-1, 1))
    return embedding_gc_removed


# add features with GC content regressed-out
embeddings = embeddings + [(f'{emb[0]}_GC_regressed_out', regress_out_gc(emb[1], gc_content)) for emb in embeddings]

models = [
    ('Lasso',
     LogisticRegressionCV(
         penalty='l1', max_iter=max_iter, solver='saga', class_weight='balanced', n_jobs=args.n_jobs)),
    # ('ElasticNet', LogisticRegressionCV(penalty='elasticnet', l1_ratios=[.5], max_iter=max_iter, solver='saga', class_weight='balanced')), # save time by skipping this
    ('Ridge',
     LogisticRegressionCV(
         penalty='l2', max_iter=max_iter, scoring='roc_auc', class_weight='balanced', n_jobs=args.n_jobs)),
]
model_names = [models[i][0] for i in range(len(models))]
models_to_save = []
evaluations = []
model_predictions = []
for i, (model_name, model) in enumerate(models):
    print("========== MODEL ", model_name, "==========")
    for c in range(new_y_true.shape[1]):

        evaluation = {}

        if args.debug:
            # only fit the first two tissues when debugging
            if c > 0:
                print(f'skipping {c}...')
                continue
        print("========== CLASS ", c, new_class_names[c], "==========")

        evaluation['model'] = model_name
        evaluation['class'] = c
        evaluation['tissue_sample'] = new_class_names[c]
        evaluation['n_samples'] = int(sum(new_y_true[:, c]))

        for name, embedding in embeddings:
            X_train, X_test, y_train, y_test = train_test_split(
                embedding, new_y_true[:, c], test_size=0.2, random_state=0, stratify=new_y_true[:, c])

            # remove features with close to 0 variance (they should be scaled by the batchnorm, but may be sparse)
            thresholder = VarianceThreshold(0.01)
            try:
                X_train = thresholder.fit_transform(X_train)
                X_test = thresholder.transform(X_test)

                # in the documentation they recommend features on similar scales if using saga optimizer
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                model.fit(X_train, y_train)
                if model.classes_[0] == 0:
                    # Returns the probability of the sample for each class in the model,
                    # where classes are ordered as they are in self.classes_
                    y_pred = model.predict_proba(X_test)[:, 1]
                elif model.classes_[0] == 1:
                    y_pred = model.predict_proba(X_test)[:, 0]
                # eval
                roc_auc = roc_auc_score(y_test, y_pred)
                precision, recall, _ = precision_recall_curve(y_test, y_pred)
                pr_auc = auc(recall, precision)

                # save predictions
                model_predictions.append({
                    'model_name': model_name,
                    'tissue': new_class_names[c],
                    'name': name,
                    'y_pred': y_pred,
                    'y_test': y_test,
                })
                # save the model
                models_to_save.append((model_name, new_class_names[c], name, deepcopy(model)))

            except ValueError:
                pr_auc = 0
                roc_auc = .5

            evaluation[f'AUPRC_{name}'] = pr_auc
            evaluation[f'AUROC_{name}'] = roc_auc

        evaluations.append(evaluation)

with open(f'{out_dir_path}/fantom_saved_models.pkl', 'wb') as outfile:
    pickle.dump(models_to_save, outfile)
with open(f'{out_dir_path}/fantom_saved_predictions.pkl', 'wb') as outfile:
    pickle.dump(model_predictions, outfile)

evals_df = pd.DataFrame.from_records(evaluations)
evals_df.to_csv(f'{out_dir_path}/fantom_AUCs.csv', sep='\t')

# plot
# Extract model type and category from column names
df_melted = pd.melt(evals_df, id_vars=['model', 'class', 'tissue_sample', 'n_samples'], var_name='metric_Category',
                    value_name='Value')
df_melted['metric'] = df_melted.metric_Category.str.split('_', expand=True)[0]
df_melted['Category'] = df_melted.metric_Category.str.split('_', expand=True)[1]
df_melted = df_melted.loc[df_melted.Category.isin(['bio', 'tech', 'full'])]
df_melted = df_melted.loc[df_melted.metric == 'AUROC']

# Set up colors based on conditions
colors = {'bio': 'green', 'tech': 'blue', 'full': 'orange'}
labels = {'bio': 'bio', 'tech': 'tech', 'full': 'full'}
markers_legend = {'bio': 4, 'tech': 5, 'full': 'x'}
markers = {'bio': 4, 'tech': 5, 'full': 'x'}

# Create subplots for each type of model
if len(model_names) in [4, 3]:
    nrows = 2
elif len(model_names) in [1, 2]:
    nrows = 1
fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 9), sharey=True)
fig.suptitle('ROC AUCs of Model Results by Tissue and Model Type', fontsize=16)

legend_handles = []

for i, model_type in enumerate(model_names):
    model_df = df_melted[df_melted['model'] == model_type]
    for idx, row in model_df.iterrows():
        if nrows == 1:
            handle = axes[i].scatter(
                row['tissue_sample'],
                row['Value'],
                s=row['n_samples'],
                color=colors[row['Category']],
                marker=markers[row['Category']],
                alpha=0.5,
                edgecolors='black'
            )
            ticks = list(range(len(model_df['tissue_sample'].unique())))
            axes[i].set_xticks(ticks)
            axes[i].set_xticklabels(model_df['tissue_sample'].unique(), rotation=45, ha='right', size=8)
        else:
            # Use axes[i, j] when nrows is greater than 1
            handle = axes[i // 2, i % 2].scatter(
                row['tissue_sample'],
                row['Value'],
                s=row['n_samples'] * 0.5,
                color=colors[row['Category']],
                marker=markers[row['Category']],
                alpha=0.5,
                # edgecolors='black'
            )
            ticks = list(range(len(model_df['tissue_sample'].unique())))
            axes[i // 2, i % 2].set_xticks(ticks)
            axes[i // 2, i % 2].set_xticklabels(model_df['tissue_sample'].unique(), rotation=45, ha='right', size=8)

    if nrows == 1:
        axes[i].set_title(f'{model_type} Model')

    else:
        axes[i // 2, i % 2].set_title(f'{model_type} Model')

legend_handles.append(Line2D([0], [0], linestyle='', marker=markers_legend['bio'], color=colors['bio'], label='bio'))
legend_handles.append(Line2D([0], [0], linestyle='', marker=markers_legend['tech'], color=colors['tech'], label='tech'))
legend_handles.append(Line2D([0], [0], linestyle='', marker=markers_legend['full'], color=colors['full'], label='full'))
plt.legend(handles=legend_handles, title='Features')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.ylim((0.45, 1.))

# Show the plot
plt.savefig(f'{out_dir_path}/fantom_per_tissue_AUCs.png')

auc_cols = list(c for c in evals_df.columns if c.startswith('AU'))
mean_results = evals_df.groupby(['model'])[auc_cols].mean()

# ==== Save markdown ====
with open(f'{out_dir_path}/summary_fantom.md', 'w') as f:
    f.write('# Downstream evaluation on Fantom5 enhancers\n\n')

    f.write('## One-by-one classifications of every tissue class from different features \n')
    # f.write(evals_df.to_csv(index=False))
    f.write(evals_df.to_markdown(buf=None, mode='w', index=False))
    f.write('\n\n')
    # f.write(f"![Failed to load the plot](fantom_per_tissue_AUCs.png)")
    # f.write('\n\n')

    f.write('Mean ROC AUC of all tissues by feature type are: \n')
    f.write(mean_results.to_markdown(buf=None, mode='w', index=True))
    f.write('\n')

    # f.write('\n\n')
    # f.write("## One-by-one classifications of every tissue class from model's predictions \n")
    # f.write('```csv\n')
    # f.write(evals_df_on_preds.to_markdown(buf=None, mode='w', index=False))
    # f.write(f"\n### Mean ROC AUC of all tissues by feature type is: {mean_roc}\n")
    # f.write(f"### Mean PR AUC of all tissues by feature type is: {mean_pr}\n")
    # f.write('```\n\n')
