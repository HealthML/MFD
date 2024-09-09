"""
run this script from the root directory
- python3 src/downstreamtasks/vista_enhancer.py 'checkpoints/2023-07-20_alex_crazy_new_model_2176/epoch=2-step=322824.ckpt' 'alex_latest_model'
- python3 src/downstreamtasks/vista_enhancer.py '[pathToCheckpoint]' '[checkpointSynonym]'

every execution creates a new folder in nucleotran/reports
- nucleotran/reports/23-07-13_12-45_alex_latest_model 
- nucleotran/reports/[date]_[time]_[synonym]

view the report
- nucleotran/reports/[date]_[time]_[synonym]/report.md
"""
import argparse
import pickle
import random
import warnings
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore", category=ConvergenceWarning)

import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import models.models as src_models
from util.load_config import config
from utils.vista_enhancer_load_windowize import vista_enhancer_load_windowize
from utils.tokenize import tokenize_extract
from utils.X_to_Xembedding import X_to_Xembedding
from utils.aggregate_embeddings import aggregate_embeddings
from utils.correlation_gc_content import correlation_gc_center_window_to_bio_tech_chart, \
    correlation_gc_seq_to_bio_tech_agg_chart
from utils.regress_out_X import regress_out_X

BEDFILE_PATH = "data/external/vista/vista_enhancers.bed"
MODEL_CENTER_BIN_LEN = 128
SHIFT_LEN = 6


def regress_out_gc(embedding, gc_content_enhancer_agg):
    embedding_gc_removed, _ = regress_out_X(embedding, gc_content_enhancer_agg.reshape(-1, 1))
    return embedding_gc_removed


def generate_random_color():
    return "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def plot_pcs(x, y, color_var, title, downstream_evaluation_path):
    """
        if isinstance(color_var[0], list):  # For tissues_agg
        unique_labels = set(label for sublist in color_var for label in sublist)
        color_map = {label: generate_random_color() for label in unique_labels}

        # Create a color list matching the length of x and y
        color_list = [''] * len(x)
        for i, labels in enumerate(color_var):
            if labels:
                color_list[i] = color_map[labels[0]]  # Taking the first label as representative, adjust as needed
            else:
                color_list[i] = '#000000'  # Black for no label

        plt.scatter(x, y, c=color_list)
    """

    if isinstance(color_var[0], list):  # For tissues_agg
        unique_labels = set(label for sublist in color_var for label in sublist)
        color_map = {label: generate_random_color() for label in unique_labels}  # Assume you have a color generator

        for i, labels in enumerate(color_var):
            if labels:
                for label in labels:
                    plt.scatter(x[i], y[i], c=color_map[label], alpha=0.5)
            else:
                plt.scatter(x[i], y[i], c='#000000', alpha=0.5)  # Black for no label
    else:  # For y_agg and gc_content_agg
        plt.scatter(x, y, c=color_var)

    plt.xlabel('PC1' if 'PC1' in title else 'PC3')
    plt.ylabel('PC2' if 'PC2' in title else 'PC4')
    plt.title(title)
    plt.colorbar()
    plt.savefig(f'{downstream_evaluation_path}/{title}.png')
    plt.close()


def auc_from_model(model, X, y, model_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    thresholder = VarianceThreshold(
        0.01)  # remove features with close to 0 variance (they should be scaled by the batchnorm, but may be sparse)
    scaler = StandardScaler()  # in the documentation they recommend features on similar scales if using saga optimizer

    try:
        X_train = thresholder.fit_transform(X_train)
    except ValueError:
        return 0.5, -1
    X_test = thresholder.transform(X_test)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)
    # take the probability of the positive class as input for AUROC
    if model.classes_[0] == 0:
        y_pred = model.predict_proba(X_test)[:,
                 1]  # Returns the probability of the sample for each class in the model, where classes are ordered as they are in self.classes_
    elif model.classes_[0] == 1:
        y_pred = model.predict_proba(X_test)[:, 0]
    else:
        raise ValueError()

    auc_score = roc_auc_score(y_test, y_pred)
    num_nonzero_params = None

    if model.penalty in ('l1', 'elasticnet'):
        num_nonzero_params = np.sum(model.coef_ != 0)
    return auc_score, num_nonzero_params, y_test, y_pred


def load_wandb_run(run_id):
    try:
        project = config['wandb']['project']
        entity = config['wandb'].get('entity', config['wandb']['username'])
    except KeyError as e:
        print('KeyError while parsing config file. Make sure you have wandb configured in src/config.yaml')
        raise e
    wandb.init(project=project, entity=entity, id=run_id, resume='must', dir='logs')


def log_df_to_wandb(df, name):
    if wandb.run is not None:
        for c in df.columns:
            wandb.run.summary.update({
                f'{name}{c}': df[c].tolist(),
            })


def main(
        checkpoint_path,
        n_jobs,
):
    np.random.seed(1)
    torch.manual_seed(1)
    random.seed(1)

    checkpoint_path = Path(checkpoint_path)
    downstream_evaluation_path = checkpoint_path.parent / 'vista_enhancer'
    downstream_evaluation_path.mkdir(parents=True, exist_ok=True)

    report_path = Path(f"{downstream_evaluation_path}/report.md")
    log_path = Path(f"{downstream_evaluation_path}/log.txt")

    # define logging functions

    def report_add(text):
        with open(report_path, "a") as report_file:
            report_file.write(f"{text}\n")

    def log(text):
        print(text)
        with open(log_path, "a") as log_file:
            log_file.write(f'{text}\n')

    def status_log(message, expected_time=None):
        time_log = f" ... (ca. {expected_time})" if expected_time else ""
        log_txt = f"\n\n>>> {message}{time_log}\n\n"
        log(log_txt)

    def report_add_log(text):
        report_add(text)
        log(text)

    def plot_pcs_report_add(x, y, color_var, title):
        plot_pcs(x, y, color_var, title, downstream_evaluation_path)
        report_add(f"![{title}](./{title}.png)")

    status_log('Attempting to load the run from W&B')
    try:
        load_wandb_run(run_id=checkpoint_path.parent.name)
        status_log('Success')
    except wandb.errors.UsageError as e:
        status_log(f'Could not load the run:\n{e.message}')

    status_log("Set up new downstream evaluation")

    report_add(f"# vista_enhancer downstream evaluation\n")

    model_cls = torch.load(checkpoint_path, map_location='cpu')['hyper_parameters'].get('model_class',
                                                                                        'IEAquaticDilated')
    model_cls = getattr(src_models, model_cls)

    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    model = model_cls.load_from_checkpoint(checkpoint_path, map_location=device)
    model_input_len = int(model.h_seq_len * MODEL_CENTER_BIN_LEN)

    vista_enhancer_vars = f"""
    ### hardcoded vars
    - bedfile_path: {BEDFILE_PATH}
    - model_center_bin_len: {MODEL_CENTER_BIN_LEN}
    - shift_len: {SHIFT_LEN}
    
    ### checkpoint vars
    - checkpoint_path: {checkpoint_path}
    - model_input_len: {model_input_len}
    """

    report_add_log(f"## vista_enhancer_vars\n{vista_enhancer_vars}\n")

    status_log("Data loading, Predict feature and quantify GC-bias")
    """
    Data loading 
    • load the model 
    o what is the input sequence length?   
    o what are the dimensions of the biological / technical features? 
    • load the enhancer sequences 
    o calculate the GC-content of each enhancer 
    • set and save the random seed  
    
    Predict features and quantify GC-bias 
    • window-ize the enhancer sequences 
    o Calculate the GC-content of each "center focus window" 
    o Note: this part is actually always the same for models of the same input length, maybe it would be better to store the data 
    (windows) and load it later? 
      
    • predict the technical and biological features for all sequences ("center focus windows") with padding  
    o Calculate the correlation of non-aggregated technical/biological features with the "center focus window" GC-content 
    • Save this table (shape = (n_bio_features + n_tech_features, 1) ) 
    • aggregate the windows using the mean / max 
    o Save the features after aggregation (for examply in a .npy or torch tensor file), make sure to also save the 
    labels/enhancer names 
    • shape = (n_bio_features, n_enhancer) and (n_tech_features, n_enhancer) 
    o calculate the correlation of the aggregated technical/biological features with the enhancer GC-content (the one calculated 
    in the beginning) 
    • Save this table (shape = (n_bio_features + n_tech_features, 1) ) 
    """

    status_log("vista_enhancer_load_windowize", "1 min")
    dna_df = vista_enhancer_load_windowize(BEDFILE_PATH, model_input_len, MODEL_CENTER_BIN_LEN, SHIFT_LEN, log_func=log)
    # seq (AGCGT, ...), seq_len (2182, ...), is_enhancer (0, 1, ...), enhancer_number (0, 1, 2, ...), tissues ("hindbrain(rhombencephalon),forebrain", "other", ...), window (AGCGT.., ...), window_index (1, 2, ...)

    status_log("tokenize_extract", "1 min")
    # X: torch.Size([n_windows, dna_length, encoding_size (16)]), Xnormal/ Xreverse only half n_windows
    X, Xnormal, Xreverse = tokenize_extract(dna_df)
    y = dna_df['is_enhancer'].tolist()
    tissues = dna_df['tissues'].tolist()
    seq_len = dna_df['seq_len'].tolist()
    window_gc_content = dna_df['window_gc_content'].tolist()
    gc_content_enhancer = dna_df['gc_content_enhancer'].tolist()
    enhancer_number = dna_df['enhancer_number'].tolist()
    window_index = dna_df['window_index'].tolist()

    n_windows_check = [
        X.shape[0] / 2,
        Xnormal.shape[0],
        Xreverse.shape[0],
        len(y),
        len(tissues),
        len(seq_len),
        len(window_gc_content),
        len(enhancer_number),
        len(window_index),
    ]

    # Check if all numbers are the same
    assert all(x == n_windows_check[0] for x in n_windows_check), "check dataframe, dimensions are not the same!"
    report_add_log(f"{n_windows_check[0]} windows extracted")

    status_log("X_to_Xembedding", "1 min")
    Xembeddings, Xembeddings_biological, Xembeddings_technical = X_to_Xembedding(Xnormal, Xreverse, model, SHIFT_LEN,
                                                                                 log_func=log)

    status_log("calculate aggregated embeddings")
    Xembeddings_biological_max, y_agg, tissues_agg, enhancer_number_agg = aggregate_embeddings(Xembeddings_biological,
                                                                                               enhancer_number, y,
                                                                                               tissues, 'max')
    log(f"len(y_agg): {len(y_agg)}")
    log(f"len(tissues_agg): {len(tissues_agg)}")
    log(f"Xembeddings_biological_max.shape: {tuple(Xembeddings_biological_max.shape)}")

    Xembeddings_biological_avg, _, _, _ = aggregate_embeddings(Xembeddings_biological, enhancer_number, y, tissues,
                                                               'mean')
    log(f"Xembeddings_biological_avg.shape: {tuple(Xembeddings_biological_avg.shape)}")

    Xembeddings_technical_max, _, _, _ = aggregate_embeddings(Xembeddings_technical, enhancer_number, y, tissues, 'max')
    log(f"Xembeddings_technical_max.shape: {tuple(Xembeddings_technical_max.shape)}")

    Xembeddings_technical_avg, _, _, _ = aggregate_embeddings(Xembeddings_technical, enhancer_number, y, tissues,
                                                              'mean')
    log(f"Xembeddings_technical_avg.shape: {tuple(Xembeddings_technical_avg.shape)}")

    preds_avg, _, _, _ = aggregate_embeddings(Xembeddings, enhancer_number, y, tissues, 'mean')

    _, unique_indices = np.unique(enhancer_number, return_index=True)
    gc_content_enhancer_agg = np.array(gc_content_enhancer)[unique_indices]

    correlation_windows_path, bio_corr, tech_corr = correlation_gc_center_window_to_bio_tech_chart(
        window_index,
        window_gc_content,
        Xembeddings_biological,
        Xembeddings_technical,
        downstream_evaluation_path,
    )
    report_add(
        f"### Correlation of non-aggregated technical/biological features with the 'center focus window' GC-content\n![correlation_gc_center_window_to_bio_tech_chart](./{correlation_windows_path})\n")

    correlation_seq_path, bio_max_corr, bio_avg_corr, tech_max_corr, tech_avg_corr = correlation_gc_seq_to_bio_tech_agg_chart(
        Xembeddings_biological_max,
        Xembeddings_biological_avg,
        Xembeddings_technical_max,
        Xembeddings_technical_avg,
        gc_content_enhancer_agg,
        downstream_evaluation_path,
    )
    df_corrs = pd.DataFrame([
        bio_corr,
        tech_corr,
        bio_max_corr,
        tech_max_corr,
        bio_avg_corr,
        tech_avg_corr,
    ])
    df_corrs.loc[0, 'features_type'] = 'biological'
    df_corrs.loc[1, 'features_type'] = 'technical'
    df_corrs.loc[2, 'features_type'] = 'biological'
    df_corrs.loc[3, 'features_type'] = 'technical'
    df_corrs.loc[4, 'features_type'] = 'biological'
    df_corrs.loc[5, 'features_type'] = 'technical'
    df_corrs.loc[2, 'aggregation_type'] = 'max'
    df_corrs.loc[3, 'aggregation_type'] = 'max'
    df_corrs.loc[4, 'aggregation_type'] = 'avg'
    df_corrs.loc[5, 'aggregation_type'] = 'avg'
    df_corrs.to_csv(
        report_path.parent / 'GC_content_correlations.tsv',
        sep='\t',
        index=False,
    )
    log_df_to_wandb(df_corrs, 'downstream_vista_enhancer/GC_content_correlations/')

    report_add(
        f"### Correlation of the aggregated technical/biological features with the enhancer GC-content\n![correlation_gc_seq_to_bio_tech_agg_chart](./{correlation_seq_path})\n")

    Xembeddings_biological_max_gc_removed = regress_out_gc(Xembeddings_biological_max, gc_content_enhancer_agg)
    Xembeddings_biological_avg_gc_removed = regress_out_gc(Xembeddings_biological_avg, gc_content_enhancer_agg)
    Xembeddings_technical_max_gc_removed = regress_out_gc(Xembeddings_technical_max, gc_content_enhancer_agg)
    Xembeddings_technical_avg_gc_removed = regress_out_gc(Xembeddings_technical_avg, gc_content_enhancer_agg)
    preds_avg_gc_removed = regress_out_gc(preds_avg, gc_content_enhancer_agg)
    Xembeddings_full_max = np.concatenate([Xembeddings_biological_max, Xembeddings_technical_max], axis=1)
    Xembeddings_full_avg = np.concatenate([Xembeddings_biological_avg, Xembeddings_technical_avg], axis=1)
    Xembeddings_full_max_gc_removed = np.concatenate([
        Xembeddings_biological_max_gc_removed,
        Xembeddings_technical_max_gc_removed,
    ], axis=1)
    Xembeddings_full_avg_gc_removed = np.concatenate([
        Xembeddings_biological_avg_gc_removed,
        Xembeddings_technical_avg_gc_removed,
    ], axis=1)
    print(Xembeddings_full_max.shape)

    embeddings = [
        # ('biological_max', Xembeddings_biological_max, Xembeddings_biological_max_gc_removed),
        ('biological_avg', Xembeddings_biological_avg, Xembeddings_biological_avg_gc_removed),
        # ('technical_max', Xembeddings_technical_max, Xembeddings_technical_max_gc_removed),
        ('technical_avg', Xembeddings_technical_avg, Xembeddings_technical_avg_gc_removed),
        # ('predictions_avg', preds_avg, preds_avg_gc_removed),
        # ('full_max', Xembeddings_full_max, Xembeddings_full_max_gc_removed),
        ('full_avg', Xembeddings_full_avg, Xembeddings_full_avg_gc_removed),
    ]

    status_log("PCA")
    """
    PCA 
    • calculate PCA on the aggregated features 
    o calculate how much variance is explained by each dimension 
    • save this information 
    o how many principal components does it take to explain 90% of the variance? 
    • save this information 
    o save the PCA-results (save the python object or the table with the loadings) 
    o plot PC1 vs PC2 
    • color: active vs inactive 
    • color: GC-content 
    • color: tissues 
    o plot PC3 vs PC4 
    • color: active vs inactive 
    • color: GC-content 
    • color: tissues 
    """
    report_add(f"# PCA")

    rows_pca = []
    for name, X, _ in embeddings:
        report_add_log(f"### PCA Analysis for {name}\n")
        pca = PCA()
        X_pca = pca.fit_transform(X)

        # Calculate Variance Explained
        explained_var = pca.explained_variance_ratio_
        n_dim = min(10, len(explained_var))

        row = {f'pc_{i}': explained_var[i] for i in range(n_dim)}
        row['features_type'] = name.split('_')[0]
        row['aggregation_type'] = name.split('_')[1]
        rows_pca.append(row)
        report_add(f"{name} - Variance explained by first {n_dim} dimensions: {explained_var[:n_dim]}\n")

        # Find Components for 90% Variance:
        cum_var = np.cumsum(explained_var)
        n_90 = np.argmax(cum_var >= 0.9) + 1
        report_add(f"{name} - Number of PCs to explain 90% variance: {n_90}")

        np.save(f'{downstream_evaluation_path}/{name}_pca.npy', X_pca)

        plot_pcs_report_add(X_pca[:, 0], X_pca[:, 1], y_agg, f"{name}_PC1_PC2_active_inactive")
        plot_pcs_report_add(X_pca[:, 0], X_pca[:, 1], gc_content_enhancer_agg, f"{name}_PC1_PC2_GC_content")
        plot_pcs_report_add(X_pca[:, 0], X_pca[:, 1], tissues_agg, f"{name}_PC1_PC2_tissues")

        plot_pcs_report_add(X_pca[:, 2], X_pca[:, 3], y_agg, f"{name}_PC3_PC4_active_inactive")
        plot_pcs_report_add(X_pca[:, 2], X_pca[:, 3], gc_content_enhancer_agg, f"{name}_PC3_PC4_GC_content")
        plot_pcs_report_add(X_pca[:, 2], X_pca[:, 3], tissues_agg, f"{name}_PC3_PC4_tissues")
    df_pca = pd.DataFrame(rows_pca)
    df_pca.to_csv(
        report_path.parent / 'pca_expl_var.tsv',
        sep='\t',
        index=False,
    )
    log_df_to_wandb(df_pca, 'downstream_vista_enhancer/pca_expl_var/')

    status_log("Enhancer prediction task ")
    """
    Enhancer prediction task 
    • fit elastic net / Ridge models on the biological / technical embeddings to predict active / inactive 
    o calculate the performance 
    • save the performance 
    o for the elastic net model, check the numberr of non-zero parameters 
    o save the models / model paramters 
      
    • calculate the performance of using just GC-content 
    o save the performance 
      
    • remove the GC-content dependency from the features, you can use the function below to do this 
    o calculate the performance with GC-content removed 
    • save the performance 
    o for the elastic net model, check the number of non-zero parameters 
    o save the models/ model parameters 
    """
    report_add(f"# Enhancer prediction task")

    if args.debug:
        max_iter = 10
    else:
        max_iter = 1000
    models = [
        # ('Linear', LogisticRegressionCV(max_iter=1000)),
        ('Lasso', LogisticRegressionCV(penalty='l1', max_iter=max_iter, solver='saga', class_weight='balanced', scoring='roc_auc', n_jobs=n_jobs)),
        # ('ElasticNet', LogisticRegressionCV(penalty='elasticnet', l1_ratios=[.5], max_iter=1000, solver='saga')),
        ('Ridge', LogisticRegressionCV(penalty='l2', max_iter=max_iter, solver='saga', class_weight='balanced', scoring='roc_auc', n_jobs=n_jobs)),
    ]

    rows_auc = []
    model_predictions = []
    for model_name, model in models:
        report_add_log(f"### {model_name}")

        # GC Content Performance
        auc_score, num_nonzero_params, y_test, y_pred = auc_from_model(
            model=model,
            X=np.array(gc_content_enhancer_agg).reshape(-1, 1),
            y=y_agg,
            model_name=model_name,
        )
        rows_auc.append({
            'tissue_type': 'all',
            'model_type': model_name,
            'features_type': 'GC content',
            'AUC': auc_score,
            'num_nonzero_params': num_nonzero_params,
        })
        report_add_log(
            f"GC-content AUC: {auc_score}{f', Number of non-zero parameters: {num_nonzero_params}' if num_nonzero_params is not None else ''}")

        for name, embedding, embedding_gc_removed in embeddings:
            features_type, aggregation_type = name.split('_')

            # With GC Content
            auc_score, num_nonzero_params, y_test, y_pred = auc_from_model(
                model=model,
                X=embedding,
                y=y_agg,
                model_name=model_name,
            )
            rows_auc.append({
                'tissue_type': 'all',
                'model_type': model_name,
                'features_type': features_type,
                'aggregation_type': aggregation_type,
                'AUC': auc_score,
                'num_nonzero_params': num_nonzero_params,
            })
            model_predictions.append({
                'model_name': model_name,
                'GC_regr_out': False,
                'features_type': features_type,
                'tissue': 'all',
                'y_test': y_test,
                'y_pred': y_pred,
            })
            report_add_log(
                f"{name} AUC: {auc_score}{f', Number of non-zero parameters: {num_nonzero_params}' if num_nonzero_params is not None else ''}")

            # Without GC Content
            auc_score, num_nonzero_params, y_test, y_pred = auc_from_model(
                model=model,
                X=embedding_gc_removed,
                y=y_agg,
                model_name=model_name,
            )
            rows_auc.append({
                'tissue_type': 'all',
                'model_type': model_name,
                'features_type': features_type,
                'aggregation_type': aggregation_type,
                'AUC': auc_score,
                'num_nonzero_params': num_nonzero_params,
                'GC_regressed_out': True,
            })
            model_predictions.append({
                'model_name': model_name,
                'GC_regr_out': True,
                'features_type': features_type,
                'tissue': 'all',
                'y_test': y_test,
                'y_pred': y_pred,
            })
            report_add_log(
                f"{name} AUC (GC-content removed): {auc_score}{f', Number of non-zero parameters: {num_nonzero_params}' if num_nonzero_params is not None else ''}")

    status_log("tissue specificity prediction task")
    """
    tissue specificity prediction task 
    • fit elastic net / Ridge models on the biological / technical embeddings to predict tissue 
    o calculate the performance 
    • save the performance 
    o for the elastic net model, check the number of non-zero parameters 
    o save the models / model paramters 
      
    • calculate the performance of using just GC-content 
    o save the performance 
      
    • remove the GC-content dependency from the features, you can use the function below to do this 
    o calculate the performance with GC-content removed 
    • save the performance 
    o for the elastic net model, check the number of non-zero parameters 
    o save the models/ model parameters 
    """
    report_add(f"# tissue specificity prediction task")

    # tissues with over 50 instances:
    common_tissue_labels = ["forebrain", "midbrain(mesencephalon)", "hindbrain(rhombencephalon)", "neuraltube", "limb",
                            "heart", "branchialarch", "eye", "dorsalrootganglion"]

    for i, common_tissue_label in enumerate(common_tissue_labels):

        report_add_log(f"## {common_tissue_label}")
        y_common_tissue_exists = [1 if common_tissue_label in t_agg else 0 for t_agg in tissues_agg]

        if args.debug:
            if i > 0:
                report_add_log(f"skipping because --debug was set.")
                continue

        for model_name, model in models:
            report_add_log(f"### {model_name}")

            # GC Content
            auc_score, num_nonzero_params, y_test, y_pred = auc_from_model(
                model=model,
                X=np.array(gc_content_enhancer_agg).reshape(-1, 1),
                y=y_common_tissue_exists, model_name=model_name
            )
            rows_auc.append({
                'tissue_type': common_tissue_label,
                'model_type': model_name,
                'features_type': 'GC content',
                'AUC': auc_score,
                'num_nonzero_params': num_nonzero_params,
            })
            report_add_log(
                f"GC-content AUC: {auc_score}{f', Number of non-zero parameters: {num_nonzero_params}' if num_nonzero_params is not None else ''}")

            # Embeddings
            for name, embedding, embedding_gc_removed in embeddings:
                features_type, aggregation_type = name.split('_')

                auc_score, num_nonzero_params, y_test, y_pred = auc_from_model(
                    model=model, X=embedding,
                    y=y_common_tissue_exists,
                    model_name=model_name,
                )
                rows_auc.append({
                    'tissue_type': common_tissue_label,
                    'model_type': model_name,
                    'features_type': features_type,
                    'aggregation_type': aggregation_type,
                    'AUC': auc_score,
                    'num_nonzero_params': num_nonzero_params,
                })
                model_predictions.append({
                    'model_name': model_name,
                    'GC_regr_out': False,
                    'features_type': features_type,
                    'tissue': common_tissue_label,
                    'y_test': y_test,
                    'y_pred': y_pred,
                })
                report_add_log(
                    f"{name} AUC: {auc_score}{f', Number of non-zero parameters: {num_nonzero_params}' if num_nonzero_params is not None else ''}")

                auc_score, num_nonzero_params, y_test, y_pred = auc_from_model(
                    model=model,
                    X=embedding_gc_removed,
                    y=y_common_tissue_exists,
                    model_name=model_name,
                )
                rows_auc.append({
                    'tissue_type': common_tissue_label,
                    'model_type': model_name,
                    'features_type': features_type,
                    'aggregation_type': aggregation_type,
                    'AUC': auc_score,
                    'num_nonzero_params': num_nonzero_params,
                    'GC_regressed_out': True,
                })
                model_predictions.append({
                    'model_name': model_name,
                    'GC_regr_out': True,
                    'features_type': features_type,
                    'tissue': common_tissue_label,
                    'y_test': y_test,
                    'y_pred': y_pred,
                })
                report_add_log(
                    f"{name} AUC (GC-content removed): {auc_score}{f', Number of non-zero parameters: {num_nonzero_params}' if num_nonzero_params is not None else ''}")
    df_auc = pd.DataFrame(rows_auc)
    df_auc['GC_regressed_out'].fillna(False, inplace=True)
    df_auc.to_csv(
        report_path.parent / 'auc.tsv',
        sep='\t',
        index=False,
    )
    log_df_to_wandb(df_auc, 'downstream_vista_enhancer/auc/')

    with open(report_path.parent / 'vista_saved_predictions.pkl', 'wb') as outfile:
        pickle.dump(model_predictions, outfile)

    status_log("Report finished")
    status_log(str(report_path.absolute()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt',
        type=str,
        required=True,
    )
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(checkpoint_path=args.ckpt, n_jobs=args.n_jobs)
