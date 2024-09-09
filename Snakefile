configfile: "src/config.yaml"

import pandas as pd
import sys
import os
from glob import glob
from snakemake.io import glob_wildcards

# used to turn on/off debugging of the entire workflow
debugging_mode = False

# used for variant effect prediction tasks
pred_types = [
    'predictions',
    'features_biological', 'features_technical',
    'predictions_biological', 'predictions_technical',
    'predictions_biological_ver2', 'predictions_technical_ver2',
]

def read_sweep_csv(path):

    cols = ["Name","ID","Sweep","basepath","metadata_file","metadata_mapping_config","model_class"]

    df = pd.read_csv(path)
    if 'ignore_classes' in df.columns:
        cols.append('ignore_classes')

    if "model_class" not in df.columns:
        print(f'"model_class" not found in columns of {path}, assuming they are all IEAquaticDilated')
        df['model_class'] = "IEAquaticDilated"
    else:
        # currently assume the model_class is IEAquaticDilated if missing.
        df.loc[df.model_class.isna(),'model_class'] = "IEAquaticDilated"
        df.loc[df.model_class.isin(['-']),'model_class'] = "IEAquaticDilated"

    return df[cols].copy()


def read_sample_files():
    # runs to be evaluated can be placed in the snakemake/ directory in separate CSV files
    # should follow the pattern snakemake/sweep*csv

    sample_files = glob('snakemake/sweep*csv')
    print(f'Found {len(sample_files)} sample files. Merging...')
    l = [read_sweep_csv(f) for f in sample_files]
    sample_df = pd.concat(l)
    print(f'{sample_df.shape[0]} runs after merging from {sample_df.Sweep.nunique()} sweeps.')

    duplicated = sample_df.ID[sample_df.ID.duplicated()]
    if len(duplicated) > 0:
        duplicated = ','.join(duplicated)
        print(f'Error: found duplicated sweep IDs. {duplicated}')
        sys.exit(1)

    if debugging_mode:
        print('debugging mode active, subsetting to the first sample only.')
        sample_df = sample_df.iloc[[0]]

    samples_imputation_df = sample_df.loc[~sample_df.ignore_classes.isna()]
    sample_df = sample_df.loc[sample_df.ignore_classes.isna()]
    return sample_df, samples_imputation_df


# read information on the different runs
samples, samples_imputation = read_sample_files()

# convenience rules

rule dump_sweep_ids:
    # rule to dump sweep ids to a file
    output:
        'sweep_id.txt'
    run:
        with open(output[0],'w') as outfile:
            for ID in samples['ID']:
                outfile.write(f'{ID}\n')

rule dump_sweeps_csv:
    # rule to dump the merged a sweeps.csv from the different files matching snakemake/sweep*csv
    output:
        'sweeps.csv'
    run:
        samples.to_csv(output[0],sep=',',index=False)

localrules:
    dump_sweep_ids,
    dump_sweeps_csv


# Checkpoint selection and model evaluation on the prediction task

rule get_wandb_val_metrics:
    # get validation set metrics to select checkpoint file(s)
    # uses wandb Api to fetch the information 
    output:
        'checkpoints/{ID}/metrics.tsv'
    conda:
        config.get('conda_env', 'nucleotran_cuda11_2')
    log:
        'checkpoints/{ID}/metrics.tsv.log'
    resources:
        partition='cpu',
        time='00:30:00',
        threads=1
    shell:
        "("
        "python src/wandb_val_metrics_epoch.py "
        "--run {wildcards[ID]} "
        "--out {output} "
        ") &> {log} "


rule all_get_wandb_val_metrics:
    # run rule above for all
    input:
        expand('checkpoints/{ID}/metrics.tsv',ID=samples.ID.values)


def get_available_checkpoint_files(wildcards):
    # fetch the input files for the rule below based on the run ID
    ret = {}
    ret['checkpoints'] = glob(f'checkpoints/{wildcards.ID}/*.ckpt')
    ret['metrics'] = f'checkpoints/{wildcards.ID}/metrics.tsv'
    return ret


rule select_best_checkpoint_file:
    # select the best available checkpoint file based on the metrics (in case there are multiple)
    input:
        unpack(get_available_checkpoint_files)
    output:
        'checkpoints/{ID}/best_model.ckpt'
    run:
        # currently this is hardcoded to select the one with the lowest validation loss (because that is what saving was based on...)
        metrics = pd.read_csv(input['metrics'],index_col=0,sep='\t')
        epochs_available = {int(os.path.basename(x).split('-')[0].replace('epoch=','')): x for x in
                            input['checkpoints']}
        metrics = metrics[metrics.epoch.isin(epochs_available.keys())]
        metrics = metrics.loc[metrics['val/loss'].idxmin()]
        best_checkpoint = epochs_available[metrics['epoch']]
        shell(f'ln -s -r {best_checkpoint} {output}')  # simply creates a symbolic link to the "best" checkpoint file

localrules:
    select_best_checkpoint_file

rule all_select_best_checkpoint_file:
    # run rule above for all
    input:
        expand('checkpoints/{ID}/best_model.ckpt',ID=samples.ID.values)


def get_genome_from_id(wildcards):
    # guesses the genome from the "basepath" given in the CSV for a given ID
    # used for some rules below.
    # infer the genome from the "basepath" argument
    # this only works if the path contains mm10 or GRCh38
    selected = samples.loc[samples.ID == wildcards.ID]['basepath'].values[0]
    if 'GRCh38' in selected:
        return 'GRCh38'
    elif 'mm10' in selected:
        return 'mm10'
    else:
        print('Error: could not determine reference genome from samples dataframe')
        sys.exit(1)

def get_model_class_from_id(wildcards):
    # gets the model class
    try:
        selected = samples.loc[samples.ID == wildcards.ID]['model_class'].values[0]
    except IndexError:
        return 'IEAquaticDilated'
    if pd.isna(selected) or selected == '':
        selected = 'IEAquaticDilated'
    return selected

rule calculate_best_model_roc_pr:
    # calculate the per-class AUPRC and AUROC
    # {set} can be train, val or test
    input:
        'checkpoints/{ID}/best_model.ckpt'
    output:
        'checkpoints/{ID}/best_model.{set}.roc_pr.tsv.gz'
    log:
        'checkpoints/{ID}/best_model.{set}.roc_pr.tsv.gz.log'
    params:
        genome=get_genome_from_id,
        dataset='full' if not debugging_mode else 'toy',
        model_cls=get_model_class_from_id # TODO: this can actually be guessed from the checkpoint file in python, no need for an argument
    resources:
        partition='gpupro,gpu',
        time="02:00:00",# could probably do with less time
        gpus=1,
        mem='128g'  # could probably do with less memory
    threads:
        6
    conda:
        config.get('conda_env', 'nucleotran_cuda11_2')
    shell:
        "("
        "python src/evaluate_model_roc_pr.py "
        "--dataset {params[dataset]} "
        "--genome {params[genome]} "
        "--set {wildcards[set]} "
        "--checkpoint {input} "
        "--num_workers {threads} "
        "--model_cls {params[model_cls]} "
        ") &> {log}"


rule all_calculate_best_model_roc_pr_validation:
    # run rule above for all
    input:
        expand('checkpoints/{ID}/best_model.{set}.roc_pr.tsv.gz',ID=samples.ID.values,set=['val','test'])

rule all_model_evaluation_on_prediction_task:
    input:
        rules.all_calculate_best_model_roc_pr_validation.input


rule download_encode_cre:
    # download encode CRE data
    # not used...
    output:
        pls="data/external/ENCODE_CRE/GRCh38-PLS.bed.gz",
        els="data/external/ENCODE_CRE/GRCh38-ELS.bed.gz",
        ctcf="data/external/ENCODE_CRE/GRCh38-CTCF.bed.gz"
    log:
        "data/external/ENCODE_CRE/download_encode_cre.log"
    shell:
        "("
        "cd data/external/ENCODE_CRE/ && bash 0_download_data_GRCh38.sh"
        ") &> {log} "


rule predict_gnomad_variants_best_model:
    # predict gnomAD variants intersected with encode cis-regulatory elements
    # {set} either els, pls, or ctcf
    # creates an hdf5 file with variant effect predictions and a txt file with the corresponding variant IDs
    input:
        ckpt='checkpoints/{ID}/best_model.ckpt',
        gnomad_variants=ancient('data/external/gnomAD/gnomad_intersected_with_encode_regulatory.tsv.gz')  # ignore time-stamps on this file
    output:
        vep='checkpoints/{ID}/best_model.gnomAD_{set}_VEP.h5',# TODO: these could be deleted once no longer needed because they are quite large
        varid='checkpoints/{ID}/best_model.gnomAD_{set}_varID.txt.gz'
    log:
        'checkpoints/{ID}/best_model.gnomAD_{set}_VEP.h5.log'
    params:
        genome = get_genome_from_id,
        debug = '' if not debugging_mode else '--debug',
        model_cls = get_model_class_from_id
    resources:
        partition='gpupro,gpu',
        time="03:00:00", # could probably do with less
        gpus=1,
        mem='128g',
    threads:
        4
    conda:
        config.get('conda_env', 'nucleotran_cuda11_2')
    shell:
        "("
        "python src/predict_gnomad_variants.py "
        "--ref {params[genome]} "
        "--set {wildcards[set]} "
        "--ckpt {input[ckpt]} "
        "--model_cls {params[model_cls]} "
        "--feature_predict predictions " # can be set to "predictions" in order to use predictions instead  
        "{params[debug]} "
        ") &> {log}"

rule all_predict_gnomad_variants_best_model:
    input:
        expand('checkpoints/{ID}/best_model.gnomAD_{set}_VEP.h5',ID=samples.ID.values,set=['els','pls'])


rule evaluate_gnomad_variants_best_model:
    # calculate enrichments for rare variants
    # fisher exact tests depending on variant effect prediction cutoff quantiles
    # {set} is either pls, els, or ctcf
    # {method} is either "absolute" (use the absolute values for VEP) or "bidirectional" (look at positive and negative VEPs separately)
    # {pred_type} is one of ['predictions','features_biological', 'features_technical', 'logits_biological', 'logits_technical'] 
    input:
        vep='checkpoints/{ID}/best_model.gnomAD_{set}_VEP.h5',
        varid='checkpoints/{ID}/best_model.gnomAD_{set}_varID.txt.gz',
        gnomad_variants=ancient('data/external/gnomAD/gnomad_intersected_with_encode_regulatory.tsv.gz')  # ignore time-stamps on this file
    output:
        results='checkpoints/{ID}/best_model.gnomAD_{set}_VEP.{pred_type}.{method}_results.tsv.gz',
        total_counts='checkpoints/{ID}/best_model.gnomAD_{set}_VEP.{pred_type}.{method}_totalcounts.tsv'
    params:
        debug='' if not debugging_mode else '--debug'
    log:
        'checkpoints/{ID}/best_model.gnomAD_{set}_VEP.{pred_type}.{method}_results.tsv.gz.log'
    resources:
        partition='cpu',
        time="05:00:00",# could probably do with less
        mem='16g'  # could probably do with less memory (looks like it needs as little as 4-5g )
    threads:
        1
    conda:
        config.get('conda_env', 'nucleotran_cuda11_2')
    shell:
        "("
        "python src/evaluate_VEP.py "
        "--vep {input[vep]} "
        "--varid {input[varid]} "
        "--method {wildcards[method]} "
        "--pred_type {wildcards[pred_type]} "
        "{params[debug]} "
        ") &> {log}"


def expand_output_pred_type(pattern, ID, pred_type=None, **kwargs):
    # infers which pred-types are available for a given model (based on the presence of a metadata mapping config)
    # pattern: the pattern to expand with wildcards {ID} and {pred_type}
    # ID: the ids for which to request outputs
    # pred_type: if not None, the requested pred-type
    # **kwargs: remaining argume
    sample_df = samples.set_index('ID').loc[ID]

    if pred_type is None:
        # when pred_type is not given, request all supported pred-types for each model 
        id = []
        pt = []
        for i, val in sample_df.iterrows():
            if val['metadata_mapping_config'] == "":
                # does not accept metadata and produces only "predictions" pred-type
                id += [i]
                pt += ["predictions"]
            elif not pd.isna(val['metadata_mapping_config']):
                # should accept metadata and produce all pred-types
                id += [i] * len(pred_types)
                pt += pred_types
            else:
                # does not accept metadata and produces only "predictions" pred-type
                id += [i]
                pt += ["predictions"]
    else:
        assert pred_type in pred_types, f"the requested prediction type '{pred_type}' is not one of {pred_types}"
        # when pred_type is given, request it for all models that support it
        id = []
        pt = []
        for i, val in sample_df.iterrows():
            if pred_type != "predictions":
                if val['metadata_mapping_config'] == "":
                    continue
                elif  pd.isna(val['metadata_mapping_config']):
                    continue
            id += [i]
            pt += [pred_type]

    ret = expand(pattern, zip, ID=id, pred_type=pt, allow_missing=True) # first expand call uses zip

    if len(kwargs):
        ret = expand(ret, **kwargs)

    return ret

# convenience rule to request gnomad evaluation for a specific set (e.g., "predictions", "logits_biological")
rule set_evaluate_gnomad_variants_best_model:
    input:
         lambda wc: expand_output_pred_type('checkpoints/{ID}/best_model.gnomAD_{set}_VEP.{pred_type}.{method}_results.tsv.gz', ID=samples.ID.values, pred_type=wc['pred_type'], set=['els','pls'], method=['bidirectional'])
    output:
        touch('checkpoints/gnomad.{pred_type}.eval.ok')

localrules:
    set_evaluate_gnomad_variants_best_model


rule all_evaluate_gnomad_variants_best_model:
    # request rule above for all prediction types
    input:
        expand_output_pred_type('checkpoints/{ID}/best_model.gnomAD_{set}_VEP.{pred_type}.{method}_results.tsv.gz', ID=samples.ID.values, set=['els','pls'], method=['bidirectional'])


# rule select_variants_from_vep_quantile_best_model:
#     # extract variants for features significantly enriched for rare variants at specific quantiles
#     # {set} is either pls, els, or ctcf
#     # {method} is either "absolute" (use the absolute values for VEP) or "bidirectional" (look at positive and negative VEPs separately)
#     # {pred_type} is one of ['predictions','features_biological', 'features_technical', 'logits_biological', 'logits_technical']
#     input:
#         vep='checkpoints/{ID}/best_model.gnomAD_{set}_VEP.h5',
#         varid='checkpoints/{ID}/best_model.gnomAD_{set}_varID.txt.gz',
#         results='checkpoints/{ID}/best_model.gnomAD_{set}_VEP.{pred_type}.{method}_results.tsv.gz'
#     output:
#         q001='checkpoints/{ID}/best_model.gnomAD_{set}_VEP.{pred_type}.{method}_results.q0_001_variants.tsv.gz',
#         q999='checkpoints/{ID}/best_model.gnomAD_{set}_VEP.{pred_type}.{method}_results.q0_999_variants.tsv.gz'
#     params:
#         debug='' if not debugging_mode else '--debug'
#     log:
#         # TODO: make more consistent log file names...
#         'checkpoints/{ID}/best_model.gnomAD_{set}_VEP.{pred_type}.{method}_results.qX.log'
#     resources:
#         partition='cpu',
#         time="05:00:00",# could probably do with less
#         mem='16g'  # could probably do with less memory
#     threads:
#         1
#     conda:
#         config.get('conda_env', 'nucleotran_cuda11_2')
#     shell:
#         "("
#         "python src/select_variants_from_VEP_quantile.py "
#         "--vep {input[vep]} "
#         "--varid {input[varid]} "
#         "--pred_type {wildcards[pred_type]} "
#         "--result_file {input[results]} "
#         "--pvalue_thresh 1e-5 "
#         "--quantiles 0.001,0.999 "
#         "{params[debug]} "
#         ") &> {log}"
#
# rule all_select_variants_from_vep_quantile_best_model:
#     # request rule above for all prediction types
#     input:
#         expand_output_pred_type(rules.select_variants_from_vep_quantile_best_model.output, ID=samples.ID.values, set=['els','pls'], method=['bidirectional'])


rule select_variants_from_vep_quantile_best_model:
    # extract variants for features significantly enriched for rare variants at specific quantiles
    # {set} is either pls, els, or ctcf
    # {method} is either "absolute" (use the absolute values for VEP) or "bidirectional" (look at positive and negative VEPs separately)
    # {pred_type} is one of ['predictions','features_biological', 'features_technical', 'logits_biological', 'logits_technical']
    input:
        vep='checkpoints/{ID}/best_model.gnomAD_{set}_VEP.h5',
        varid='checkpoints/{ID}/best_model.gnomAD_{set}_varID.txt.gz',
        results='checkpoints/{ID}/best_model.gnomAD_{set}_VEP.{pred_type}.{method}_results.tsv.gz'
    output:
        q1 = 'checkpoints/{ID}/best_model.gnomAD_{set}_VEP.{pred_type}.{method}_results.q0_1_variants.tsv.gz',
        q9 = 'checkpoints/{ID}/best_model.gnomAD_{set}_VEP.{pred_type}.{method}_results.q0_9_variants.tsv.gz',
        q01 = 'checkpoints/{ID}/best_model.gnomAD_{set}_VEP.{pred_type}.{method}_results.q0_01_variants.tsv.gz',
        q99 = 'checkpoints/{ID}/best_model.gnomAD_{set}_VEP.{pred_type}.{method}_results.q0_99_variants.tsv.gz',
        q001='checkpoints/{ID}/best_model.gnomAD_{set}_VEP.{pred_type}.{method}_results.q0_001_variants.tsv.gz',
        q999='checkpoints/{ID}/best_model.gnomAD_{set}_VEP.{pred_type}.{method}_results.q0_999_variants.tsv.gz'
    params:
        debug='' if not debugging_mode else '--debug'
    log:
        # TODO: make more consistent log file names...
        'checkpoints/{ID}/best_model.gnomAD_{set}_VEP.{pred_type}.{method}_results.qX.log'
    resources:
        partition='cpu',
        time="05:00:00",# could probably do with less
        mem='16g'  # could probably do with less memory
    threads:
        1
    conda:
        config.get('conda_env', 'nucleotran_cuda11_2')
    shell:
        "("
        'for quants in "0.1,0.9" "0.01,0.99" "0.001,0.999"; do '
        "python src/select_variants_from_VEP_quantile.py "
        "--vep {input[vep]} "
        "--varid {input[varid]} "
        "--pred_type {wildcards[pred_type]} "
        "--result_file {input[results]} "
        "--pvalue_thresh 1e-5 "
        "--quantiles $quants "
        "{params[debug]} "
        "; done"
        ") &> {log}"


rule all_select_variants_from_vep_quantile_best_model:
    # request rule above for all prediction types
    input:
        expand_output_pred_type(rules.select_variants_from_vep_quantile_best_model.output,
            ID=samples.ID.values, set=['els','pls'], method=['bidirectional'])


rule all_gnomad:
    # convenience rule to run all gmomAD-related rules
    input:
        rules.all_select_variants_from_vep_quantile_best_model.input,
        rules.all_evaluate_gnomad_variants_best_model.input,
        rules.all_predict_gnomad_variants_best_model.input


rule vista_enhancer_prediction_task:
    # TODO: comments
    # TODO: make this work for models that dont use metadata
    input:
        ckpt='checkpoints/{ID}/best_model.ckpt',
        vista_bed='data/external/vista/vista_enhancers.bed'
    output:
        auc='checkpoints/{ID}/vista_enhancer/auc.tsv',
        report='checkpoints/{ID}/vista_enhancer/report.md',
        txt='checkpoints/{ID}/vista_enhancer/log.txt',
        pca=expand('checkpoints/{{ID}}/vista_enhancer/{name}_pca.npy', name=['biological_max','biological_avg','technical_max','technical_avg','predictions_avg']),
        pca_expl_var='checkpoints/{ID}/vista_enhancer/pca_expl_var.tsv',
        gc_correl='checkpoints/{ID}/vista_enhancer/GC_content_correlations.tsv'
    params:
        debug='' if not debugging_mode else '--debug'
    log:
        'checkpoints/{ID}/vista_enhancer/report.log'
    resources:
        partition='cpu,gpu,gpupro',
        time="24:30:00",
        gpus=0,
        mem='32g'  # could probably do with less memory
    threads:
        8
    conda:
        config.get('conda_env', 'nucleotran_cuda11_2')
    shell:
        "("
        "PYTHONPATH=$PWD/src python src/downstreamtasks/vista_enhancer.py "
        "--ckpt {input[ckpt]} "
        "--n_jobs {threads} "
        "{params[debug]} "
        ") &> {log}"


rule all_vista_enhancer_prediction_task:
    input:
        # TODO: make work for models that dont use metadata
        expand('checkpoints/{ID}/vista_enhancer/auc.tsv', ID=samples.loc[samples.model_class != 'OligonucleotideModel'].ID.values)


rule predict_and_evaluate_gtex_finemapped_variants:
    # run gtex fine-mapped variant effect predictions and export results
    # produces similar-ish results to the gnomAD scripts above
    # also produces other output files not listed depending if the model uses metadata or not (logits_..., features_...)
    input:
        ckpt='checkpoints/{ID}/best_model.ckpt',
        gtex_variants_pos='data/external/eQTL_catalog/highPIP_variants_agg.tsv.gz',
        gtex_variants_neg='data/external/eQTL_catalog/lowPIP_variants_agg.tsv.gz'
    output:
        results='checkpoints/{ID}/best_model.gtex_eqtl_VEP.predictions.results.tsv.gz',
        variants=[
            'checkpoints/{ID}/best_model.gtex_eqtl_VEP.predictions.results.q0_9_variants.tsv.gz',
            'checkpoints/{ID}/best_model.gtex_eqtl_VEP.predictions.results.q0_95_variants.tsv.gz',
            'checkpoints/{ID}/best_model.gtex_eqtl_VEP.predictions.results.q0_99_variants.tsv.gz',
        ],
        ok=touch('checkpoints/{ID}/best_model.gtex_eqtl_VEP.all.ok') # this file can be deleted to force a re-run
    log:
        'checkpoints/{ID}/best_model.gtex_eqtl_VEP.predictions.results.tsv.gz.log'
    params:
        debug = '' if not debugging_mode else '--debug',
        model_cls = get_model_class_from_id
    threads:
        1
    resources:
        partition='gpupro,gpu',
        gpus=1,
        time='03:00:00',
        mem='16g'  # can probably do with less of everything...
    conda:
        config.get('conda_env', 'nucleotran_cuda11_2')
    shell:
        "("
        "python src/predict_and_evaluate_gtex_eqtl_finemapped_variants.py "
        "--ckpt {input[ckpt]} "
        "--pvalue_thresh 1e-5 "
        "--model_cls {params[model_cls]} "
        "--feature_predict predictions " # can also be set to "predictions"
        "{params[debug]} "
        ") &> {log}"


rule fantom_enhancer_prediction:
    input:
        ckpt='checkpoints/{ID}/best_model.ckpt',
        bed_path='data/external/fantom/fantom5_enhancers_hg38.bed',
        label_id_path='data/external/fantom/fantom5_tissue_labels_selected_enh.csv'
    output:
        models='checkpoints/{ID}/fantom/fantom_saved_models.pkl',
        results='checkpoints/{ID}/fantom/fantom_AUCs.csv',
        results_plot = 'checkpoints/{ID}/fantom/fantom_per_tissue_AUCs.png',
        results_summary = 'checkpoints/{ID}/fantom/summary_fantom.md'
    log:
        'checkpoints/{ID}/fantom/fantom.log'
    params:
        debug = '' if not debugging_mode else '--debug'
    threads:
        8
    resources:
        partition='cpu',
        mem='48g',
        time='24:00:00'
    conda:
        config.get('conda_env', 'nucleotran_cuda11_2')
    shell:
        "("
        "PYTHONPATH=$PWD/src "
        "python src/downstreamtasks/evaluate_fantom.py "
        "--ckpt {input[ckpt]} "
        "--out checkpoints/{wildcards[ID]}/fantom/ "
        "--n_jobs {threads} "
        "{params[debug]} "
        ") &> {log}"

        
rule all_fantom:
    # TODO: implement for models that dont use metadata
    input:
        expand(rules.fantom_enhancer_prediction.output, ID=samples.loc[samples.model_class != 'OligonucleotideModel'].ID.values)


rule all_predict_and_evaluate_gtex_finemapped_variants:
    # run rule above for all variants
    input:
        expand(rules.predict_and_evaluate_gtex_finemapped_variants.output,ID=samples.ID.values)


rule all_gtex:
    # convenience rule to run all GTEx-related rules
    input:
        rules.all_predict_and_evaluate_gtex_finemapped_variants.input


rule data_imputation:
    input:
        ckpt='checkpoints/{ID}/best_model.ckpt',
    output:
        results='checkpoints/{ID}/imputation/auroc.tsv',
    log:
        'checkpoints/{ID}/imputation/imputation.log'
    params:
        debug = '' if not debugging_mode else '--debug'
    threads:
        5
    resources:
        partition='gpu,gpupro',
        mem='160gb',
        time='24:00:00',
        gpus=1
    conda:
        config.get('conda_env', 'nucleotran_cuda11_2')
    shell:
        "("
        "PYTHONPATH=$PWD/src "
        "python src/downstreamtasks/data_imputation.py "
        "--ckpt {input[ckpt]} "
        "{params[debug]} "
        ") &> {log}"


rule all_data_imputation:
    # TODO: implement for models that dont use metadata
    input:
        expand(rules.data_imputation.output, ID=samples_imputation.ID.values)


rule all:
    # convenience rule to run everything
    input:
        rules.all_model_evaluation_on_prediction_task.input,
        rules.all_gnomad.input,
        rules.all_gtex.input,
        rules.all_vista_enhancer_prediction_task.input,
        rules.all_fantom.input


rule all_data_export:
    # convenience rule to create a tar archive with results
    # does not trigger re-runs (all inputs wrapped in ancient(...))
    input:
        # config
        ancient(glob('snakemake/sweep*csv')),
        # results
        ancient(rules.all_get_wandb_val_metrics.input),
        ancient(rules.all_select_best_checkpoint_file.input),
        ancient(expand('checkpoints/{ID}/best_model.{set}.roc_pr.tsv.gz', ID=samples.ID.values, set=['val','test'])),
        ancient(expand_output_pred_type(rules.evaluate_gnomad_variants_best_model.output, ID=samples.ID.values, set=['els','pls'], method=['bidirectional'])),
        ancient(expand_output_pred_type(rules.select_variants_from_vep_quantile_best_model.output, ID=samples.ID.values, set=['els','pls'], method=['bidirectional'])),
        ancient(expand(rules.predict_and_evaluate_gtex_finemapped_variants.output, ID=samples.ID.values)),
        ancient(rules.all_vista_enhancer_prediction_task.input),
        ancient(rules.all_fantom.input),
        # log files
        ancient(rules.all_get_wandb_val_metrics.log),
        ancient(rules.all_select_best_checkpoint_file.log),
        ancient(expand_output_pred_type(rules.evaluate_gnomad_variants_best_model.log, ID=samples.ID.values, set=['els','pls'], method=['bidirectional'])),
        ancient(expand_output_pred_type(rules.select_variants_from_vep_quantile_best_model.log, ID=samples.ID.values, set=['els','pls'], method=['bidirectional'])),
        ancient(expand(rules.predict_and_evaluate_gtex_finemapped_variants.log, ID=samples.ID.values)),
        ancient(rules.all_vista_enhancer_prediction_task.log),
        ancient(rules.all_fantom.log)

    output:
        'results_dump.tar.gz'
    run:
        inputs = []
        for i in input:
            # dont know if this is actually necessary...
            if isinstance(i, str):
                i = [i]
            inputs += i
        inputs = ' '.join(inputs)
        shell(f'tar -hzvcf {output} {inputs}')

localrules:
    all_data_export







