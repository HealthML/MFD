Metadata-guided Feature Disentanglement for Functional Genomics
==============================
*Alexander Rakowski, Remo Monti, Viktoriia Huryn, Marta Lemanczyk, Uwe Ohler, Christoph Lippert*

Accepted for publication at ECCB 2024.

arXiv preprint: https://arxiv.org/abs/2405.19057

Data used for model training: https://figshare.com/ndownloader/files/38200983

**Code will be added soon**

# Setup

1. clone the repository
2. `bash download_data.sh`
3. `cp src/config.yaml.template src/config.yaml`
4. edit the paths in the `src/config.yaml`

# Weights and biases
Training relies on weights and biases (wandb). wandb project and user name / entity are defined inside `src/config.yaml`.
For an example on how to set up a sweep consider the information [here](https://github.com/HealthML/nucleotran/blob/tissue_experiments/docs/running_sweeps.md).

# Snakemake
After training has completed, a snakemake workflow automates model (checkpoint) evaluation.
Follow these steps to set up snakemake:

- install snakemake in a new conda environment preferably called `snakemake` (see information on https://snakemake.readthedocs.io/en/stable/)
- make sure you have configured `src/config.yaml` (variables in there are also used by snakemake)
- change the name of the GPU (pytorch,wandb,pytorch lightning) environment from `nucleotran_cuda11_2` to the name of your environment inside the in `src/config.yaml`
- In order to configure which checkpoints to evaluate, you can place separate sweep metadata files into `./snakemake/` as long as they follow the pattern `./snakemake/sweep*.csv`. Each sweep CSV file needs to have at least these columns: `"Name","ID","Sweep","basepath","metadata_file","metadata_mapping_config"`. A CSV like this can be exported from the wandb website or using the wandb API. More information on configuring a workflow with tabular data can be found [here](https://snakemake.readthedocs.io/en/stable/snakefiles/configuration.html#tabular-configuration).
  - There is a script `src/wandb_create_sweep_csv.py` that can create such CSV files from a list of run IDs.
- rules (analysis steps) are defined inside `Snakefile`
  - each rule defines input files, output files, and code which is executed to produce those files.

# Running the snakemake workflow with slurm
Snakemake rules define their own compute resources (e.g., cores, GPUs). These can be used to automatically generate job scripts and distribute the workflow using a scheduler like slurm.
This repository comes with a snakemake slurm configuration located at `slurm/config.yaml` that works on the DHC-lab cluster.
Rules can be invoked by requesting their output files (e.g., `checkpoints/{ID}/best_model.ckpt`), where `{ID}` is a "wildcard" that will match the wand run ID in this case (more about wildcards [here](https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#wildcards)).
Alternatively, rules can be invoked with the rule name (as long as their ouput files do not contain wildcards, or they have no output files but request only input files).

There is a script that sets up some sensible default snakemake arguments called `run_snakemake_sbatch.sh`.
## Submitting jobs
- for example, to request AUC values for the "best" checkpoints saved for all models listed in the `snakemake/sweep.csv` file(s):
  - check what would be run: `bash run_snakemake_cluster.sh -n all_calculate_best_model_roc_pr_validation`
  - submit a job that will execute the steps by submitting other jobs: `sbatch run_snakemake_cluster.sh all_calculate_best_model_roc_pr_validation`

If you only want to request the outputs for a specific run, and set (`train`,`test`,`val`), you could instead:

- check what would be run: `bash run_snakemake_cluster.sh -n checkpoints/{ID}/best_model.{set}.roc_pr.tsv.gz`
  - `{ID}` is a wildcard that needs to be substituted witht he run ID, and `{set}` has to be substituted with either `train`, `test` or `val`
- submit to the cluster `sbatch run_snakemake_cluster.sh checkpoints/{ID}/best_model.{set}.roc_pr.tsv.gz`

The `Snakefile` contains many rules starting with `all_...` that allow running a specific rule for all runs listed in the `snakemake/sweep*csv` files. 

## Workflow debugging

To run the workflow in debugging mode (subset to only one sample ID and trigger the `--debug` flag for many rules), you can set  `debugging_mode = True` at the top of the `Snakefile`. This will make many rules run on just a subset of the data and make them complete a lot faster (make sure to remove the output files after execution though!).

## Convenience rules

To run all model evaluation rules related to the prediction task:
- `sbatch run_snakemake_cluster.sh all_model_evaluation_on_prediction_task`

To run all gnomAD rules:
- `sbatch run_snakemake_cluster.sh all_gnomad`

To run all GTEx rules:
- `sbatch run_snakemake_cluster.sh all_gtex`

To run all evaluation and downstream tasks:
- `sbatch run_snakemake_cluster.sh all`

To export results from evaluation tasks to a file `results_dump.tar.gz` (runs locally and does not trigger re-runs)
- `bash run_snakemake_cluster.sh all_data_export`

See description of the tasks below.

# Downstream analysis steps defined in `Snakefile` 
### Status 7.11.2023

Below, all rules are briefly documented. Usually there is a rule starting with `all_<rule name>` to run them for all run IDs. 

## Selecting the "best" checkpoint and collecting performance metrics on the prediction task

### Rule `get_wandb_val_metrics`

Uses wandb to collect the logged metrics across epochs for a given run `{ID}`.
This file is used to select the best checkpoint.
The metrics TSV will contain all the metrics logged during training like `train/loss`, `train/loss_pred`, etc.

outputs:
-  `checkpoints/{ID}/metrics.tsv`


### Rule `select_best_checkpoint_file`

Selects the "best" *available* checkpoint based on `val/loss` for run `{ID}` if more than one checkpoint was saved.

outputs:
- `checkpoints/{ID}/best_model.ckpt`

> Tip: you can circumvent this selection by simply defining the `best_model.ckpt` yourself for each `{ID}` if you want to select on different criteria.

### Rule `calculate_best_model_roc_pr`

outputs:
- `checkpoints/{ID}/best_model.{set}.roc_pr.tsv.gz`

used command-line-tool:
- `src/evaluate_model_roc_pr.py`

Calculates the area under the precision recall curve and area under the receiver operator characteristic curve of the model stored in the `best_checkpoint.ckpt` file for each run `{ID}` and set `{set}` (`train`,`test`,`val`). Predictions are averaged across forward and reverse strands.

The rule `calculate_best_model_roc_pr` will request to run this rule for all run IDs and the validation set. 


## gnomAD variants
These rules evaluate the zero-shot performance of the best model's variant effect predictions to distinguish rare variants (potentially damaging variants) from common variants (likely benign variants). The original data were taken from Benegas et al. [*GPN-MSA: an alignment-based DNA language model for genome-wide variant effect prediction*](https://www.biorxiv.org/content/10.1101/2023.10.10.561776v1).

The full dataset is available on [huggingface](https://huggingface.co/datasets/songlab/human_variants). This dataset contains human variants. Find more information about gnomAD [here](https://gnomad.broadinstitute.org/). 

Variants were intersected with ENCODE-defined cis-regulatory elements (ENCODE CREs) (https://screen.encodeproject.org/) and filtered for non-coding variants. The ENCODE CREs comprise potential enhancers (`els`), promoters (`pls`) and CTCF-bound elements (`ctcf`).

The intersection brings the dataset down to about 1 million variants. Processing steps are described in `data/external/00_download_gnomad_variants_intersect_with_encode.ipynb`.

### Rule `predict_gnomad_variants_best_model`

This rule will perform variant effect predictions (`model(alt_seq) - model(ref_seq)`) for all variants in `{set}` for run `{ID}` and dump them to an HDF5 file. 

outputs (by name):
- `vep`: HDF5 file containing variant effect predictions
  - the HDF5 contains 5 datasets `predictions`, `features_biological`, `features_technical`, `logits_biological`, `logits_technical`.
- `varid` a text file containing the variant IDs for the predictions in `vep`.

used command-line-tool:
- `src/evaluate_VEP.py`

special wildcards:
- `{set}` the ENCODE-CRE set to use (`ctcf`,`els`,`pls`). By default the workflow will choose `els`

> Note: the HDF5 files are very large (up to 10G), and should be deleted once the other rules depending on these files below have finished.

### Rule `evaluate_gnomad_variants_best_model`

Calculates the enrichment of rare variants at different VEP thresholds (quantiles). 

outputs (by name):
- `results`: a TSV file containing, for each feature and evaluated threshold, the enrichment for rare variants (see below)
- `total_counts`: a small file containing total counts of Rare/Common variants in the analysed set

used command-line-tool:
- `src/evaluate_VEP.py`

special wildcards:
- `{pred_type}` one of the types of predictions (`predictions`, `features_biological`, `features_technical`, `logits_biological`, `logits_technical`)
  - by default the workflow will run all of these
- `{method}` whether to look at both extreme sides of VEPs separately (`bidirectional`) or take the absolute value of VEPs (`absolute`). By default will do `bidirectional` analysis

**Example output for one feature**

| q      | cut        | Common | Rare  | OR       | pval                    | idx |
|--------|------------|--------|-------|----------|-------------------------|-----|
| 0.0001 | -0.02372   | 44     | 48    | 1.00999  | 1.0                     | 0   |
| 0.001  | -0.00870   | 426    | 493   | 1.07151  | 0.30594                 | 0   |
| 0.01   | -0.00250   | 4480   | 4773  | 0.98623  | 0.51006                 | 0   |
| 0.1    | -0.00040   | 44262  | 47523 | 0.99337  | 0.34005                 | 0   |
| 0.5    | 0.00001    | 220321 | 238463| 1.00412  | 0.32598                 | 0   |
| 0.9    | 0.00041    | 47829  | 43891 | 0.83433  | 4.06703e-149            | 0   |
| 0.99   | 0.00233    | 5333   | 3853  | 0.66619  | 1.24165e-82             | 0   |
| 0.999  | 0.00787    | 556    | 365   | 0.60748  | 8.64057e-14             | 0   |
| 0.9999 | 0.02182    | 62     | 30    | 0.44794  | 0.00023                 | 0   |

- q: the quantile
- cut: the cutoff corresponding to the quantile
- Common: the number of common variants at the cutoff
- Rare: the number of rare variants  at the cutoff
- OR: the odds-ratio for the test comparing variants in that quantile agains all the rest (measures the enrichment of rare variants)
- pval: p-value for the fisher exact test
- idx: the index for the feature (in the same order it comes out of the model, i.e., 0 is the first feature)

In the example above, the feature's variant effect predictions *deplete* rare variant at larger quantiles.

### Rule `select_variants_from_vep_quantile_best_model`

Extract the variants for features for which variant effect predictions were significantly enriched for rare variants. These files can be used to see if different types of predictions or models select different sets of variants, or calculate the number of unique variants prioritized by the model.

outputs (by name):
   - `q001` a TSV file containing variants selected at quantile `0.001`
   - `q999` a TSV file containing variants selected at quantile `0.999`

special wildcards:
- uses the same wildcards as the rule above

used command-line-tool:
- `src/select_variants_from_VEP_quantile.py`

the default is to select at quantiles `0.999` and `0.001` and select features enriched with p-value threshold `1e-5`.

**Example output**

| varID         | N   | i_select                              |
|---------------|-----|---------------------------------------|
| 1_924305_G_A  | 6   | 76,151,246,512,558,2026               |
| 1_924310_C_G  | 6   | 160,517,558,632,1584,18               |
| 1_924533_A_G  | 3   | 160,632,1584                          |
| 1_938654_G_A  | 116 | 27,66,70,126,133,153,[...],15         |
| 1_941787_A_G  | 2   | 495,537                               |
| 1_961346_A_C  | 1   | 699                                   |

- varID: the variant identifier
- N: the number of features that selected this variant
- i_select: the feature index values (match `idx` in output of the rule above) in named afte the order they come out of the model


## GTEx variants

This task was inspired by Karrolus et al. 2023. [*Current sequence-based models capture gene expression determinants in promoters but mostly ignore distal enhancers*](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-02899-9). But re-processes the [GTEx data from the eQTL-catalog](https://www.ebi.ac.uk/eqtl/Methods/) (it does not use the data provided with the Karrolus study). Find more information about GTEx [here](https://www.science.org/doi/10.1126/science.aaz1776). 

The goal is the measure the zero-shot performance of the model VEPs to distinguish Susie-finemapped-eQTL variants (potentially causal expression eQTL variants with high "posterior inclusion probabilities"; PIP) from position- and allele-frequency-matched non-finemapped variants (potentially non-causal variants with low PIP). 

To arrive at the two sets, the following filters were applied:
- exclude all variants that overlap protein-coding genes
- exclude all variants that overlap the gene with which they are associated (if the gene is a lncRNA for example)

for the **positive set**:
- keep only variants with `PIP > 0.95` in the tissue of interest, and at least `PIP > 0.5` across tissues in which the variant was associated with gene expression.

for the **negative set**:
- for each positive variant, select variants in a +- 5kb window around the variant
  - make sure the variant does not overlap the gene associated with the positive variant
- keep only variants with `max(PIP) < 0.05 ` across all tissues
- keep only the variant with the closest minor allele frequency to the positive variant
  - in case there are ties, select the physically closest variant (based on variant position)

Data download and processing are described in 
- `data/external/eQTL_catalog/00_download_eqtl_catalog_gene_annotation.ipynb`
- `data/external/eQTL_catalog/01_download_eqtl_catalog_susie_and_sumstats.ipynb`
- `data/external/eQTL_catalog/02_get_highpip_variants_select_negatives.ipynb`

These filters lead to a set of 2339 positives (SNPs) and 2304 negatives (SNPs), some negatives were selected for multiple positives (explaining the small difference between the numbers). There are also some longer indels in the files but these are not evaluated by the rule below. 

### Rule `predict_and_evaluate_gtex_finemapped_variants`

Run the GTEx variant effect predictions and export result tables similar to those for the gnomAD task above.
Unlike for the gnomAD task, this only looks at absolute values for the VEPs. It additionally performs logistic regression of the absolute VEPs against an indicator variable (0/1) indicating if the variant was fine-mapped (positive) or not (negative), as well as a zero-shot AUROC against those labels. It also calculates enrichments and performs Fisher exact tests for fine-mapped variants at two VEP quantiles (0.9 and 0.99, i.e., the 10% and 1% largest VEPs vs the rest) for each feature and prediction type. 

outputs (by name):
- `results` a TSV containing, for each feature and evaluated threshold, the enrichment for fine-mapped variants
  - there is one results file for each prediction type (`predictions`, `features_biological`, ...)
- `variants` a TSV containing variants selected at quantile `0.99` for features with VEPs significantly enriched for eQTL variants (`p<1e-5`).
  - there is one variants file for each prediction type

used command-line-tool:
   - `src/predict_and_evaluate_gtex_eqtl_finemapped_variants.py`


The `variants` TSV files have the same format at the on from gnomAD rule `select_variants_from_vep_quantile_best_model`.

The `results` TSV contains many columns (displayed transposed here):

| idx | 0         | the feature index |
|-----|-----------|-------------|
| mean_diff | 0.000407  | the average absolute VEPs |
| sd_diff | 0.00125 | the standard deviation of the absolute VEPs |
| coef | -0.00514 | coefficient in the logistic regression on (0/1); log(OR)/sd(VEP) |
| se | 0.0295 | standard deviation of the coefficient |
| stat | -0.174 | test statisic |
| pval | 0.862 | p-value |
| ci_low | -0.0630 | 95% confidence interval lower bound |
| ci_high | 0.0527 | 95% confience interval upper bound |
| pseudo_r2 | 4.71e-06 | pseudo r-squared |
| converged | True | if the optimization converged |
| N | 4643 | total number of observations |
| N_pos | 2339 | number of positive variants |
| N_neg | 2304 | number of negative variants |
| AUROC | 0.506 | zero-shot area under the ROC curve |
| cut_q0.9 | 0.711 | cutoff for quantile 0.9 |
| N_pos_q0.9 | 245 | number of positives selected at q0.9 |
| N_neg_q0.9 | 222 | number of negatives selected at q0.9 |
| OR_q0.9 | 1.10 | odds ratio at q0.9 |
| pfisher_q0.9 | 0.354 | p-value for the Fisher exact test at q0.9 |
| cut_q0.99 | 3.29 | cutoff for quantile 0.99 |
| N_pos_q0.99 | 24 | number of positives selected at q0.99 |
| N_neg_q0.99 | 24 |  number of negatives selected at q0.99  |
| OR_q0.99 | 0.985 | odds ratio at q0.99 |
| pfisher_q0.99 | 1.0 | p-value for the Fisher exact test at q0.99 |



Project Organization (TODO: update this)
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources including notebooks for processing
    │   └── processed      <- Main data for model training etc
    │    
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         and a short description. notebooks are ordered in groups with folders
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported (TODO: make this work)
    ├── src               <- Source code for use in this project.
       ├── __init__.py    <- Makes src a Python module
       │
       ├── dataloading    <- pytorch dataloaders and related classes
       │
       ├── features       <- Scripts to turn raw data into features for modeling
       │
       ├── models         <- model-related classes etc



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


# Useful for debugging: run it interactive on terminal without wandb
python3 src/train_wandb.py --no_wandb_use_config_file_directly=sweep_configurations/default_sweep_config.json
