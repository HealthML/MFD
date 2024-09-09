### Create a sweep in W&B
Sample configuration for human data:
```
method: grid
metric:
  goal: max
  name: val/auroc
parameters:
  C:
    values:
      - 120
  basepath:
    values:
      - data/processed/GRCh38/221111_128bp_minoverlap64_mincov2_nc10_tissues
  batch_size:
    values:
      - 256
  coeff_cov:
    values:
      - 0.1
  lr:
    values:
      - 0.001
  max_epochs:
    values:
      - 100
  metadata_mapping_config:
    values:
      - src/config_metadata_target_interactions.yaml
  out_pool_type:
    values:
      - max
  seed_everything:
    values:
      - 1
  seq_len:
    values:
      - 2176
  seq_order:
    values:
      - 2
```

You might want to change the `coeff_cov` value to 0 to disable the independence regularization, and 
`metadata_mapping_config` to your desired metadata config file.

### SBATCH script
```
#!/bin/bash -eux

#SBATCH --job-name=your_job_name
#SBATCH --mail-type=FAIL,ARRAY_TASKS
#SBATCH --mail-user=name.surname@hpi.de
#SBATCH --partition=gpupro,gpua100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=3-00:00:00
#SBATCH -o logs_directory/%x_%A_%a.log

echo "${SLURM_JOB_ID}"
echo "${SLURM_ARRAY_JOB_ID:-''}"
echo "${SLURM_ARRAY_TASK_ID:-''}"
. activate.sh
python src/train_wandb.py --sweep_id="${1}"
```

Run it as:

`sbatch script_name.sh sweep_id`

where `script_name.sh` is the filename with which you save the sbatch file, and `sweep_id` is the sweep ID
you get assigned by W&B.

You will need an `activate.sh` script which loads your environment, create a `logs_directory` for slurm logs,
and you might want to edit the `--job-name` and `--mail-user` sbatch parameters.

### Model checkpoints
You can find your trained models by accessing each separate model page in your sweep and obtaining its W&B run ID.
The saved checkpoint will be under `checkpoints/{run_id}`.