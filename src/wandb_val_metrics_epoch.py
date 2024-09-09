import wandb

from util.load_config import config
import argparse

wandb.login()


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=str, required=True, help='run id')
    p.add_argument('--out', type=str, required=True, help='output tsv')
    args = p.parse_args()
    return args


if __name__ == '__main__':

    try:
        project = config['wandb']['project']
        entity = config['wandb'].get('entity', config['wandb']['username'])
    except KeyError as e:
        print('KeyError while parsing config file. Make sure you have wandb configured in src/config.yaml')
        raise e

    args = get_args()

    run = wandb.Api().run(f"{entity}/{project}/{args.run}")

    keys = ['epoch', 'val/loss', 'val/loss_pred', 'val/loss_cov', 'val/auroc', 'val/loss_epoch']
    keys_avail = list((k for k in keys if k in run.history().columns))

    # better to query history rather than summary - runs that crashed (e.g., because of timeout) won't have summary
    result = run.history(samples=100000, keys=keys_avail)
    result = result.rename(columns={c: c.replace('_epoch', '') for c in result.columns if 'val/' in c})

    result.to_csv(args.out, sep='\t')
