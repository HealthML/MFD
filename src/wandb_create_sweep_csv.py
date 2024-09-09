import wandb

from util.load_config import config
import argparse
from models import models
import pandas as pd
import torch

wandb.login()

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run_str', type=str, required=False, help='run id')
    p.add_argument('--run_txt', type=str, required=False, help='file with one run ID per line')
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

    if args.run_str is not None:
        assert args.run_txt is None, 'Error: specify either --run_str or --run_txt, not both.'
        run_str = args.run_str.split(',')
    elif args.run_txt is not None:
        run_str = []
        with open(args.run_txt, 'r') as infile:
            for l in infile:
                l = l.strip()
                if l == '':
                    continue
                else:
                    run_str.append(l)
    else:
        raise(ValueError('need to specify either --run_str or --run_txt'))

    runs = [wandb.Api().run(f"{entity}/{project}/{run}") for run in run_str]

    config_list = []
    for run in runs:
        config = {k: v for k,v in run.config.items() if not k.startswith('_')}
        config['ID'] = run.id
        try:
            config['Sweep'] = run.sweep_id
        except AttributeError:
            config['Sweep'] = ''
        config['Name'] = run.name
        config_list.append(config)


    result = pd.DataFrame(config_list)

    result.to_csv(args.out, sep=',',index=False)


    





