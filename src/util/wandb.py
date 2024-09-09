from .load_config import config
import json
import yaml
import wandb


def read_config(file):
    assert isinstance(file, str)
    # read a config file
    if file.lower().endswith('.json'): 
        with open(file, 'r') as f:
            sweep_configuration = json.load(f)
    elif file.lower().endswith('.yaml') or file.lower().endswith('.yml'):
        with open(file, 'r') as f:
            sweep_configuration = yaml.safe_load(f)
    else:
        raise ValueError(f'cannot determine file format from ending: {file}, should be one of .json/.yaml/.yml')
    return sweep_configuration


def create_sweep(sweep_config_file=None):

    if sweep_config_file is None:
        sweep_config_file = config['wandb']['sweep_config_file']

    sweep_configuration = read_config(sweep_config_file)
        
    sweep_id = wandb.sweep(sweep=sweep_configuration,
                           entity = config['wandb']['entity'] if 'entity' in config['wandb'] else None,
                           project=config['wandb']['project']
                           )

    with open(config['wandb']['current_sweep_id_file'], 'w') as f:
        f.write(sweep_id)

    return sweep_id


def current_sweep_id():

    try:
        with open(config['wandb']['current_sweep_id_file']) as f:
            sweep_id = f.read()
    except FileNotFoundError as e:
        print('current-sweep id file ' + config['wandb']['current_sweep_id_file'] + ' does not exist.')
        raise e
    assert isinstance(sweep_id, str)
    assert len(sweep_id) == 8
    return sweep_id
