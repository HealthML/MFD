import subprocess
import time as tme
from .load_config import config
import datetime
import os

# run bash commands and return output as 2d list of strings---------------------
def bash_command(cmd, as_list=True):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    output = p.communicate()[0].decode('utf-8')
    if as_list:
        return [line.split() for line in output.splitlines()]
    return output

def allocate_run(bash_script, partition='hpcpu,vcpu', cpus_per_task=8, gpus=0, mem='8gb', time='3-00:00:00', job_name='sweep_%j_run_cpu', output='slogs/%x_%A_%a_sweep_run_cpu_logs.out'):
    if config['cluster'] == 'dhclab':
        allocate_run_dhclab(bash_script, partition, cpus_per_task, gpus, mem, time, job_name, output)
    elif config['cluster'] == 'max':
        allocate_run_max(bash_script, partition, cpus_per_task, gpus, mem, time, job_name, output)
    else:
        raise Exception(f"cluster {config['cluster']} not supported")

def allocate_run_dhclab(bash_script, partition='hpcpu,vcpu', cpus_per_task=8, gpus=0, mem='8gb', time='3-00:00:00', job_name='sweep_%j_run_cpu', output='slogs/%x_%A_%a_sweep_run_cpu_logs.out'):
    bash_script_file = f"""#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --output={output}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gpus={gpus}
#SBATCH --mem={mem}
#SBATCH --time={time}

eval "$(conda shell.bash hook)"
conda activate {config['conda_environment']}

{bash_script}
    """
    # create tmp bash script file 
    tmp_bash_script_name = 'tmp_bash_script.sh'
    with open(tmp_bash_script_name, 'w') as f:
        f.write(bash_script_file)
    
    # remoteRunHistory
    history_folder = "remoteRunHistory"
    if not os.path.exists(history_folder):
        os.makedirs(history_folder)
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M")
    with open(f"{history_folder}/{date_string}.sh", 'w') as f:
        f.write(bash_script_file)
    
    tme.sleep(2)

    # allocate and run bash script
    bash_command(cmd=f'sbatch {tmp_bash_script_name}', as_list=False)

    tme.sleep(2)

    # remove tmp bash script file
    bash_command(cmd=f'rm {tmp_bash_script_name}', as_list=False)

    confirmation_message = f'\nJob submitted!\n\n--------------------------\n{bash_script_file}\n--------------------------\n\nFind logs in {output}'
    print(confirmation_message)


def dhclab_mem_param_to_max_mem_param(dhclab_mem_param):
    n_gigabytes = int(dhclab_mem_param.split('gb')[0])
    return f'{n_gigabytes}G'

def dhclab_time_param_to_max_time_param(dhclab_time_param):
    n_days = int(dhclab_time_param.split('-')[0])
    n_hours = int(dhclab_time_param.split('-')[1].split(':')[0])
    n_minutes = int(dhclab_time_param.split('-')[1].split(':')[1])
    return f'{n_days*24 + n_hours}:{n_minutes}:00'

def allocate_run_max(bash_script, partition='gpupro,gpua100', cpus_per_task=4, gpus=0, mem='4gb', time='3-00:00:00', job_name='sweep_run', output='slogs/sweep_run_gpu_logs.out'):
    """
    this is a hacky solution that translates the dhclab slurm commands to max qsub commands
    please do it better, by inventing an interface that works for both clusters
    for now - hardcoded h_gpu=1
    """
    
    bash_script_file = f"""#!/bin/bash

#start from current directory
#$ -V #pass conda env variables to the GridEngine coz it is stupid
#$ -cwd
#$ -l m_mem_free={dhclab_mem_param_to_max_mem_param(mem)}
#$ -l h_gpu=1
{f'#$ -l h_gpu={gpus}' if gpus > 0 else ''}
#$ -l h=maxg02|maxg03|maxg04|maxg05|maxg06|maxg07|maxg08
#$ -pe smp {cpus_per_task}
#$ -l h_rt={dhclab_time_param_to_max_time_param(time)}
#$ -o {config['max_cluster_repo_directory']}/{output}
#$ -N {job_name} #name

eval "$(conda shell.bash hook)"
conda activate {config['conda_environment']}

{bash_script}
    """
    # create tmp bash script file 
    tmp_bash_script_name = 'tmp_bash_script.sh'
    with open(tmp_bash_script_name, 'w') as f:
        f.write(bash_script_file)
    
    tme.sleep(2)

    # allocate and run bash script
    bash_command(cmd=f'qsub {tmp_bash_script_name}', as_list=False)

    tme.sleep(2)

    # remove tmp bash script file
    bash_command(cmd=f'rm {tmp_bash_script_name}', as_list=False)

    confirmation_message = f'\nJob submitted!\n\n--------------------------\n{bash_script_file}\n--------------------------\n\nFind logs in {output}'
    print(confirmation_message)