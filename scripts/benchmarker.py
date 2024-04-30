#!/usr/bin/env python3

import os
import subprocess
from subprocess import Popen, PIPE
import re 
import pandas as pd
import json
import argparse
import pdb


# define the global regex for a number
number = r'[+-]?((\d+\.\d*)|(\.\d+)|(\d+))([eE][+-]?\d+)?'

def log_results(result,cluster=False):

    script_dir = os.path.dirname(os.path.realpath(__file__))
    results_dir = script_dir.replace("scripts","tmp_results")
    if cluster:
        jobid = os.environ['SLURM_JOB_ID']
        print(jobid)
        results_file = results_dir + "/results_" + jobid +".csv"
    else:
        results_file = results_dir + "/results_tmp.csv"


    data = pd.DataFrame(result)

    #get CPU information
    output = subprocess.check_output(['lscpu']).decode("utf-8")

    pattern = r'^Model\sname:\s*(?P<cpuname>.*)'
    x = re.search(pattern, output,re.MULTILINE)
    cpu_name = x['cpuname']

    pattern = r'Core\(s\) per socket:\s*(?P<cps>.*)'
    x = re.search(pattern, output,re.MULTILINE)
    cores_per_socket = x['cps']

    pattern = r'Socket\(s\):\s*(?P<sockets>.*)'
    x = re.search(pattern, output,re.MULTILINE)
    sockets = x['sockets']

    data['CPU_name'] = cpu_name
    data['Sockets'] = sockets
    data['Cores_per_socket'] = cores_per_socket


    # get DEVICE information
    #NVIDIA
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader']).decode("utf-8")
        data["GPU_name"] = output.split("\n")[0]
    except:
        # AMD
        try:
            output = subprocess.check_output(['rocm-smi', '--showproductname']).decode("utf-8")

            pattern = r'Card series:\s*(?P<gpuname>.*)'
            x = re.search(pattern, output,re.MULTILINE)
            data["GPU_name"] = x['gpuname']

        except:
            data["GPU_name"] = float("nan")
    
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    if os.path.isfile(results_file):
        data_tmp = pd.read_csv(results_file)
        data = pd.concat([data_tmp, data],ignore_index=True)
        data = data.sort_values(by='Name')
        data.to_csv(results_file,sep=',',index=False)
    else:
        data.to_csv(results_file,sep=',',index=False)

    return(results_file)


def get_regex(config):

    nan = float('nan')
    result_size = nan
    cpu_result_time = nan
    cpu_result_watt = nan 
    cpu_result_joule = nan 
    gpu_result_time = nan
    gpu_result_watt = nan 
    gpu_result_joule = nan 

    with open("run.out", "r") as text_file:
        output = text_file.read()

    application = config['case_info']['application']

    # regex the results:
    if ("pmt" in application) & (("cuda" in application) | ("hip" in application)):
        gpu_time_pattern = r'GPU_TIME:\s(?P<rgtime>' + number + r')\ss'
        gpu_watt_pattern = r'GPU_WATTS:\s(?P<rgwatt>' + number + r')\sW'
        gpu_joule_pattern = r'GPU_JOULES:\s(?P<rgjoule>' + number + r')\sJ'
        
        cpu_time_pattern = r'CPU_TIME:\s(?P<rctime>' + number + r')\s'
        cpu_watt_pattern = r'CPU_WATTS:\s(?P<rcwatt>' + number + r')\s'
        cpu_joule_pattern = r'CPU_JOULES:\s(?P<rcjoule>' + number + r')\s'

        try:
            x = re.search(gpu_time_pattern, output,re.MULTILINE)
            gpu_result_time = x['rgtime']
        except TypeError as error:
            print(error)
            print("Could not find REGEX match for GPU_TIME: in output")
            exit(1)

        try:
            x = re.search(gpu_watt_pattern, output,re.MULTILINE)
            gpu_result_watt = x['rgwatt']
        except TypeError as error:
            print(error)
            print("Could not find REGEX match for GPU_WATTS: in output")
            exit(1)

        try:
            x = re.search(gpu_joule_pattern, output,re.MULTILINE)
            gpu_result_joule = x['rgjoule']
        except TypeError as error:
            print(error)
            print("Could not find REGEX match for GPU_JOULES: in output")
            exit(1)

        try:
            x = re.search(cpu_time_pattern, output,re.MULTILINE)
            cpu_result_time = x['rctime']
        except TypeError as error:
            print(error)
            print("Could not find REGEX match for CPU_TIME: in output")
            exit(1)

        try:
            x = re.search(cpu_watt_pattern, output,re.MULTILINE)
            cpu_result_watt = x['rcwatt']
        except TypeError as error:
            print(error)
            print("Could not find REGEX match for CPU_WATTS: in output")
            exit(1)

        try:
            x = re.search(cpu_joule_pattern, output,re.MULTILINE)
            cpu_result_joule = x['rcjoule']
        except TypeError as error:
            print(error)
            print("Could not find REGEX match for CPU_JOULES: in output")
            exit(1)


    elif ("pmt" in application):
        cpu_time_pattern = r'CPU_TIME:\s(?P<rctime>' + number + r')\ss'
        cpu_watt_pattern = r'CPU_WATTS:\s(?P<rcwatt>' + number + r')\sW'
        cpu_joule_pattern = r'CPU_JOULES:\s(?P<rcjoule>' + number + r')\sJ'

        try:
            x = re.search(cpu_time_pattern, output,re.MULTILINE)
            cpu_result_time = x['rctime']
        except TypeError as error:
            print(error)
            print("Could not find REGEX match for CPU_TIME: in output")
            exit(1)

        try:
            x = re.search(cpu_watt_pattern, output,re.MULTILINE)
            cpu_result_watt = x['rcwatt']
        except TypeError as error:
            print(error)
            print("Could not find REGEX match for CPU_WATTS: in output")
            exit(1)

        try:
            x = re.search(cpu_joule_pattern, output,re.MULTILINE)
            cpu_result_joule = x['rcjoule']
        except TypeError as error:
            print(error)
            print("Could not find REGEX match for CPU_JOULES: in output")
            exit(1)
    else:
        time_pattern = r'TIME:\s(?P<rtime>' + number + r')\ss'
        x = re.search(time_pattern, output,re.MULTILINE)
        cpu_result_time = x['rtime']

    
    # regex the results:
    size_pattern = r'SIZE:\s(?P<rsize>' + number + r')\s'

    x = re.search(size_pattern, output,re.MULTILINE)
    result_size = x['rsize']

    # save the results
    result = {
    "Name": config['case_info']['application'],
    "Args": config['case_info']['args'],
    "env_vars": config['case_info']['env_vars'],
    "Size": [result_size],
    "CPU_time": [cpu_result_time],
    "GPU_time": [gpu_result_time],
    "CPU_energy": [cpu_result_joule],
    "GPU_energy": [gpu_result_joule],
    "CPU_power": [cpu_result_watt],
    "GPU_power": [gpu_result_watt],
    }

    print(cpu_result_time, gpu_result_time)

    return(result)

def run(config):

    application = config['case_info']['application'][0]
    
    args = config['case_info']['args']

    # get the path of this python script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    bin_dir = script_dir.replace("scripts","bin")
    executable = bin_dir + "/" + application

    if not os.path.isfile(executable):
        print("Executable....")
        print(executable)
        print("DOES NOT EXIST")
        exit(1)

    CMD = []
    CMD = config['sbatch_data']['launcher'] + " " + executable
    print(args)
    for arg in args:
        CMD += " " + str(arg)
    print("Running this command ...")
    print("CMD: " + CMD)
    process = Popen(CMD.split(" "), stdout=PIPE, stderr=PIPE)

    output, error = process.communicate()
    output = output.decode('utf-8').strip()
    error = error.decode('ISO-8859-1').strip()

    with open("run.out", "w") as text_file:
        text_file.write(output)
    with open("run.err", "w") as text_file:
        text_file.write(error)
    
    #return(output)


def read_config():
    f = open('bench_config.json')
    config = json.load(f)
    return(config)

def create_jobscript(config):

    script_dir = os.path.dirname(os.path.realpath(__file__))


    job_string_text = "#!/bin/bash\n\n"

    job_string_text += "#SBATCH --partition=" + config['sbatch_data']['partition'] + "\n"
    job_string_text += "#SBATCH --nodes=" + config['sbatch_data']['nodes'] + "\n"
    job_string_text += "#SBATCH --ntasks=" + config['sbatch_data']['ntasks'] + "\n"
    job_string_text += "#SBATCH --cpus-per-task=" + config['sbatch_data']['cpus-per-task'] + "\n"
    job_string_text += "#SBATCH --time=" + config['sbatch_data']['time'] + "\n"

    if config['sbatch_data']['has_gpus']:
        job_string_text += "#SBATCH --gpus-per-node=" + config['sbatch_data']['gpus-per-node'] + "\n"

    job_string_text += "\n"

    job_string_text += "module purge\n"
    for module in config["modules"]:
        job_string_text += "module load " + module + "\n"
    job_string_text += "\n"

    job_string_text += script_dir + "/benchmarker.py"


    f = open(config['sbatch_data']["script_name"], "w")
    f.write(job_string_text)
    f.close()

def benchmark(config):

    run(config)
    result = get_regex(config)
    log_results(result,config['case_info']['sbatch_job'])



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--jobscript", help="Create a Jobscript based off info from 'bench_config.json'",action="store_true")
    args = parser.parse_args()

    config = read_config()

    if args.jobscript:
        create_jobscript(config)
    else:
        benchmark(config)




