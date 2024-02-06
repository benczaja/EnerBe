#!/usr/bin/env python3

import os
import subprocess
import re 
import pandas as pd

#Goal 1: benchmark time vs size

# QUESTIOn Benchmark in Python or make script to benchmark?????


def log_results(result):

    script_dir = os.path.dirname(os.path.realpath(__file__))
    results_dir = script_dir.replace("scripts","results")
    results_file = results_dir + "/results.csv"

    data = pd.DataFrame(result)


    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    if os.path.isfile(results_file):
        data_tmp = pd.read_csv(results_file)
        print("read in")
        print(data_tmp)
        data = pd.concat([data_tmp, data],ignore_index=True)
        data = data.sort_values(by='Name')
        print("data new")
        print(data)
        data.to_csv(results_file,sep=',',index=False)
    else:
        data.to_csv(results_file,sep=',',index=False)

    print(data)


def benchmark(executable, size, args=None, env_var=None):

    # get the path of this python script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    bin_dir = script_dir.replace("scripts","bin")
    
    executable = bin_dir + "/" + executable

    if not os.path.isfile(executable):
        print("Executable....")
        print(executable)
        print("DOES NOT EXIST")
        exit(1)

    # call the executable:
    if env_var:
        name = env_var.split("=")[0]
        value = env_var.split("=")[1]
        os.environ[name] = value
        output = subprocess.check_output([executable, str(size), str(args)]).decode("utf-8")
    else:
        output = subprocess.check_output([executable, str(size), str(args)]).decode("utf-8")
    
    # regex the results:
    number = r'[+-]?((\d+\.\d*)|(\.\d+)|(\d+))([eE][+-]?\d+)?'
    size_pattern = r'^SIZE:\s(?P<rsize>' + number + r')\s'
    time_pattern = r'^TIME:\s(?P<rtime>' + number + r')\ssec'

    x = re.search(size_pattern, output,re.MULTILINE)
    result_size = x['rsize']

    x = re.search(time_pattern, output,re.MULTILINE)
    result_time = x['rtime']

    nan = float('nan')
    # save the results
    result = {
    "Name": [executable.split("/")[-1]],
    "Args": [args],
    "env_var": [env_var],
    "Size": [result_size],
    "Total_time": [result_time],
    "CPU_time": [nan],
    "GPU_time": [nan],
    "Total_energy": [nan],
    "CPU_energy": [nan],
    "GPU_energy": [nan],

    }

    log_results(result)


if __name__ == "__main__":

    matrix_sizes = range(0,1800,100)

    for size in matrix_sizes:
        benchmark("dgemm",size, "-p", env_var="OMP_NUM_THREADS=8")

