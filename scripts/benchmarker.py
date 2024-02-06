#!/usr/bin/env python3

import os
import subprocess
import re 
import pandas as pd

# define the global regex for a number
number = r'[+-]?((\d+\.\d*)|(\.\d+)|(\d+))([eE][+-]?\d+)?'
#Goal 1: benchmark time vs size

# QUESTIOn Benchmark in Python or make script to benchmark?????


def log_results(result):

    script_dir = os.path.dirname(os.path.realpath(__file__))
    results_dir = script_dir.replace("scripts","results")
    results_file = results_dir + "/results.csv"

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
    

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    if os.path.isfile(results_file):
        data_tmp = pd.read_csv(results_file)
        data = pd.concat([data_tmp, data],ignore_index=True)
        data = data.sort_values(by='Name')
        data.to_csv(results_file,sep=',',index=False)
    else:
        data.to_csv(results_file,sep=',',index=False)



def benchmark(executable, size, args=None, env_var=None):

    print("Running...\nApplication: " + executable +
          " Size: " + str(size) +
          " Args: " + str(args) +
          " Vars: " + str(env_var) 
          )
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

    applications = [
        "dgemm",
        "sgemm",
        "saxpy",
        "daxpy",
        ]
    
    matrix_sizes = range(0,1000,100)

    args = ["-s", "-p"]

    env_vars = [
        "OMP_NUM_THREADS=2",
        "OMP_NUM_THREADS=4",
        "OMP_NUM_THREADS=8",
        "OMP_NUM_THREADS=16",
        "OMP_NUM_THREADS=32",
        ]

    for application in applications:
        for size in matrix_sizes: 
            for arg in args:
                if arg.count("p"):
                    benchmark(application, size, arg)
                else:
                    for var in env_vars:
                        benchmark(application, size, arg, var)


