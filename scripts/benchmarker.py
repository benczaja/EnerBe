#!/usr/bin/env python3

import os
import subprocess
import re 
import pandas as pd

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


def benchmark(application, size, args=None, env_var=None,cluster=False):

    nan = float('nan')
    result_size = nan
    cpu_result_time = nan
    cpu_result_watt = nan 
    cpu_result_joule = nan 
    gpu_result_time = nan
    gpu_result_watt = nan 
    gpu_result_joule = nan 

    print("Running...\nApplication: " + application +
          " Size: " + str(size) +
          " Args: " + str(args) +
          " Vars: " + str(env_var) 
          )
    
    # get the path of this python script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    bin_dir = script_dir.replace("scripts","bin")
    executable = bin_dir + "/" + application

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
        output = subprocess.run([executable, str(size), str(args)], capture_output=True).stdout.decode("utf-8")
    else:
        output = subprocess.run([executable, str(size), str(args)], capture_output=True).stdout.decode("utf-8")
        

    # regex the results:
    if ("pmt" in application) & ("gpu" in application):
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
    "Name": [application],
    "Args": [args],
    "env_var": [env_var],
    "Size": [result_size],
    "CPU_time": [cpu_result_time],
    "GPU_time": [gpu_result_time],
    "CPU_energy": [cpu_result_joule],
    "GPU_energy": [gpu_result_joule],
    "CPU_power": [cpu_result_watt],
    "GPU_power": [gpu_result_watt],
    }

    print(cpu_result_time, gpu_result_time)

    log_results(result,cluster)


if __name__ == "__main__":


    applications = [
        "dgemm_pmt",
        "dgemm_pmt_gpu",
        #"saxpy",
        #"daxpy",
        ]
    
    matrix_sizes = range(0,400,200)
    args = ["-s", "-p"]
    
    env_vars = []
    #    "OMP_NUM_THREADS=2",
    #    "OMP_NUM_THREADS=4",
    #    "OMP_NUM_THREADS=8",
    #    "OMP_NUM_THREADS=16",
    #    "OMP_NUM_THREADS=32",
    #    ]
    
    for application in applications:
        for size in matrix_sizes: 
            for arg in args:
                if arg.count("s"):
                    benchmark(application, size, arg,cluster=False)
                else:
                    for var in env_vars:
                        benchmark(application, size, arg, var,cluster=False)


