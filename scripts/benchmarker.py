#!/usr/bin/python3

import os
import subprocess
import re 

#Goal 1: benchmark time vs size

# QUESTIOn Benchmark in Python or make script to benchmark?????



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

    print(result_size, result_time)


if __name__ == "__main__":

    matrix_sizes = range(0,1000,100)

    for size in matrix_sizes:
        benchmark("dgemm",size, "-s")

