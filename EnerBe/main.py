import json
import argparse
import os
from subprocess import Popen, PIPE
import re 


class BenchMarker:
        '''
        The Parent class that holds all of the infromation from the run.
        '''
        def __init__(self):
                
                self.number = r'[+-]?((\d+\.\d*)|(\.\d+)|(\d+))([eE][+-]?\d+)?'

                self.modules = []

                self.sbatch_data = {}
                self.case_info = {}
                self.log = {}
                self.results = {
                    "NAME": [float('nan')],
                    "ALGO": [float('nan')],
                    "PRECISION": [float('nan')],
                    "OMP_THREADS": [float('nan')],
                    "GPU ID": [float('nan')],
                    "SIZE": [float('nan')],
                    "CPU_time": [float('nan')],
                    "GPU_time": [float('nan')],
                    "CPU_energy": [float('nan')],
                    "GPU_energy": [float('nan')],
                    "CPU_power": [float('nan')],
                    "GPU_power": [float('nan')],
                }

        def read_config(self, config_file_name):
                """
                Read JSON config file
                :param config_file_name: Name of the config file
                """
                f = open(config_file_name)
                config = json.load(f)

                self.modules = config['modules']
                self.sbatch_data = config['sbatch_data']
                self.case_info = config['case_info']

                f.close()

        def write_jobscript(self):

            job_string_text = "#!/bin/bash\n\n"
            job_string_text += "#SBATCH --partition=" + self.sbatch_data['partition'] + "\n"
            job_string_text += "#SBATCH --nodes=" + self.sbatch_data['nodes'] + "\n"
            job_string_text += "#SBATCH --ntasks=" + self.sbatch_data['ntasks'] + "\n"
            job_string_text += "#SBATCH --cpus-per-task=" + self.sbatch_data['cpus-per-task'] + "\n"
            job_string_text += "#SBATCH --time=" + self.sbatch_data['time'] + "\n"

            if self.sbatch_data['has_gpus']:
                job_string_text += "#SBATCH --gpus-per-node=" + self.sbatch_data['gpus-per-node'] + "\n"

            job_string_text += "\n"

            job_string_text += "module purge\n"
            for module in self.modules:
                job_string_text += "module load " + module + "\n"
            job_string_text += "\n"

            job_string_text += "benchmarker.py"


            f = open(self.sbatch_data["script_name"], "w")
            f.write(job_string_text)
            f.close()

        
        def run(self):

            application = self.case_info['application'][0]

            args = self.case_info['args']

            # get the path of this python script
            script_dir = os.path.dirname(os.path.realpath(__file__))

            bin_dir = script_dir + "/../bin" # this is a dirty solution
            executable = bin_dir + "/" + application

            if not os.path.isfile(executable):
                print("Executable....")
                print(executable)
                print("DOES NOT EXIST")
                exit(1)

            CMD = []
            CMD = self.sbatch_data['launcher'] + " " + executable
            for arg in args:
                CMD += " " + str(arg)
            print("Running this command ...")
            print("CMD: " + CMD)
            process = Popen(CMD.split(" "), stdout=PIPE, stderr=PIPE)

            output, error = process.communicate()
            output = output.decode('ISO-8859-1').strip()
            error = error.decode('ISO-8859-1').strip()

            with open("run.out", "w") as text_file:
                text_file.write(output)
            with open("run.err", "w") as text_file:
                text_file.write(error)

        def get_regex(self):
        
            nan = float('nan')
            cpu_result_time = nan
            cpu_result_watt = nan 
            cpu_result_joule = nan 
            gpu_result_time = nan
            gpu_result_watt = nan 
            gpu_result_joule = nan 

            with open("run.out", "r") as text_file:
                output = text_file.read()

            application = self.case_info['application'][0]
            number = self.number


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
            name_pattern = r'NAME:\s(?P<rname>\w*)'
            algo_pattern = r'ALGO:\s(?P<ralgo>\w*)'
            precision_pattern = r'PRECISION:\s(?P<rprecision>\d*)\sbytes'
            omp_threads_pattern = r'OMP\_THREADS:\s(?P<romp_threads>\d*)\s'
            gpu_id_pattern = r'GPU\sID:\s(?P<rgpu_id>\d*)\s'

            x = re.search(name_pattern, output,re.MULTILINE)
            self.results['NAME'] = [x['rname']]
            x = re.search(algo_pattern, output,re.MULTILINE)
            self.results['ALGO'] = [x['ralgo']]
            x = re.search(precision_pattern, output,re.MULTILINE)
            self.results['PRECISION'] = [x['rprecision']]
            x = re.search(omp_threads_pattern, output,re.MULTILINE)
            self.results['OMP_THREADS'] = [x['romp_threads']]
            x = re.search(gpu_id_pattern, output,re.MULTILINE)
            self.results['GPU_ID'] = [x['rgpu_id']]
            x = re.search(size_pattern, output,re.MULTILINE)
            self.results['SIZE'] = [x['rsize']]
            
            self.results["CPU_time"] = [cpu_result_time]
            self.results["GPU_time"] = [gpu_result_time]
            self.results["CPU_energy"] = [cpu_result_joule]
            self.results["GPU_energy"] = [gpu_result_joule]
            self.results["CPU_power"] = [cpu_result_watt]
            self.results["GPU_power"] = [gpu_result_watt]



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-j","--jobscript", help="Create a Jobscript based off info from 'bench_config.json'",action="store_true")
    parser.add_argument("-r","--run", help="Run the Benchmark",action="store_true")
    parser.add_argument("-c","--config", help="Pass specific .json config to script",type=str)
    args = parser.parse_args()


    benchmarker = BenchMarker()

    if args.config:
        config = args.config
        benchmarker.read_config(args.config)
    else:
        config = __file__.replace("main.py","bench_config.json")
        benchmarker.read_config(config)
        
    if args.jobscript:
        benchmarker.write_jobscript()
        exit(0)

    if args.run:

        #benchmarker.run()
        benchmarker.get_regex()
        print(benchmarker.results)
        
        #EnerBe.log_results()