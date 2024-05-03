import json
import argparse
import os
import subprocess
from subprocess import Popen, PIPE
import re
import pandas as pd


class BenchMarker:
        '''
        The Parent class that holds all of the infromation from the run.
        '''
        def __init__(self):
                
                # One REGEX to rule them all
                self.regex_number = r'[+-]?((\d+\.\d*)|(\.\d+)|(\d+))([eE][+-]?\d+)?'

                self.EnerBe_root_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])
                # These will be picked up by the config
                self.modules = []
                self.sbatch_data = {}
                self.case_info = {}

                # These will be picked up when running the applications                
                self.arch_info = {
                    "CPU_NAME": [float('nan')],
                    "Sockets": [float('nan')],
                    "Sockets": [float('nan')],
                    "GPU_NAME": [float('nan')],
                }
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

            bin_dir = self.EnerBe_root_dir + "/bin"
            executable = bin_dir + "/" + application

            if not os.path.isfile(executable):
                print("Executable....")
                print(executable)
                print("DOES NOT EXIST")
                exit(1)

            # some dumb logic to encorperate no launcher
            if self.sbatch_data['launcher'] == "bash":
                CMD = executable
            else:
                CMD = self.sbatch_data['launcher'] + " " + executable

            for arg in self.case_info['args']:
                CMD += " " + arg

            print("Running this command ...")
            print(CMD.split(" "))
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

            # regex the results:
            if ("pmt" in application) & (("cuda" in application) | ("hip" in application)):
                gpu_time_pattern = r'GPU_TIME:\s(?P<rgtime>' + self.regex_number + r')\ss'
                gpu_watt_pattern = r'GPU_WATTS:\s(?P<rgwatt>' + self.regex_number + r')\sW'
                gpu_joule_pattern = r'GPU_JOULES:\s(?P<rgjoule>' + self.regex_number + r')\sJ'

                cpu_time_pattern = r'CPU_TIME:\s(?P<rctime>' + self.regex_number + r')\s'
                cpu_watt_pattern = r'CPU_WATTS:\s(?P<rcwatt>' + self.regex_number + r')\s'
                cpu_joule_pattern = r'CPU_JOULES:\s(?P<rcjoule>' + self.regex_number + r')\s'

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
                cpu_time_pattern = r'CPU_TIME:\s(?P<rctime>' + self.regex_number + r')\ss'
                cpu_watt_pattern = r'CPU_WATTS:\s(?P<rcwatt>' + self.regex_number + r')\sW'
                cpu_joule_pattern = r'CPU_JOULES:\s(?P<rcjoule>' + self.regex_number + r')\sJ'

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
                time_pattern = r'TIME:\s(?P<rtime>' + self.regex_number + r')\ss'
                x = re.search(time_pattern, output,re.MULTILINE)
                cpu_result_time = x['rtime']

            # regex the results:
            size_pattern = r'SIZE:\s(?P<rsize>' + self.regex_number + r')\s'
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

        def get_architecture(self):

        
            #get CPU information
            output = subprocess.check_output(['lscpu']).decode("utf-8")
        
            pattern = r'^Model\sname:\s*(?P<cpuname>.*)'
            x = re.search(pattern, output,re.MULTILINE)
            self.arch_info['CPU_NAME'] = x['cpuname']
        
            pattern = r'Core\(s\) per socket:\s*(?P<cps>.*)'
            x = re.search(pattern, output,re.MULTILINE)
            self.arch_info['Cores_per_socket'] = x['cps']
        
            pattern = r'Socket\(s\):\s*(?P<sockets>.*)'
            x = re.search(pattern, output,re.MULTILINE)
            self.arch_info['Sockets'] = x['sockets']
        
            # get DEVICE information
            #NVIDIA
            try:
                output = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader']).decode("utf-8")
                self.arch_info["GPU_NAME"] = output.split("\n")[0]
                
            except:
                # AMD
                try:
                    output = subprocess.check_output(['rocm-smi', '--showproductname']).decode("utf-8")
        
                    pattern = r'Card series:\s*(?P<gpuname>.*)'
                    x = re.search(pattern, output,re.MULTILINE)
                    self.arch_info["GPU_NAME"] = x['gpuname']
        
                except:
                    self.arch_info["GPU_NAME"] = float("nan")



        def to_csv(self):
            
            results_dir = __file__.replace("main.py","tmp_results")

            if not os.path.exists(results_dir):
                os.mkdir(results_dir)

            if self.case_info["sbatch_job"]:
                jobid = os.environ['SLURM_JOB_ID']
                results_file = results_dir + "/results_" + jobid +".csv"
            else:
                results_file = results_dir + "/tmp_results.csv"

            out_data = self.results
            out_data.update(self.arch_info)
            
            out_data = pd.DataFrame(out_data)
            
            if os.path.isfile(results_file):
                print("Moving existing tmp_results to tmp_results.OLD.csv")
                subprocess.check_output(["mv",results_file, results_file.replace(".csv", ".OLD.csv")])
            
            out_data.to_csv(results_file,sep=',',index=False)
            print("Wrote results to " + results_file)


        def concatonate_csvs(self,temp_result_csvs):

            results_dir = os.path.dirname(os.path.realpath(temp_result_csvs[0])) + "/"

            count = 0 
            for filename in temp_result_csvs:
                try:
                    data_tmp = pd.read_csv(filename)

                    if count ==0:
                        data = data_tmp
                        count += 1
                    else:
                        data = pd.concat([data_tmp, data],ignore_index=True)
                        data = data.sort_values(by='NAME')
                        count += 1
                    print("Appending... " + filename)

                except FileNotFoundError:
                    print(FileNotFoundError)

            if "tmp_results" in results_dir:
                data.to_csv(results_dir + "results.csv",sep=',',index=False)
                print("All .csvs appended to: \n" + results_dir + "results.csv")
            else:
                "tmp_results directory not found"
                exit(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-j","--jobscript", help="Create a Jobscript based off info from 'bench_config.json'",action="store_true")
    parser.add_argument("-r","--run", help="Run the Benchmark",action="store_true")
    parser.add_argument("-c","--config", help="Pass specific .json config to script",type=str)
    parser.add_argument("--concatonate",metavar='N', type=str, nargs='*', help="Concatonate multiple tmp_results.csv together")

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
    if args.concatonate:
        csvs = args.concatonate
        benchmarker.concatonate_csvs(csvs)
        exit(0)

    if args.run:

        benchmarker.run()
        benchmarker.get_regex()
        benchmarker.get_architecture()
        benchmarker.to_csv()