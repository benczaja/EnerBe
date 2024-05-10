import json
import argparse
import os
import subprocess
from subprocess import Popen, PIPE
import re
import pandas as pd
import shutil

from plot import Plotter


class BenchMarker:
        '''
        The Parent class that holds all of the infromation from the run.
        '''
        def __init__(self):
                
                # One REGEX to rule them all
                self.regex_number = '[+-]?((\d+\.\d*)|(\.\d+)|(\d+))([eE][+-]?\d+)?'

                self.EnerBe_root_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])
                self.EnerBe_log_dir = self.EnerBe_root_dir + "/EnerBe/" + "log"
                self.EnerBe_sbatch_dir = self.EnerBe_root_dir + "/EnerBe/" + "sbatch"
                # These will be picked up by the config
                self.modules = []
                self.sbatch_data = {}
                self.case_info = {}

                self.tmp_out_file = "run.out"
                self.tmp_err_file = "run.err"

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
                    "MPI_RANKS": [float('nan')],
                    "NGPUs": [float('nan')],
                    "GPU ID": [float('nan')],
                    "SIZE": [float('nan')],

                    "TOTAL_TIME": [float('nan')],
                    "TOTAL_TIME_var": [float('nan')],
                    "TOTAL_TIME_std": [float('nan')],
                    
                    "CPU_TIME": [float('nan')],
                    "CPU_TIME_var": [float('nan')],
                    "CPU_TIME_std": [float('nan')],
                    
                    "GPU_TIME": [float('nan')],
                    "GPU_TIME_var": [float('nan')],
                    "GPU_TIME_std": [float('nan')],
                    
                    "CPU_JOULES": [float('nan')],
                    "CPU_JOULES_var": [float('nan')],
                    "CPU_JOULES_std": [float('nan')],
                    
                    "GPU_JOULES": [float('nan')],
                    "GPU_JOULES_var": [float('nan')],
                    "GPU_JOULES_std": [float('nan')],

                    "CPU_WATTS": [float('nan')],
                    "CPU_WATTS_var": [float('nan')],
                    "CPU_WATTS_std": [float('nan')],

                    "GPU_WATTS": [float('nan')],
                    "GPU_WATTS_var": [float('nan')],
                    "GPU_WATTS_std": [float('nan')],
                    "NRUNS": [float('nan')],
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

            self.clean_logs()

            input_parameters = self.case_info["input_parameters"]
            
            if not os.path.exists(self.EnerBe_sbatch_dir):
                print("Making dir: " + self.EnerBe_sbatch_dir )
                os.mkdir(self.EnerBe_sbatch_dir)

            for input_parameter in input_parameters:
                tmp_idx = input_parameters.index(input_parameter)

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

                job_string_text += "python " + self.EnerBe_root_dir + "/EnerBe/main.py --run " + input_parameter


                batch_file = self.EnerBe_sbatch_dir + "/" + self.sbatch_data["script_name"]
                batch_file = batch_file.replace(".sh", "." + str(tmp_idx) + ".sh")
                
                f = open(batch_file, "w")
                f.write(job_string_text)
                f.close()

        def launch_jobscript(self):

            for input_parameter in self.case_info["input_parameters"]:
                tmp_idx = self.case_info["input_parameters"].index(input_parameter)
                batch_file = self.EnerBe_sbatch_dir + "/" + self.sbatch_data["script_name"]
                batch_file = batch_file.replace(".sh", "." + str(tmp_idx) + ".sh")

                print("Launching Jobscript: ")
                output = subprocess.check_output([
                    'sbatch',
                    '-a 1-' + self.sbatch_data['num_jobs'],
                    '--output='+self.EnerBe_log_dir +"/slurmjob.%j.out",
                    '--error='+self.EnerBe_log_dir +"/slurmjob.%j.err",
                    batch_file]).decode("utf-8")
                print(batch_file)

        def clean_logs(self):
            
            if not os.path.exists(self.EnerBe_log_dir):
                os.mkdir(self.EnerBe_log_dir)
            else:
                shutil.rmtree(self.EnerBe_log_dir, ignore_errors=False)
                os.mkdir(self.EnerBe_log_dir)

            if not os.path.exists(self.EnerBe_sbatch_dir):
                os.mkdir(self.EnerBe_sbatch_dir)
            else:
                shutil.rmtree(self.EnerBe_sbatch_dir, ignore_errors=False)
                os.mkdir(self.EnerBe_sbatch_dir)
        
        def run(self,param):

            if not os.path.exists(self.EnerBe_log_dir):
                os.mkdir(self.EnerBe_log_dir)

            tmp_idx = self.case_info['input_parameters'].index(param)

            application = self.case_info['application'][0]

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

            if len(self.case_info['args']) >0:
                for arg in self.case_info['args']:
                    CMD += " " + arg
            CMD += " " + param

            print("Running this command ...")
            print(CMD.split(" "))
            process = Popen(CMD.split(" "), stdout=PIPE, stderr=PIPE)

            output, error = process.communicate()
            output = output.decode('ISO-8859-1').strip()
            error = error.decode('ISO-8859-1').strip()

            try:                
                jobid = os.environ['SLURM_JOB_ID']
                self.tmp_out_file = self.tmp_out_file.replace(".out", "." + str(jobid) + ".out")
                self.tmp_err_file = self.tmp_err_file.replace(".err", "." + str(jobid) + ".err")
            except:
                self.tmp_out_file = self.tmp_out_file.replace(".out", "." + str(tmp_idx) + ".out")
                self.tmp_err_file = self.tmp_err_file.replace(".err", "." + str(tmp_idx) + ".err")



            print("logging out/err to " + self.EnerBe_log_dir)
            with open(self.EnerBe_log_dir + "/" +self.tmp_out_file, "w") as text_file:
                text_file.write(output)
            with open(self.EnerBe_log_dir + "/" + self.tmp_err_file, "w") as text_file:
                text_file.write(error)

        def get_regex(self):
        
            with open(self.EnerBe_log_dir + "/" + self.tmp_out_file, "r") as text_file:
                output = text_file.read()

            application = self.case_info['application'][0]

            # regex the results:
            if ("pmt" in application) & (("cuda" in application) | ("hip" in application)):

                device_types = ["CPU_", "GPU_"] # I am sorry for the _    
                patterns = [
                    "TIME",
                    "TIME_var",
                    "TIME_std",
                    "WATTS", 
                    "WATTS_var", 
                    "WATTS_std", 
                    "JOULES",
                    "JOULES_var",
                    "JOULES_std"
                ]

            elif ("pmt" in application):
                device_types = ["CPU_"]  # I am sorry for the _ 
                patterns = [
                    "TIME",
                    "TIME_var",
                    "TIME_std",
                    "WATTS", 
                    "WATTS_var", 
                    "WATTS_std", 
                    "JOULES",
                    "JOULES_var",
                    "JOULES_std"
                ]
            else:
                device_types = ["TOTAL_"] 
                patterns = [
                    "TIME",
                    "TIME_var",
                    "TIME_std",
                ]
            
            for device_type in device_types:
                for pattern in patterns:
                    tmp_string =  device_type + pattern + ":\s(?P<group>" + self.regex_number + ")\s"
                    tmp_string = f'({tmp_string})'

                    try:
                        x = re.search(tmp_string, output,re.MULTILINE)
                        self.results[device_type + pattern] = [x['group']]
                    except TypeError as error:
                        print(error)
                        print("Could not find REGEX match for "+ device_type + pattern +": in " + self.tmp_out_file)
                        exit(1)

            # regex the results:
            name_pattern = r'NAME:\s(?P<rname>\w*)'
            algo_pattern = r'ALGO:\s(?P<ralgo>\w*)'
            precision_pattern = r'PRECISION:\s(?P<rprecision>\d*)\sbytes'
            omp_threads_pattern = r'OMP\_THREADS:\s(?P<romp_threads>\d*)\s'
            mpi_ranks_pattern = r'MPI\_RANKS:\s(?P<rmpi_ranks>\d*)\s'
            ngpus_pattern = r'NGPUs:\s(?P<rngpus>\d*)\s'
            gpu_id_pattern = r'GPU\sID:\s(?P<rgpu_id>\d*)\s'
            size_pattern = r'SIZE:\s(?P<rsize>' + self.regex_number + r')\s'
            nruns_pattern = r'NRUNS:\s(?P<rnruns>' + self.regex_number + r')'

            x = re.search(name_pattern, output,re.MULTILINE)
            self.results['NAME'] = [x['rname']]
            x = re.search(algo_pattern, output,re.MULTILINE)
            self.results['ALGO'] = [x['ralgo']]
            x = re.search(precision_pattern, output,re.MULTILINE)
            self.results['PRECISION'] = [x['rprecision']]
            x = re.search(omp_threads_pattern, output,re.MULTILINE)
            self.results['OMP_THREADS'] = [x['romp_threads']]
            x = re.search(mpi_ranks_pattern, output,re.MULTILINE)
            self.results['MPI_RANKS'] = [x['rmpi_ranks']]
            x = re.search(ngpus_pattern, output,re.MULTILINE)
            self.results['NGPUs'] = [x['rngpus']]
            x = re.search(gpu_id_pattern, output,re.MULTILINE)
            self.results['GPU ID'] = [x['rgpu_id']]
            x = re.search(size_pattern, output,re.MULTILINE)
            self.results['SIZE'] = [x['rsize']]
            x = re.search(nruns_pattern, output,re.MULTILINE)
            self.results['NRUNS'] = [x['rnruns']]
            

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

            try:                
                jobid = os.environ['SLURM_JOB_ID']
                results_file = results_dir + "/results_" + jobid +".csv"
            except:
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

    parser.add_argument("-c","--config", help="Pass specific .json config to script",type=str)
    parser.add_argument("--concatonate",metavar='N', type=str, nargs='*', help="Concatonate multiple tmp_results.csv together")
    parser.add_argument("-s","--sbatch", help="Create aand submit Jobscript based off info from 'bench_config.json'",action="store_true")
    parser.add_argument("-p","--plot", help="Plot the Benchmark",action="store_true")
    parser.add_argument("-r","--run", help="Run the Benchmark",type=str)

    args = parser.parse_args()


    benchmarker = BenchMarker()

    if args.config:
        config = args.config
        benchmarker.read_config(args.config)
    else:
        config = __file__.replace("main.py","bench_config.json")
        benchmarker.read_config(config)
        
    
    if args.sbatch:
        benchmarker.write_jobscript()
        benchmarker.launch_jobscript()
        exit(0)

    if args.concatonate:
        csvs = args.concatonate
        benchmarker.concatonate_csvs(csvs)
        exit(0)

    if args.plot:

        plotter = Plotter()
        plotter.load_data(benchmarker.EnerBe_root_dir + "/EnerBe/tmp_results/results.csv")
        #Maybe a good place to apply masks to the data
        plotter.plot_data  = plotter.plot_data[plotter.plot_data['NAME'] == "xgemm"]
        plotter.GPU_TPE_plot(x="SIZE",hue="GPU_NAME",title="xgemm",style="PRECISION")

    if args.run:

        benchmarker.run(args.run)
        benchmarker.get_regex()
        benchmarker.get_architecture()
        benchmarker.to_csv()

