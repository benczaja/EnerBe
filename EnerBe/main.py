import json


class Case:
        '''
        The Parent class that holds all of the infromation from the run.
        '''
        def __init__(self):
                self.modules = []
                self.sbatch_data = {}
                self.case_info = {}

                self.results = {}
                self.log = {}

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