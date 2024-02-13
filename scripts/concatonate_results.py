import sys
import os
import pandas as pd

script_dir = os.path.dirname(os.path.realpath(__file__)) + "/"

N_files = len(sys.argv)
filenames = []

results_dir = os.path.dirname(os.path.realpath(script_dir + sys.argv[1]))
for i in range(1, N_files):
    filenames.append(script_dir + sys.argv[i])

count = 0 
for filename in filenames:

    try:
        data_tmp = pd.read_csv(filename)
        
        if count ==0:
            data = data_tmp
            count += 1
        else:
            data = pd.concat([data_tmp, data],ignore_index=True)
            data = data.sort_values(by='Name')
            count += 1

    except FileNotFoundError:
        print(FileNotFoundError)

if "tmp_results" in results_dir:
    data.to_csv(results_dir + "/results.csv",sep=',',index=False)

