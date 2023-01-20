import datetime
import glob
import pandas as pd
import os
import yaml
import numpy as np
import sys

if sys.platform == 'linux':
    splitter = '/'
else:
    splitter = '\\'

def get_time(string):
    string = string.split(' Event')[0]
    return datetime.datetime.strptime(string,"%Y-%m-%d %H:%M:%S.%f")

def generate_index_name(string, splitter=splitter): 
    string = string.split(splitter)[1]
    size = string.split('(')[1].split('-')[0]
    nobj = string.split('(')[1].split('-')[1]
    depth = string.split('(')[1].split('-')[2]
    seed = string.split('_')[1]
    method = string.split('_')[-1]
    maxTime = string.split('(')[1].split(')')[0].split('-')[-1]
    return "{}_{}_{}_{}_{}".format(size, nobj, depth, seed, method), maxTime, nobj

def combine_index(size, nobj, depth, seed, method):
    return "{}_{}_{}_{}_{}".format(size, nobj, depth, seed, method)

if __name__ == "__main__":
    files = glob.glob('./**/log.txt')
    combination, timestep = [], []
    for file in files:
        with open(file) as f:
            lines = f.readlines()

        with open(os.path.join(file[:-8], 'rewards.yaml'), 'r') as f: # check
            reward = yaml.load(f, Loader=yaml.FullLoader)
        reward = np.array(reward)
        ind = np.where(reward==1000)[0]

        title, maxTime, nobj = generate_index_name(file)
        
        if ind.shape[0] != 0:
            ind = ind + 1
            nobj = int(nobj)
            maxTime = int(maxTime)
            new_lines = [lines[0]]
            for line in lines: 
                if 'Step' in line: 
                    new_lines.append(line)
            new_lines.append(lines[-1])
            time_list = []

            start = get_time(lines[0])
            for i in ind:
                txt = new_lines[i]
                end = get_time(txt)
                dt = (end - start).total_seconds()
                time_list.append(dt)
            time_list.append((nobj - len(time_list))*maxTime)
            
            deltaT = sum(time_list) / nobj
        else:
            deltaT = int(maxTime)

        combination.append(title)
        timestep.append(deltaT)

    df1 = pd.DataFrame({"binded":combination, "time":timestep})
    if os.path.exists("./detections_results.csv"):
        df2 = pd.read_csv("./detections_results.csv")
        df2["binded"] = df2.apply(lambda x: combine_index(x['size'], x['nobj'], x['depth'], x['seed'], x['method']), axis=1)
        df3 = df2.merge(df1, on=["binded"])
        df3.to_csv('./detections_results_timestep.csv', index=0)
        print("detections_results_timestep.csv saved!")
    else:
        df1.to_csv('./timestep.csv', index=0)
        print("timestep.csv saved!")



