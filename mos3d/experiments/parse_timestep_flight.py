import datetime
import glob
import pandas as pd
import os
import yaml
import numpy as np
import sys
from scipy.spatial.transform import Rotation as scipyR
import pickle
from numpy import linalg as LA
from tqdm import tqdm

if sys.platform == 'linux':
    splitter = '/'
else:
    splitter = '\\'

def get_time(string):
    string = string.split(' Event')[0]
    return datetime.datetime.strptime(string,"%Y-%m-%d %H:%M:%S.%f")

def generate_index_name(string, splitter=splitter): 
    string = string.split(splitter)[-2]
    size = string.split('(')[1].split('-')[0]
    nobj = string.split('(')[1].split('-')[1]
    depth = string.split('(')[1].split('-')[2]
    seed = string.split('_')[1]
    method = string.split('_')[-1]
    maxTime = string.split('(')[1].split(')')[0].split('-')[-1]
    return "{}_{}_{}_{}_{}".format(size, nobj, depth, seed, method), maxTime, nobj

def combine_index(size, nobj, depth, seed, method):
    return "{}_{}_{}_{}_{}".format(size, nobj, depth, seed, method)

def quat_to_euler(xyzw):
    return scipyR.from_quat(xyzw).as_euler("xyz", degrees=True)

if __name__ == "__main__":
    files = glob.glob('./**/log.txt')
    traverse = False
    if len(files) == 0:
        files = glob.glob('./**/**/log.txt')
        traverse = True
    combination, timestep, real_n_obj_found = [], [], []
    
    w = 45
    v = 0.1
    # w = 51.6 #45
    # v = .087 #0.1

    # 20 min: 1200
    # 10 min: 600
    time_limit = {
        '240': 600, 
        '360': 600, 
    }

    for file in tqdm(files):
        with open(file) as f:
            lines = f.readlines()

        with open(os.path.join(file[:-8], 'rewards.yaml'), 'r') as f: # check
            reward = yaml.load(f, Loader=yaml.FullLoader)
        reward = np.array(reward)
        ind = np.where(reward==1000)[0]
        _ind = np.where(reward==1000)[0]

        with open( os.path.join(file[:-8], 'states.pkl'), 'rb') as f:
            r_states = pickle.load(f)
        r_states = np.array([quat_to_euler(list(p.robot_pose)[3:]) for p in r_states])
        state_trans = np.diff(r_states, axis=0)
        del_th = LA.norm(state_trans, 1, axis=1)
        flight_time = del_th/w # deg/sec
        flight_time[np.where(flight_time==0)] = v # sec/voxel

        title, maxTime, nobj = generate_index_name(file)
        maxTime = time_limit[maxTime]
        
        if ind.shape[0] != 0:
            ind = ind + 1
            nobj = int(nobj)
            maxTime = int(maxTime)
            new_lines = [lines[0]]
            _new_lines = []
            for line in lines: 
                if 'Step' in line: 
                    new_lines.append(line)
                    _new_lines.append(line)
            new_lines.append(lines[-1])
            
            time_list = []
            start = get_time(lines[0])
            for i, j in zip(ind, _ind):
                # planning time (with a bit of moving time)
                txt = new_lines[i]
                end = get_time(txt)
                dt_plan = (end - start).total_seconds()
                
                # flight time
                dt_flight = flight_time[:j].sum()

                dt = dt_flight + dt_plan
                if dt > maxTime:
                    break
                time_list.append(dt)

            n_obj_found = len(time_list)
            
            time_list.append((nobj - n_obj_found)*maxTime)
            deltaT = sum(time_list) / nobj
        else:
            n_obj_found = 0
            deltaT = int(maxTime)

        combination.append(title)
        timestep.append(deltaT)
        real_n_obj_found.append(n_obj_found)

    df1 = pd.DataFrame({"binded":combination, "time":timestep, "n_obj_found":real_n_obj_found})
    
    # traverse = True
    # df1 = pd.read_csv("./timestep.csv")
    # files = glob.glob('./**/**/log.txt')

    if os.path.exists("./detections_results.csv"):
        df2 = pd.read_csv("./detections_results.csv")
        df2["binded"] = df2.apply(lambda x: combine_index(x['size'], x['nobj'], x['depth'], x['seed'], x['method']), axis=1)
        df3 = df2.merge(df1, on=["binded"])
        df3.to_csv('./detections_results_with_flightT.csv', index=0)
        print("detections_results_with_flightT.csv saved!")
    else:
        if traverse:
            dirs = np.unique([file.split(splitter)[-3] for file in files])
            df2 = pd.DataFrame()
            for _dir in dirs:
                df2_1 = pd.read_csv(os.path.join(_dir, 'detections_results.csv'))
                df2 = df2.append(df2_1)
            df2["binded"] = df2.apply(lambda x: combine_index(x['size'], x['nobj'], x['depth'], x['seed'], x['method']), axis=1)
            df3 = df2.merge(df1, on=["binded"])
            df3.to_csv('./detections_results_with_flightT.csv', index=0)
            print("detections_results_with_flightT.csv saved!")
                
        else:
            df1.to_csv('./timestep.csv', index=0)
            print("timestep.csv saved!")



