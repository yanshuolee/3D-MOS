import datetime
import glob
import pandas as pd
import os

def get_time(string):
    string = string.split(' Event')[0]
    return datetime.datetime.strptime(string,"%Y-%m-%d %H:%M:%S.%f")

def generate_index_name(string, splitter='/'):
    string = string.split(splitter)[1]
    size = string.split('(')[1].split('-')[0]
    nobj = string.split('(')[1].split('-')[1]
    depth = string.split('(')[1].split('-')[2]
    seed = string.split('_')[1]
    method = string.split('_')[-1]
    return "{}_{}_{}_{}_{}".format(size, nobj, depth, seed, method)

def combine_index(size, nobj, depth, seed, method):
    return "{}_{}_{}_{}_{}".format(size, nobj, depth, seed, method)

if __name__ == "__main__":
    files = glob.glob('./**/log.txt')
    combination, timestep = [], []
    for file in files:
        with open(file) as f:
            lines = f.readlines()

        start = get_time(lines[0])
        for i in range(-1, -10, -1):
            try:
                end = get_time(lines[i])
                break
            except:
                pass
        deltaT = (end - start).total_seconds()

        combination.append(generate_index_name(file))
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



