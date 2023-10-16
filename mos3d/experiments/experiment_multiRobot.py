# Copyright 2022 Kaiyu Zheng
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sciex import Experiment, Trial, Event, Result
# from mos3d.tests.experiments.runner import *
# from mos3d.tests.experiments.experiment import make_domain, make_trial
from mos3d.experiments.runner import *
from mos3d.experiments.experiment import make_domain, make_trial
from mos3d import *
import matplotlib.pyplot as plt
import os
import random

ABS_PATH = os.path.dirname(os.path.abspath(__file__))

VIZ = False
output_dir = os.path.join("results", "scalability-II")
prior_type = "uniform"
discount_factor = 0.99
detect_after_look = True



def main():
    # Some arbitrary seeds for reproductive world generation;
    # How many trials is enough? Suppose in a world of size
    # 4x4x4, there are 5 objects. Then there are (4*4*4)^5,
    # around 1billion possible such worlds. To have 95+/-5
    # confidence interval in the results, assuming the possible
    # worlds distribute normally, then we need to run 384 trials.
    # For 95+/-10 confidence interval, it is 96.
    #
    # For our purpose, we don't care about arbitrary worlds that
    # much (we care to some extend); We care more about whether
    # the algorithm works for a randomly chosen world, under
    # various world settings; If 100 trials (across different settings)
    # indicate that our approach is better, then we have a pretty
    # good confidence that our approach is better. For safety,
    # we can bump that to 200. That means each setting takes
    # about 25 trials; to round it up, do 30.
    num_trials = 40
    seeds = [random.randint(1, 1000000) for i in range(500)]
    
    ##### MRSM #####
    # scenarios = [(8, 2, 3, 10, 3.0, 500, 240)]
    scenarios = [(12, 2, 3, 10, 3.0, 500, 240)]

    VIZ = False
    simulation = False
    _lambda = .9 
    n_robots = 2
    parallel = False
    ##### MRSM #####

    random.shuffle(scenarios)
    # Split the seeds into |scenarios| groups
    splitted_seeds = []
    for i in range(len(scenarios)):
        if (i+1)*num_trials > len(seeds):
            print((i+1)*num_trials, len(seeds))
            raise ValueError("Not enough seeds generated.")
        splitted_seeds.append(seeds[i*num_trials:(i+1)*num_trials])

    all_trials = []
    for i in range(len(scenarios)):
        # m, m, d
        n, k, d, max_depth, planning_time, max_steps, max_time = scenarios[i]

        for seed in splitted_seeds[i]:
            random.seed(seed)

            ##### Environment #####
            # worldstr = make_domain(n, k, d)

            with open(os.path.join(os.path.abspath(__file__).split('experiment_multiRobot.py')[0],
                    'GEB-empty.txt'), 'r') as file:
                worldstr = file.read()
            ##### Environment #####

            ## parameters
            big = 1000
            small = 1
            exploration_const = 1000
            alpha = ALPHA  # ALPHA = 1e5
            beta = BETA    # BETA = 0

            params = {"prior_type": prior_type,
                      "discount_factor": discount_factor,
                      "max_depth": max_depth,
                      "planning_time": planning_time,
                      "max_steps": max_steps,
                      "max_time": max_time,
                      "detect_after_look": detect_after_look,
                      "big": big,
                      "small": small,
                      "exploration_const": exploration_const,
                      "alpha": alpha,
                      "beta": beta}
            if n == 4:
                setting_hier = [(1,1,max_depth), (2,2,max_depth)]
                setting_op = [(1,1,max_depth), (1,2,max_depth)]
            elif n == 8:
                setting_hier = [(1,1,max_depth), (2,2,max_depth), (4,4,max_depth)]
                setting_op = [(1,1,max_depth), (1,2,max_depth), (1,4,max_depth)]
            elif n == 16:
                setting_hier = [(1,1,max_depth), (2,2,max_depth), (4,4,max_depth)]
                setting_op = [(1,1,max_depth), (1,2,max_depth), (1,4,max_depth)]
                alpha = 1e7
            elif n == 32:
                setting_hier = [(1,1,max_depth), (4,4,max_depth), (8,8,max_depth)]
                setting_op = [(1,1,max_depth), (1,4,max_depth), (1,8,max_depth)]
                alpha = 1e8
            elif n == 64:
                setting_hier = [(1,1,max_depth), (4,4,max_depth), (8,8,max_depth)]
                setting_op = [(1,1,max_depth), (1,4,max_depth), (1,8,max_depth)]
                alpha = 1e9
            else: # For testing
                setting_hier = [(1,1,max_depth), (2,2,max_depth), (4,4,max_depth)]
                setting_op = [(1,1,max_depth), (1,2,max_depth), (1,4,max_depth)]

            params['alpha'] = alpha
            
            trial_name = "domain%s_%s" % (str(scenarios[i]).replace(", ", "-"), str(seed))
            
            """Test"""
            # matroid_trial = make_trial(trial_name, worldstr,
            #                  "matroid", "octree", viz=VIZ,
            #                  **params)
            # config = {}
            # config["simulation"] = simulation
            # config["lambda"] = _lambda
            # config["n_robots"] = n_robots
            # result = matroid_trial.run(cnf=config)

            # np.where(np.array(result[0]._things)==1000)
            """Test"""

            """MR"""
            matroid_trial = make_trial(trial_name, worldstr,
                             "MR", "octree", viz=VIZ,
                             **params)
            config = {}
            config["simulation"] = simulation
            config["lambda"] = _lambda
            config["n_robots"] = n_robots
            config["parallel"] = parallel
            config["traj"] = "/home/yanshuo/Documents/Multiuav/GEB/exp/flight_history/test-mrsm.csv"
            # config["traj"] = "/home/yanshuo/Documents/Multiuav/GEB/exp/flight_history/test-capam.csv"
            # config["traj"] = "/home/yanshuo/Documents/Multiuav/GEB/exp/flight_history/test-pdfac.csv"
            result = matroid_trial.run(cnf=config)

            np.where(np.array(result[0]._things)==1000)
            """MR"""

    # Generate scripts to run experiments and gather results
    exp = Experiment("ScalabilityYAgainQQ", all_trials, output_dir, verbose=True)
    exp.generate_trial_scripts(split=400)
    print("Find multiple computers to run these experiments.")


if __name__ == "__main__":
    main()
