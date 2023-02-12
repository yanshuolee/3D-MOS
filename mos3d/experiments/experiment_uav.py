# Copyright 2023 Yanshuo Lee
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

save_results = True
output_dir = os.path.join("results", "scalability-II")
prior_type = "uniform"
discount_factor = 0.99
detect_after_look = True

def save_trial_results(exp_path, trial_name, trial_results, log, config):
    trial_path = os.path.join(exp_path, trial_name)
    if not os.path.exists(trial_path):
        os.makedirs(trial_path)

    config_path = os.path.join(trial_path, "config.yaml")
    print("Saving configuration for trial %s at %s..." % (trial_name, config_path))
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    print("Saving results for trial %s..." % (trial_name))
    for result in trial_results:
        if isinstance(result, list):
            result_path = os.path.join(trial_path, "object_found.npy")
            with open(result_path, 'wb') as f:
                np.save(f, np.array(result))
        else:
            result_path = os.path.join(trial_path, result.filename)
            result.save(result_path)

    log_path = os.path.join(trial_path, "log.txt")
    with open(log_path, "w") as f:
        print("| Saving events to %s..." % (log_path))
        for event in log:
            f.write(str(event) + "\n")

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
    num_trials = 1
    seeds = [random.randint(1, 1000000) for i in range(500)]
    scenarios = [(8, 2, 2, 10, 3.0, 500, 600)] # limit to 10 min.
    VIZ = False

    random.shuffle(scenarios)
    # Split the seeds into |scenarios| groups
    splitted_seeds = []
    for i in range(len(scenarios)):
        if (i+1)*num_trials > len(seeds):
            raise ValueError("Not enough seeds generated.")
        splitted_seeds.append(seeds[i*num_trials:(i+1)*num_trials])

    all_trials = []
    for i in range(len(scenarios)): # n, k, d, max_depth, planning_time, max_steps, max_time
        n, k, d, max_depth, planning_time, max_steps, max_time = scenarios[i]

        for seed in splitted_seeds[i]:
            random.seed(seed)

            # Make trials
            # worldstr = make_domain(n, k, d)
            with open(os.path.join(os.path.abspath(__file__).split('experiment_uav.py')[0],
                     'GEB.txt'), 'r') as file:
                worldstr = file.read()

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

            params['alpha'] = alpha

            # Use online flying version of trail
            params['_type'] = "online"
            params['n_target'] = 2

            trial_name = "domain%s_%s" % (str(scenarios[i]).replace(", ", "-"), str(seed))
            pouct_trial = make_trial(trial_name, worldstr,
                                     "pouct", "octree", **params)
            multires_trial = make_trial(trial_name, worldstr,
                                        "hierarchical", "octree",
                                        setting=setting_hier, **params)
            options_trial = make_trial(trial_name, worldstr,
                                        "options", "octree",
                                       setting=setting_op, **params)
            pomcp_trial = make_trial(trial_name, worldstr,
                                     "pomcp", "particles",
                                     num_particles=1000, **params)
            random_trial = make_trial(trial_name, worldstr,
                                      "purelyrandom", "octree", **params)
            porollout_trial = make_trial(trial_name, worldstr,
                                         "porollout", "octree",
                                         porollout_policy=PolicyModel(detect_after_look=detect_after_look),
                                         **params)
            greedy_trial = make_trial(trial_name, worldstr,
                                      "greedy", "octree",
                                      **params)
            bruteforce_trial = make_trial(trial_name, worldstr,
                                          "bruteforce", "octree", viz=VIZ,
                                          **params)
            gcb_trial = make_trial(trial_name, worldstr,
                                   "gcb", "octree", viz=VIZ,
                                   **params)
            gcb_complete_trial = make_trial(trial_name, worldstr,
                                   "gcbcomplete", "octree", viz=VIZ,
                                   **params)
            gcb_sfss_trial = make_trial(trial_name, worldstr,
                             "gcbsfss", "octree", viz=VIZ,
                             **params)
            """Test"""
            trial = gcb_complete_trial
            # trial = gcb_sfss_trial

            # trial = multires_trial
            # trial = gcb_trial
            # trial = bruteforce_trial

            # np.where(np.array(result[0]._things)==1000)
            result = trial.run_gcb(logging=True)
            # result = trial.run(logging=True)
            """Test"""

            # all_trials.extend([pouct_trial,
            #                    options_trial,
            #                    pomcp_trial,
            #                    porollout_trial,
            #                    ])
            # all_trials.extend([multires_trial,
            #                    random_trial,
            #                    greedy_trial,
            #                    bruteforce_trial,
            #                    ])
            # all_trials.extend([gcb_trial
            #                    ])
            # all_trials.extend([gcb_complete_trial
            #                    ])
            # all_trials.extend([gcb_sfss_trial
            #                    ])
            # all_trials.extend([gcb_complete_trial,
            #                    gcb_sfss_trial
            #                    ])

            if save_results:
                save_trial_results("./results/", trial.name, result, trial.log, trial.config)

    # Generate scripts to run experiments and gather results
    # exp = Experiment("ScalabilityYAgainQQ", all_trials, output_dir, verbose=True)
    # exp.generate_trial_scripts(split=400)
    # print("Find multiple computers to run these experiments.")


if __name__ == "__main__":
    main()
