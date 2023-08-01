import pomdp_py
from mos3d import AbstractPolicyModel, AbstractM3Belief, M3Agent,\
    AbstractM3ObservationModel, AbstractM3Agent,\
    AbstractM3TransitionModel
from mos3d.oopomdp import Actions, MotionAction, LookAction, DetectAction
# import multiprocessing
import concurrent.futures
import random
import copy
import time
from mos3d.planning.gcb_utils import *
from mos3d.planning import cost_fn
from itertools import product
from scipy.spatial.distance import cdist
import numpy as np
from scipy.spatial.transform import Rotation as R

voxel_base = 15 # cm
unit_length = 30 # cm
f = lambda x: int(x*unit_length/voxel_base)
f_inv = lambda x: int(x*voxel_base/unit_length)
def generalized_cost_benefit(agent, subgoal_pos, G, coverage, objective_fn, budget, total_area, max_time, subg_counter_limit = 5, verbose=False):
    _G = copy.deepcopy(G)
    complete_G = copy.deepcopy(G)
    _coverage = copy.deepcopy(coverage)
    len_G = len(G)
    n_subgoal = len(subgoal_pos)
    time_B = max_time - 60 # leave one minute to execute.
    subg_counter = 0
    start = time.time()
    while n_subgoal > 0:
        # compute cost fn
        s1=time.time()
        current_cov = coverage_fn(_coverage)
        X = [x for x in subgoal_pos]
        cov_output = map(compute_coverage_fn, [agent]*n_subgoal, X, [_coverage]*n_subgoal, [current_cov]*n_subgoal)
        # marginal_gain = np.fromiter(cov_output, dtype=np.int16) 
        marginal_gain = np.array(list(cov_output))
        if verbose: print('phase-1', (time.time()-s1)) 
        
        # Solve TSP
        s2=time.time()
        objective_in = list(map(generate_subgoal_union, [_G]*n_subgoal, X))
        ########################################
        siner = time.time()
        obj_out = map(objective_fn, objective_in)
        cost_gain = np.fromiter(obj_out, dtype=np.float64) 
        
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     obj_out = executor.map(objective_fn, objective_in, chunksize=10)
        #     cost_gain = np.fromiter(obj_out, dtype=np.float64) # np.array([i for i in obj_out])
        if verbose: print('tsp map', time.time()-siner)
        ########################################
        if verbose: print('phase-2', time.time()-s2, 'avg', (time.time()-s2)/len(objective_in), 'n_subg', len(objective_in[0])) # 0.005s
        
        # Compute ratio
        s3 = time.time()
        ratio = marginal_gain / cost_gain
        best_subgoal_idx = ratio.argmax()
        if verbose: print('phase-3', time.time()-s3)
        
        # Finalize
        s4 = time.time()
        subgoal_pos_list = list(subgoal_pos)
        best_subgoal = {subgoal_pos_list[best_subgoal_idx][:3]}
        if cost_gain[best_subgoal_idx] <= budget:
            _G = _G | best_subgoal
            complete_G = complete_G | {subgoal_pos_list[best_subgoal_idx]}
            voxels = get_fov_voxel(agent, subgoal_pos_list[best_subgoal_idx])
            _coverage = _coverage.union(voxels)
            subg_counter = 0
        
        if ((time.time() - start) > time_B) or (subg_counter >= subg_counter_limit):
            if verbose: print("Time", time.time() - start, "subg_counter", subg_counter)
            break
        
        subg_counter += 1
        subgoal_pos = subgoal_pos - {subgoal_pos_list[best_subgoal_idx]}
        n_subgoal -= 1
        if verbose: print('phase-4', time.time()-s4)
        if verbose: print('===========', time.time()-s1)
    
    tsp_path, cost = cost_fn.nn_tsp(complete_G)
    print('== Coverage:', '{}%'.format(round(len(_coverage)*100/total_area, 2)))
    return tsp_path

class GCBPlanner_complete_ROS(pomdp_py.Planner):
    """
    Randomly plan, but still detect after look.
    Note that this version of GCB plan a path first then traverse.
    """
    def __init__(self, env, c = 0.7071067811865475, stride = 3):
        w, h, l = env._gridworld.width, env._gridworld.height, env._gridworld.length
        self.total_area = w*h*l - len(env.object_poses) # eliminate obstacles.
        
        self.subgoal_set = []
        # Left square area
        for x in [10, 21]:
            for y in [11, 22, 33]:
                self.subgoal_set.extend(generate_subgoal_coord_uav((f(x), f(y), 4), angle=60))
                self.subgoal_set.extend(generate_subgoal_coord_uav((f(x), f(y), 8), angle=60))
    
        self.subgoal_set = set(self.subgoal_set)
        self.G = OrderedSet()
        self.coverage = set()
        self.next_best_subgoal = None # (x,y,z,thx,thy,thz)
        self.look_table = {}
        for ang in range(0, 360, 60):
            self.look_table[tuple(R.from_euler('xyz', [0, 0, ang], degrees=True).as_quat())] = "look_ang_{}".format(ang)
        self.action_queue = []
        self.cost_fn = cost_fn.main()
        # self.step_counter = 0
        self.B = (w+h+l) * 2
        self.p = True
        self.paths = None

    def plan(self, agent, env, max_time):
        # If there is no subgoal, plan it.
        if self.next_best_subgoal is None:
            print("GCB planning...")
            if self.p:
                self.paths = generalized_cost_benefit(agent, self.subgoal_set, self.G, self.coverage, 
                                                      self.cost_fn, budget=self.B, total_area=self.total_area, max_time=max_time)
                self.paths.reverse()
                self.p = False
                '''
                self.paths = self.paths[12:23] + self.paths[36:47] + self.paths[0:12]
                self.is_detect = [True]*len(self.paths)
                pack = {"traj":self.paths, "detect":self.is_detect}
                import pickle
                with open("/home/yanshuo/Documents/3D-MOS/mos3d/experiments/results/gcb-GEB-path.pickle", 'wb') as f:
                    pickle.dump(pack, f)
                '''

        if len(self.paths) > 0:
            self.next_best_subgoal = self.paths.pop()
        else:
            return

        return self.next_best_subgoal
