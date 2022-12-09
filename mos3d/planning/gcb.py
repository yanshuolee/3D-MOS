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
from mos3d.planning import gcb_utils, cost_fn
from itertools import product
from scipy.spatial.distance import cdist
import numpy as np

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])[0]

def get_fov_voxel(agent, pos):
    volume = agent.observation_model._gridworld.robot.camera_model.get_volume(pos)
    filtered_volume = {tuple(v) for v in volume if agent.observation_model._gridworld.in_boundary(v)}
    return filtered_volume

def coverage_fn(X):
    return len(X)

def compute_coverage_fn(agent, x, _coverage, current_cov):
    voxels = get_fov_voxel(agent, x)
    del_f = coverage_fn(_coverage.union(voxels)) - current_cov # error
    return del_f

def generate_subgoal_coord(xyz, c):
    x, y, z = xyz
    return [
        (x, y, z, 0, 0, 0, 1), # -x
        (x, y, z, 0, 1, 0, 0), # +x
        (x, y, z, 0, 0, c, c), # -y
        (x, y, z, 0, 0, -c, c),# +y
        (x, y, z, 0, c, 0, c), # +z
        (x, y, z, 0, -c, 0, c) # -z
    ]

def generate_subgoal_union(s1, s2):
    union = s1|{s2[:3]}
    v = np.array([list(i) for i in union])
    return v

import multiprocessing
pool = multiprocessing.Pool(processes=4)
def generalized_cost_benefit(agent, subgoal_pos, G, coverage, objective_fn, budget, opt_gcb = True):
    _G = copy.deepcopy(G)
    _coverage = copy.deepcopy(coverage)
    len_G = len(G)
    gain, gain_coverage, gain_coord = [], [], []
    n_subgoal = len(subgoal_pos)
    while n_subgoal > 0:
        s1=time.time()
        current_cov = coverage_fn(_coverage)
                
        X = [x for x in subgoal_pos]
        cov_output = map(compute_coverage_fn, [agent]*n_subgoal, X, [_coverage]*n_subgoal, [current_cov]*n_subgoal)
        marginal_gain = np.array(list(cov_output))
        print('phase-1', (time.time()-s1)) 
        s2=time.time()
        objective_in = list(map(generate_subgoal_union, [_G]*n_subgoal, X))
        ########################################
        siner = time.time()
        obj_out = map(objective_fn, objective_in)
        cost_gain = np.fromiter(obj_out, dtype=np.float64) 
        # obj_out = pool.map(objective_fn, objective_in)
        
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     obj_out = executor.map(objective_fn, objective_in, chunksize=10)
        #     cost_gain = np.fromiter(obj_out, dtype=np.float64) # np.array([i for i in obj_out])
        print('tsp map', time.time()-siner)
        ########################################
        # print('cost', (time.time()-s2)/len(objective_in), 'n_subg', len(objective_in[0])) # 0.005s
        print('total cost', time.time()-s2, 'avg', (time.time()-s2)/len(objective_in), 'n_subg', len(objective_in[0])) # 0.005s
        s3 = time.time()
        # cost_gain = list(obj_out) # XX
        # cost_gain = [i for i in obj_out]
        
        ratio = marginal_gain / cost_gain
        best_subgoal_idx = ratio.argmax()
        # ratio = list(map(lambda x,y: x/y, marginal_gain, cost_gain)) # denominator will be zero at first
        # best_subgoal_idx = argmax(ratio)
        print('phase-2', time.time()-s3)
        
        # cost
        s4 = time.time()
        subgoal_pos_list = list(subgoal_pos)
        best_subgoal = {subgoal_pos_list[best_subgoal_idx][:3]}
        if cost_gain[best_subgoal_idx] <= budget:
            _G = _G | best_subgoal
            voxels = get_fov_voxel(agent, subgoal_pos_list[best_subgoal_idx])
            _coverage = _coverage.union(voxels)
            gain.append(ratio[best_subgoal_idx])
            gain_coverage.append(voxels)
            gain_coord.append(subgoal_pos_list[best_subgoal_idx]) # (x,y,z,thx,thy,thz)

        elif opt_gcb:
            break
        
        subgoal_pos = subgoal_pos - {subgoal_pos_list[best_subgoal_idx]}
        n_subgoal -= 1
        print('phase-3', time.time()-s4)
        print('==== while', time.time()-s1)

    max_idx = argmax(gain)

    # update current G
    G = G | {list(_G)[len_G+max_idx]} if len(G) != len(_G) else G

    # update current coverage
    voxels = get_fov_voxel(agent, gain_coord[max_idx])
    coverage = coverage.union(gain_coverage[max_idx])
    print('best subgoal', gain_coord[max_idx])
    return gain_coord[max_idx], G, coverage

class GCBPlanner(pomdp_py.Planner):
    """Randomly plan, but still detect after look"""
    def __init__(self, env, c = 0.7071067811865475, stride = 3):
        w, h, l = env._gridworld.width, env._gridworld.height, env._gridworld.length
        w_range, h_range, l_range = [i for i in range(0, w, stride)], \
            [i for i in range(0, h, stride)], [i for i in range(0, l, stride)]
        self.subgoal_set = []
        for i in product(w_range, h_range, l_range):
            self.subgoal_set.extend(generate_subgoal_coord(i, c))
        self.subgoal_set = set(self.subgoal_set)
        self.G = gcb_utils.OrderedSet()
        self.coverage = set()
        self.next_best_subgoal = None # (x,y,z,thx,thy,thz)
        self.look_table = {(0,1,0,0):'look+thx',(0,0,0,1):'look-thx',
                            (0,0,-c,c):'look+thy',(0,0,c,c):'look-thy',
                            (0,c,0,c):'look+thz',(0,-c,0,c):'look-thz'}
        self.action_queue = []
        self.cost_fn = cost_fn.main()
        self.step_counter = 0
        self.B = np.sqrt(w**2+h**2+l**2) * 2

    def plan(self, agent, env):
        if len(self.action_queue) != 0:
            action = self.action_queue.pop()
            return action

        # If there is no subgoal, plan it.
        if self.next_best_subgoal is None:
            next_best_subgoal, G, coverage = generalized_cost_benefit(agent, self.subgoal_set, self.G, self.coverage, self.cost_fn, budget=self.B)
            self.step_counter = cdist([list(env.robot_pose[:3])], [next_best_subgoal[:3]], 'cityblock')[0][0]
            self.next_best_subgoal = next_best_subgoal
            self.G = G
            self.coverage = coverage
            self.subgoal_set = self.subgoal_set - {next_best_subgoal}

        # Go to subgoal with greedy
        if self.step_counter == 0:
            # look and detect
            theta_name = self.look_table[tuple(list(self.next_best_subgoal)[-4:])]
            for a in agent.policy_model.get_all_actions():
                if a.name == theta_name:
                    break
            self.action_queue = [DetectAction(), a]
            self.next_best_subgoal = None
            action = self.action_queue.pop()
            return action

        subgoal_s = list(self.next_best_subgoal) # (x,y,z,thx,thy,thz)
        dists, next_a = [], []
        for pos in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            a = MotionAction((pos,(0,0,0)), 'custom')
            if env.action_valid(a):
                next_state, _ = env.state_transition(a, execute=False)
                next_a.append(a)
                distance = cdist([list(next_state.robot_pose[:3])], [subgoal_s[:3]], 'cityblock')[0][0]
                dists.append(distance)
        if np.array(dists)[dists==min(dists)].shape[0] > 1:
            idx = np.random.choice(np.where(dists==min(dists))[0], 1)[0]
        else:
            idx = argmin(dists)

        if dists[idx] < 1:
            # look and detect
            theta_name = self.look_table[tuple(subgoal_s[-4:])]
            for a in agent.policy_model.get_all_actions():
                if a.name == theta_name:
                    break
            self.action_queue = [DetectAction(), a, next_a[idx]]
            self.next_best_subgoal = None
            action = self.action_queue.pop()
        else:
            action = next_a[idx]

        self.step_counter -= 1

        return action
        