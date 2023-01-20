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

def generalized_cost_benefit(agent, subgoal_pos, G, coverage, objective_fn, budget, total_area, opt_gcb = False, verbose=False):
    _G = copy.deepcopy(G)
    _coverage = copy.deepcopy(coverage)
    len_G = len(G)
    gain, gain_coverage, gain_coord = [], [], []
    n_subgoal = len(subgoal_pos)
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
            voxels = get_fov_voxel(agent, subgoal_pos_list[best_subgoal_idx])
            _coverage = _coverage.union(voxels)
            gain.append(ratio[best_subgoal_idx])
            gain_coverage.append(voxels)
            gain_coord.append(subgoal_pos_list[best_subgoal_idx]) # (x,y,z,thx,thy,thz)
        elif opt_gcb:
            break
        
        subgoal_pos = subgoal_pos - {subgoal_pos_list[best_subgoal_idx]}
        n_subgoal -= 1
        if verbose: print('phase-4', time.time()-s4)
        if verbose: print('===========', time.time()-s1)
    
    max_idx = argmax(gain)

    # update current G
    G = G | {list(_G)[len_G+max_idx]} if len(G) != len(_G) else G

    # update current coverage
    voxels = get_fov_voxel(agent, gain_coord[max_idx])
    coverage = coverage.union(gain_coverage[max_idx])
    print('== Best subgoal:', gain_coord[max_idx])
    print('== Current coverage:', '{}%'.format(round(len(coverage)*100/total_area, 2)))
    return gain_coord[max_idx], G, coverage

class GCBPlanner(pomdp_py.Planner):
    """Randomly plan, but still detect after look"""
    def __init__(self, env, c = 0.7071067811865475, stride = 3):
        w, h, l = env._gridworld.width, env._gridworld.height, env._gridworld.length
        self.total_area = w*h*l
        w_range, h_range, l_range = [i for i in range(0, w, stride)], \
            [i for i in range(0, h, stride)], [i for i in range(0, l, stride)]
        self.subgoal_set = []
        for i in product(w_range, h_range, l_range):
            self.subgoal_set.extend(generate_subgoal_coord(i, c))
        self.subgoal_set = set(self.subgoal_set)
        self.G = OrderedSet()
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
        # If there are still action in queue.
        if len(self.action_queue) != 0:
            action = self.action_queue.pop()
            return action

        # If there is no subgoal, plan it.
        if self.next_best_subgoal is None:
            print("GCB planning...")
            if agent._gridworld.width > 8:
                opt_gcb = True
            else:
                opt_gcb = False
            next_best_subgoal, G, coverage = generalized_cost_benefit(agent, self.subgoal_set, self.G, self.coverage, 
                                                                      self.cost_fn, budget=self.B, total_area=self.total_area, opt_gcb=opt_gcb)
            self.step_counter = cdist([list(env.robot_pose[:3])], [next_best_subgoal[:3]], 'cityblock')[0][0]
            self.next_best_subgoal = next_best_subgoal
            self.G = G
            self.coverage = coverage
            self.subgoal_set = self.subgoal_set - {next_best_subgoal}

        # If we are unable to arrive destination due to greedy planning limitation, then just detect.
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

        # Go to subgoal with next one step greedy planning.
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

        # Generate a set of action if the next step is the destination.
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
