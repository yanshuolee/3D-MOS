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

def generalized_cost_benefit(agent, subgoal_pos, G, coverage, objective_fn, budget):
    _G = copy.deepcopy(G)
    _coverage = copy.deepcopy(coverage)
    len_G = len(G)
    gain, gain_coverage, gain_coord = [], [], []
    while len(subgoal_pos) != 0:
        current_cov = coverage_fn(_coverage)
        """ # slow version
        marginal_gain1, cost_gain1 = [], []
        current_cost = objective_fn(np.array([list(i) for i in _G]))
        t3, t2 = [], []
        s1=time.time()
        for x in subgoal_pos: # x: (x,y,z,thx,thy,thz)
            s3=time.time()
            voxels = get_fov_voxel(agent, x)
            del_f = coverage_fn(_coverage.union(voxels)) - current_cov # error
            t3.append(time.time()-s3)
            # print('coverage', time.time()-s3) # 0.002s
            marginal_gain1.append(del_f)

            # cost
            s2=time.time()
            check1=objective_fn(np.array([list(i) for i in _G|{x[:3]}]))
            t2.append(time.time()-s2)
            del_c = max(check1 - current_cost, 1e-15)
            
            # print('cost', time.time()-s2) # 0.005s
            cost_gain1.append(del_c)
        print('for loop', time.time()-s1, '#subgoal', len(subgoal_pos), 'cov', sum(t3)/len(t3), 'cost', sum(t2)/len(t2)) # 0.6s
        """
        n_subgoal = len(subgoal_pos)
        X = [x for x in subgoal_pos]
        cov_output = map(compute_coverage_fn, [agent]*n_subgoal, X, [_coverage]*n_subgoal, [current_cov]*n_subgoal)
        marginal_gain = list(cov_output)
        s2=time.time()
        objective_in = list(map(generate_subgoal_union, [_G]*n_subgoal, X))
        obj_out = map(objective_fn, objective_in)
        # print('cost', (time.time()-s2)/len(objective_in)) # 0.005s
        cost_gain = list(obj_out)

        ratio = list(map(lambda x,y: x/y, marginal_gain, cost_gain)) # denominator will be zero at first
        best_subgoal_idx = argmax(ratio)

        # cost
        subgoal_pos_list = list(subgoal_pos)
        best_subgoal = {subgoal_pos_list[best_subgoal_idx][:3]}
        if objective_fn(np.array([list(i) for i in _G|best_subgoal])) <= budget:
            _G = _G | best_subgoal
            voxels = get_fov_voxel(agent, subgoal_pos_list[best_subgoal_idx])
            _coverage = _coverage.union(voxels)
            gain.append(ratio[best_subgoal_idx])
            gain_coverage.append(voxels)
            gain_coord.append(subgoal_pos_list[best_subgoal_idx]) # (x,y,z,thx,thy,thz)

        subgoal_pos = subgoal_pos - {subgoal_pos_list[best_subgoal_idx]}

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

    def plan(self, agent, env):
        if len(self.action_queue) != 0:
            action = self.action_queue.pop()
            return action

        # If there is no subgoal, plan it.
        if self.next_best_subgoal is None:
            next_best_subgoal, G, coverage = generalized_cost_benefit(agent, self.subgoal_set, self.G, self.coverage, self.cost_fn, budget=10000000)
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

        return MotionAction(((3,6,5),(0,0,0)),'custom')
        return random.sample(agent.policy_model.get_all_actions(history=agent.history), 1)[0]
        if self._should_detect:
            return DetectAction()
        else:
            return random.sample(agent.policy_model.get_all_actions(history=agent.history), 1)[0]

    # def update(self, agent, real_action, real_observation, **kwargs):
    #     robot_state = agent.belief.mpe().robot_state
    #     if isinstance(real_action, LookAction):
    #         objects_observing = set({objid
    #                                  for objid in real_observation.voxels
    #                                  if real_observation.voxels[objid].label == objid})
    #         objects_found = robot_state['objects_found']
    #         if len(objects_observing - set(objects_found)) > 0:
    #             self._should_detect = True
    #             return
    #     self._should_detect = False