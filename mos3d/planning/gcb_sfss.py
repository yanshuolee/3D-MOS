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
from mos3d.planning import cost_fn, prims
from itertools import product
from scipy.spatial.distance import cdist
import numpy as np
import os

def coord2index(point):
    pos = np.where((vertexes[:,0]==point[0])&(vertexes[:,1]==point[1])&(vertexes[:,2]==point[2]))[0][0]
    return pos

def generate_subgoal_union_idx(s1, s2):
    point = s2[:3]
    pos = np.where((vertexes[:,0]==point[0])&(vertexes[:,1]==point[1])&(vertexes[:,2]==point[2]))[0][0]
    union = s1|{pos}
    return union

_path = []
def dfs(node, mst):
    # global path
    if node not in mst[:,0]:
        return node
    for desc in np.where(mst[:,0]==node)[0]:
        _path.append(node) # print(node)
        dfs(mst[desc,1], mst)
        _path.append(mst[desc,1]) # print(mst[desc,1])

def generate_path(path, complete_G, mst):
    subg = {}
    for p, pos in zip(path, complete_G):
        if p in subg:
            subg[p].append(pos)
        else:
            subg[p] = [pos]
    #
    target = subg.keys()
    while True:
        leaf_node = set(mst[:, 1]) - set(mst[:, 0])
        del_node = leaf_node-target
        if len(del_node) == 0:
            break
        for i in del_node:
            mst = np.delete(mst, np.where(mst[:,1] == i)[0][0], axis=0)
    
    global _path
    _path = []
    dfs(0, mst)
    #

    trajectory, detect_bool = [], []
    detected = []
    for p in _path:
        if p in subg:
            sub_subg = subg[p]
            for i in sub_subg:
                if i in detected:
                    trajectory.append(i)
                    detect_bool.append(False)
                    break
                else:
                    trajectory.append(i)
                    detect_bool.append(True)
                    detected.append(i)
        else:
            sg = tuple(np.append(vertexes[p],[0,1,0,0]).astype(np.int))
            trajectory.append(sg)
            detect_bool.append(False)
    
    
    # for p in subg:
    #     sub_subg = subg[p]
    #     traj = []
    #     is_detect = []
    #     for i in p[:len(p)//2]:
    #         traj.append(tuple(np.append(vertexes[i],[0,1,0,0]).astype(np.int)))
    #         is_detect.append(False)
    #     traj_ = copy.deepcopy(traj);traj_.reverse()
    #     is_detect_ = copy.deepcopy(is_detect)
    #     for i in sub_subg:
    #         traj.append(i);is_detect.append(True)
    #     traj = traj+traj_; is_detect = is_detect+is_detect_
    #     trajectory.extend(traj)
    #     detect_bool.extend(is_detect)
    return trajectory, detect_bool

def generalized_cost_benefit(agent, subgoal_pos, G, coverage, objective_fn, mst, budget, total_area, max_time, subg_counter_limit = 5, verbose=False):
    _G = copy.deepcopy(G) # In tree structure, this will be the index of the tree.
    complete_G = copy.deepcopy(G)
    _coverage = copy.deepcopy(coverage)
    n_subgoal = len(subgoal_pos)
    time_B = max_time - 60 # leave one minute to execute.
    subg_counter = 0
    current_cost = 0
    GUX = 0
    G = set()
    path = []
    record = {"cost":[], "coverage":[]}
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
        # objective_in = list(map(generate_subgoal_union_idx, [_G]*n_subgoal, X))
        ########################################
        siner = time.time()
        # obj_out = map(objective_fn, objective_in)
        # cost_gain_ = np.fromiter(obj_out, dtype=np.float64) 
        # cost_gain = cost_gain_ - current_cost

        ### MST ###
        objective_in = list(map(coord2index, X))
        obj_out = map(objective_fn, objective_in, [G]*n_subgoal)
        obj_out = np.array(list(obj_out)) #np.fromiter(obj_out, dtype=[('',np.float64),('',np.float64)]) 
        cost_gain_ = obj_out[:,0]
        cost_gain = cost_gain_ - GUX # c(GUX)-c(G)
        cost_gain[np.where(cost_gain==0)] = 99999 # set original point to max
        
        ### MST ###

        if verbose: print('tsp map', time.time()-siner)
        ########################################
        if verbose: print('phase-2', time.time()-s2, 'avg', (time.time()-s2)/len(objective_in), 'n_subg', len(objective_in)) # 0.005s
        
        # Compute ratio
        s3 = time.time()
        ratio = marginal_gain / cost_gain
        best_subgoal_idx = ratio.argmax()
        if verbose: print('phase-3', time.time()-s3)
        
        # print(ratio.max())
        if ratio.max() == 0:
            break
        
        # Finalize
        s4 = time.time()
        subgoal_pos_list = list(subgoal_pos)
        best_subgoal = {subgoal_pos_list[best_subgoal_idx][:3]}
        if cost_gain_[best_subgoal_idx] <= budget:
            # _G = _G | best_subgoal
            complete_G = complete_G | {subgoal_pos_list[best_subgoal_idx]}
            voxels = get_fov_voxel(agent, subgoal_pos_list[best_subgoal_idx])
            _coverage = _coverage.union(voxels)
            # current_cost = cost_gain_[best_subgoal_idx]
            ### MST ###
            _G = _G | {objective_in[best_subgoal_idx]}
            GUX = cost_gain_[best_subgoal_idx]
            G = G|set(obj_out[best_subgoal_idx][1])
            
            # p = obj_out[best_subgoal_idx][1]
            # p_ = copy.deepcopy(p);p_.reverse()
            # p = p_+p[1:]
            # path.append(p)
            path.append(objective_in[best_subgoal_idx])

            ### MST ###
            subg_counter = 0

            record["cost"].append(cost_gain_[best_subgoal_idx])
            record["coverage"].append(round(len(_coverage)*100/total_area, 2))
        
        if ((time.time() - start) > time_B) or (subg_counter >= subg_counter_limit):
            if verbose: print("Time", time.time() - start, "subg_counter", subg_counter)
            break
        
        subg_counter += 1
        subgoal_pos = subgoal_pos - {subgoal_pos_list[best_subgoal_idx]}
        n_subgoal -= 1
        if verbose: print('phase-4', time.time()-s4)
        if verbose: print('===========', time.time()-s1)
    
    trajectory, detect_bool = generate_path(path, complete_G, mst)
    print('== Coverage:', '{}%'.format(round(len(_coverage)*100/total_area, 2)))
    return trajectory, detect_bool

class GCBPlanner_sfss(pomdp_py.Planner):
    """
    Randomly plan, but still detect after look.
    Note that this version of GCB plan a path first then traverse.
    """
    def __init__(self, env, c = 0.7071067811865475, stride = 3):
        w, h, l = env._gridworld.width, env._gridworld.height, env._gridworld.length
        self.total_area = w*h*l
        w_range, h_range, l_range = [i for i in range(0, w, stride)], \
            [i for i in range(0, h, stride)], [i for i in range(0, l, stride)]
        self.subgoal_set = []
        for i in product(w_range, h_range, l_range):
            self.subgoal_set.extend(generate_subgoal_coord(i, c))
        
        # For the use of path cost recovery
        global vertexes 
        vertexes = np.array(self.subgoal_set)
        vertexes = np.unique(vertexes[:, :3], axis=0)
        robot_p = list(env.robot_pose[:3])
        robot_p_idx = cdist(vertexes, [robot_p], 'euclidean').argmin()
        vertexes[0], vertexes[robot_p_idx] = copy.deepcopy(vertexes[robot_p_idx]), copy.deepcopy(vertexes[0])
        graph = cdist(vertexes, vertexes, 'euclidean')
        g = prims.Graph(graph)
        mst = g.primMST()
        self.mst = g

        '''
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        subgoal_set = vertexes
        root = vertexes[0]
        for start, end, length in mst:
            ax.plot([subgoal_set[int(start)][0], subgoal_set[int(end)][0]],
                    [subgoal_set[int(start)][1], subgoal_set[int(end)][1]],
                    [subgoal_set[int(start)][2], subgoal_set[int(end)][2]],color = 'g')
        ax.scatter(subgoal_set[:,0],subgoal_set[:,1],subgoal_set[:,2], marker='o')
        ax.scatter(root[0],root[1],root[2], marker='*', s=300)
        ax.set(xlabel='x', ylabel='y', zlabel='z')
        plt.show()
        '''

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
        ### MST ###
        self.B = 99999 #(w+h+l) * 2
        ### MST ###
        self.p = True
        self.paths = None
        self.is_detect = None
        self.detect = None

    def plan(self, agent, env, max_time):
        # If there are still action in queue.
        if len(self.action_queue) != 0:
            action = self.action_queue.pop()
            return action

        # If there is no subgoal, plan it.
        if self.next_best_subgoal is None:
            import pickle
            filep = "/home/yanshuo/Documents/3D-MOS/mos3d/experiments/results/sfss-{}-path.pickle".format(env._gridworld.width)
            fileD = "/home/yanshuo/Documents/3D-MOS/mos3d/experiments/results/sfss-{}-detect.pickle".format(env._gridworld.width)
            if self.p:
                if os.path.exists(filep):
                    with open(filep, 'rb') as f:
                        self.paths = pickle.load(f)
                    with open(fileD, 'rb') as f:
                        self.is_detect = pickle.load(f)
                    self.p = False
                    print("GCB loaded!")
                else:
                    print("GCB planning...")
                # if self.p:
                    self.paths, self.is_detect = generalized_cost_benefit(agent, self.subgoal_set, self.G, self.coverage, 
                                                self.mst.traverse, mst=self.mst.mst, budget=self.B, total_area=self.total_area, max_time=max_time)
                    self.paths.reverse()
                    self.is_detect.reverse()
                    self.p = False

                    with open(filep, 'wb') as f:
                        pickle.dump(self.paths, f)

                    with open(fileD, 'wb') as f:
                        pickle.dump(self.is_detect, f)

            if len(self.paths) > 0:
                next_best_subgoal = self.paths.pop()
                self.detect = self.is_detect.pop()
            else:
                return

            self.step_counter = cdist([list(env.robot_pose[:3])], [next_best_subgoal[:3]], 'cityblock')[0][0]
            self.next_best_subgoal = next_best_subgoal

        # If we are unable to arrive destination due to greedy planning limitation, then just detect.
        if self.step_counter == 0:
            # look and detect
            theta_name = self.look_table[tuple(list(self.next_best_subgoal)[-4:])]
            for a in agent.policy_model.get_all_actions():
                if a.name == theta_name:
                    break
            if self.detect:
                self.action_queue = [DetectAction(), a]
            else:
                self.action_queue = [a]
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
            if self.detect:
                self.action_queue = [DetectAction(), a, next_a[idx]]
            else:
                self.action_queue = [a, next_a[idx]]
            self.next_best_subgoal = None
            action = self.action_queue.pop()
        else:
            action = next_a[idx]

        self.step_counter -= 1

        return action
