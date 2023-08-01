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
from scipy.spatial.transform import Rotation as R

voxel_base = 15 # cm
unit_length = 30 # cm
f = lambda x: int(x*unit_length/voxel_base)
f_inv = lambda x: int(x*voxel_base/unit_length)
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
            sg = tuple(np.append(vertexes[p],[0,1,0,0]).astype(np.int)) # TODO check rotation angle
            trajectory.append(sg)
            detect_bool.append(False)
    
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
        # best_subgoal = {subgoal_pos_list[best_subgoal_idx][:3]}
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
        
        # if len(_coverage)/total_area > 0.55:
        #     break

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

class GCBPlanner_sfss_ROS(pomdp_py.Planner):
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
                self.subgoal_set.extend(generate_subgoal_coord_uav((f(x), f(y), 6), angle=60)) # 4 -> 6, in case of uav collision
                self.subgoal_set.extend(generate_subgoal_coord_uav((f(x), f(y), 8), angle=60))
    
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
        for start, end, length in mst:
            ax.plot([subgoal_set[int(start)][0], subgoal_set[int(end)][0]],
                    [subgoal_set[int(start)][1], subgoal_set[int(end)][1]],
                    [subgoal_set[int(start)][2], subgoal_set[int(end)][2]],color = 'g')
        ax.scatter(subgoal_set[:,0],subgoal_set[:,1],subgoal_set[:,2], marker='o')
        # ax.scatter(root[0],root[1],root[2], marker='*', s=300)
        ax.set(xlabel='x', ylabel='y', zlabel='z')
        plt.show()
        '''

        self.subgoal_set = set(self.subgoal_set)
        self.G = OrderedSet()
        self.coverage = set()
        self.next_best_subgoal = None # (x,y,z,thx,thy,thz)
        self.look_table = {}
        for ang in range(0, 360, 60):
            self.look_table[tuple(R.from_euler('xyz', [0, 0, ang], degrees=True).as_quat())] = "look_ang_{}".format(ang)
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
        # If there is no subgoal, plan it.
        if self.next_best_subgoal is None:
            print("GCB planning...")
            if self.p:
                self.paths, self.is_detect = generalized_cost_benefit(agent, self.subgoal_set, self.G, self.coverage, 
                                             self.mst.traverse, mst=self.mst.mst, budget=self.B, total_area=self.total_area, max_time=max_time)
                self.paths.reverse()
                self.is_detect.reverse()
                self.p = False

                """
                pack = {"traj":self.paths, "detect":self.is_detect}
                import pickle
                with open("/home/yanshuo/Documents/3D-MOS/mos3d/experiments/results/mst-GEB-path.pickle", 'wb') as f:
                    pickle.dump(pack, f)
                """

        if len(self.paths) > 0:
            self.next_best_subgoal = self.paths.pop()
            self.detect = self.is_detect.pop()
        else:
            return

        return self.next_best_subgoal, self.detect