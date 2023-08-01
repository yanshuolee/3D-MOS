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
from mos3d.planning import matroid_utils as mat
from itertools import product
from scipy.spatial.distance import cdist
import numpy as np
import os
import pickle
from datetime import datetime
from itertools import combinations
from bidict import bidict

def MRSM(agent, 
         subgoal_pos, 
         graph_info,
         budget, 
         total_area, 
         verbose=True):
    
    #################
    _lambda = .9 
    n_clusters = 3
    #################
    
    _G = OrderedSet() # In tree structure, this will be the index of the tree.
    G = set()
    selected_vertices = OrderedSet() # complete_G
    _coverage = set()
    
    # time_B = max_time - 60 # leave one minute to execute.
    subg_counter = 0
    FS = 0

    adj_mat, vertex_idx = graph_info
    n_edges = adj_mat.shape[0]*(adj_mat.shape[0]-1)//2
    edge_idx = [(i, j) for i in range(adj_mat.shape[0]) for j in range(i)] # C_adj_mat.shape[0]_2
    selected_graph, V = mat.random_sample_edges(edge_idx, 
                                                subgoal_pos, 
                                                vertex_idx.inverse, 
                                                adj_mat, 
                                                n_clusters) # This will modify edge_idx
    selected_vertices = selected_vertices | V
    for i in V:
        _coverage = _coverage | get_fov_voxel(agent, i)
    
    B, _, _, _ = mat.cal_B(selected_graph, adj_mat.shape[0])
    lda = _lambda
    # lda = ((len(_coverage)/total_area)/abs(B)) * _lambda * n_clusters
    FS = (len(_coverage)/total_area) + (lda*B)

    record = {"cost":[], "total_coverage":[], "coverage":[], "B":[], "marginal":[], "B*lmd":[], "lmd":[], "graph":[]}
    start = time.time()
    iteration = 0
    while len(edge_idx) > 0:
        if verbose: print('Iteration', iteration) 
        
        # compute coverage func
        s1=time.time()
        # current_cov = mat.coverage_fn(_coverage)
        
        # follows edge_idx order
        cov_output = map(mat.compute_coverage_fn, 
                         edge_idx, 
                         [agent]*len(edge_idx), 
                         [subgoal_pos]*len(edge_idx), 
                         [vertex_idx.inverse]*len(edge_idx), 
                         [_coverage]*len(edge_idx))
        cov_output = list(cov_output)
        coverage = np.array([i[0] for i in cov_output])

        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     results = executor.map(mat.compute_coverage_fn, [agent]*n, edges, [_coverage]*n)
        # cov_output = [i for i in results]

        if verbose: print('phase-1', (time.time()-s1)) 
        
        # Compute balancing func
        s2=time.time()
        (B, is_indep, b1, b2, b3) = mat.balancing_fn(edge_idx, 
                                         selected_graph, 
                                         adj_mat, 
                                         n_edges, 
                                         budget=budget, 
                                         n_clusters=n_clusters)
        # B = (B - B.min())/(B.max()-B.min())
        lda = _lambda
        # lda = (max(coverage/total_area)/abs(max(B))) * _lambda * n_clusters
        objective = (coverage/total_area) + (lda*B) - FS
        
        ########################################
        if verbose: print('phase-2', time.time()-s2)
        
        s3 = time.time()
        obj_max = objective.max()
        max_idx = np.where(objective==obj_max)[0]
        if obj_max < 0:
            print()
        if len(max_idx) > 1:
            # best_edge_idx = np.random.choice(max_idx, 1)[0]
            best_edge_idx = max_idx[0]
        else:
            best_edge_idx = max_idx[0]
        if verbose: print('phase-3', time.time()-s3)

        pos1, pos2 = cov_output[best_edge_idx][1]

        # Finalize
        print("f(SUe)={}, B(SUe)={}, marginal gain={}, F(S)={}, _coverage={}".format((coverage/total_area)[best_edge_idx], 
                                          B[best_edge_idx], 
                                          objective[best_edge_idx], FS, len(_coverage)/total_area))
        s4 = time.time()
        if is_indep[best_edge_idx]:
            (a, b) = edge_idx[best_edge_idx]
            selected_graph.add_edge(a, b, weight=adj_mat[a][b])
            selected_vertices = selected_vertices | {pos1} | {pos2}

            FS = objective[best_edge_idx] + FS

            _coverage = _coverage.union(mat.get_fov_voxel(agent, pos1))
            _coverage = _coverage.union(mat.get_fov_voxel(agent, pos2))


        record["coverage"].append((coverage/total_area)[best_edge_idx])
        record["B"].append(B[best_edge_idx])
        record["marginal"].append(objective[best_edge_idx])
        record["B*lmd"].append(lda*B[best_edge_idx])
        record["lmd"].append(lda)
        record["total_coverage"].append(round(len(_coverage)/total_area, 2))
        record["graph"].append(selected_graph.copy())

        #     ### MST ###
        #     subg_counter = 0

        #     record["cost"].append(cost_gain_[best_subgoal_idx])
        #     record["coverage"].append(round(len(_coverage)*100/total_area, 2))
        
        # if ((time.time() - start) > time_B) or (subg_counter >= subg_counter_limit):
        #     if verbose: print("Time", time.time() - start, "subg_counter", subg_counter)
        #     break
        
        # subg_counter += 1

        # remove angles or edges
        try:
            subgoal_pos[pos1[:3]].remove(pos1[3:])
            subgoal_pos[pos2[:3]].remove(pos2[3:])
        except:
            pass
        if (len(subgoal_pos[pos1[:3]]) == 0) or (len(subgoal_pos[pos2[:3]]) == 0):
            edge_idx.pop(best_edge_idx)
        if verbose: print('phase-4', time.time()-s4)
        iteration += 1
        if verbose: print('===========', time.time()-s1)
    
    trajectories = mat.generate_path(selected_graph)
    print('== Coverage:', '{}%'.format(round(len(_coverage)*100/total_area, 2)))
    print('== Time elapsed:', '{}%'.format(time.time()-start))
    
    # idx to coordinates
    selected_vertices_ = {}
    for i in selected_vertices:
        if i[:3] not in selected_vertices_:
            selected_vertices_[i[:3]] = [i[3:]]
        else:
            selected_vertices_[i[:3]].append(i[3:]) 

    vidx_i = vertex_idx.inverse
    traj_coord = []
    for traj in trajectories:
        tmp = []
        for i in traj:
            if len(selected_vertices_[vidx_i[i]]) > 0:
                tmp.extend([vidx_i[i]+j for j in selected_vertices_[vidx_i[i]]])
                selected_vertices_[vidx_i[i]] = []
            else:
                tmp.append(vidx_i[i]+(0,0,0,1))
        traj_coord.append(tmp)

    print("Routing:", mat.cal_routing(traj_coord))

    '''
    import networkx as nx
    msts = nx.minimum_spanning_tree(selected_graph, algorithm="prim")
    nx.draw_networkx(msts)
    '''
    
    '''
    
    '''
    print()
    return {
        "traj_index": trajectories,
        "vertex_idx": vertex_idx,
        "traj_coord": traj_coord,
        "selected_graph": selected_graph,
    }

class MatroidPlanner(pomdp_py.Planner):
    """
    Randomly plan, but still detect after look.
    Note that this version of GCB plan a path first then traverse.
    """
    def __init__(self, env, stride = 3):
        w, h, l = env._gridworld.width, env._gridworld.height, env._gridworld.length
        self.total_area = w*h*l
        w_range, h_range, l_range = [i for i in range(0, w, stride)], \
            [i for i in range(0, h, stride)], [i for i in range(1, l, stride)]
        # self.subgoal_set = {i:mat.generate_subgoal_coord(c) for i in product(w_range, h_range, l_range)}
        self.subgoal_set = {i:mat.generate_subgoal_coord_uav() for i in product(w_range, h_range, l_range)}
        
        vertexes = list(self.subgoal_set.keys())
        self.graph = (cdist(vertexes, vertexes, 'euclidean'), 
                      bidict({tuple(v):i for i, v in enumerate(vertexes)}))

        '''
        fp = "/home/yanshuo/Documents/3D-MOS/mos3d/experiments/results/matroid/2023-05-16-20-45-04.pickle"
        fp = "/home/yanshuo/Documents/Multiuav/model/v2-B-5/2023-05-26-19-35-17.pickle"
        with open(fp, 'rb') as f:
            results = pickle.load(f)

        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        import networkx as nx
        T = nx.minimum_spanning_tree(results["selected_graph"], algorithm="prim")
        nx.draw_networkx(T)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        cls = list(nx.connected_components(T))
        vertex_idx = results["vertex_idx"].inverse
        c = ["g","b", "r"]; labels=["uav1","uav2","uav3"]
        for edge in [(u,v) for (u, v, d) in T.edges(data=True)]:
            for i in range(len(cls)):
                if edge[0] in cls[i]:
                    color = c[i]
                    label = labels[i]
                    break
            ax.plot([vertex_idx[edge[0]][0], vertex_idx[edge[1]][0]],
                    [vertex_idx[edge[0]][1], vertex_idx[edge[1]][1]],
                    [vertex_idx[edge[0]][2], vertex_idx[edge[1]][2]],color = color)
            ax.scatter((vertex_idx[edge[0]][0], vertex_idx[edge[1]][0]),
                    (vertex_idx[edge[0]][1], vertex_idx[edge[1]][1]),
                    (vertex_idx[edge[0]][2], vertex_idx[edge[1]][2]), 
                    marker='o', color=color)

        handles, _ = plt.gca().get_legend_handles_labels()
        handles.extend([Line2D([0], [0], label=l, color=c_) for (l, c_) in zip(labels, c)])
        plt.legend(handles=handles)
        '''

        ''' Draw subgoals
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        subgoal_set = np.array(vertexes)
        # for start, end, length in mst:
        #     ax.plot([subgoal_set[int(start)][0], subgoal_set[int(end)][0]],
        #             [subgoal_set[int(start)][1], subgoal_set[int(end)][1]],
        #             [subgoal_set[int(start)][2], subgoal_set[int(end)][2]],color = 'g')
        ax.scatter(subgoal_set[:,0],subgoal_set[:,1],subgoal_set[:,2], marker='o', s=50)
        # arrow_len = 4
        # for pos in subgoal_set:
        #     for ang in [0, 60, 120, 180, 240, 300]:
        #         print(arrow_len*np.cos(ang * np.pi / 180.),
        #             arrow_len*np.sin(ang * np.pi / 180.),
        #             np.sqrt((arrow_len*np.cos(ang * np.pi / 180.))**2 + (arrow_len*np.sin(ang * np.pi / 180.))**2))
        #         a = Arrow3D([pos[0], pos[0]+arrow_len*np.cos(ang * np.pi / 180.)], 
        #                     [pos[1], pos[1]+arrow_len*np.sin(ang * np.pi / 180.)], 
        #                     [pos[2], pos[2]], mutation_scale=20, 
        #                     lw=2, arrowstyle="-|>", color="tab:orange")
        #         ax.add_artist(a)

        # ax.scatter(root[0],root[1],root[2], marker='*', s=300)
        ax.set(xlabel='x', ylabel='y', zlabel='z')
        plt.show()
        '''

        # self.subgoal_set = set(self.subgoal_set)
        
        self.next_best_subgoal = None # (x,y,z,thx,thy,thz)
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
        
        print("Multi-robot planning...")
        results = MRSM(agent, 
                      self.subgoal_set, 
                      self.graph, 
                      budget=self.B, 
                      total_area=self.total_area
                      )
        
        '''
        import pickle
        root_path = "/home/yanshuo/Documents/3D-MOS/mos3d/experiments/results/matroid"
        with open("{}/{}.pickle".format(root_path, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),
                  'wb') as f:
            pickle.dump(results, f)
        '''
        
        return 
        
        
