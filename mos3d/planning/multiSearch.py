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
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import json
from matplotlib import pyplot as plt

def MRSM(agent, 
         subgoal_pos, 
         graph_info,
         budget, 
         total_area,
         param, 
         verbose=True):
    
    ####### Param ########
    complete_G = True
    draw_iter_graph = True
    _lambda = param["lambda"]
    n_clusters = param["n_robots"]
    parallel = param["parallel"]
    method = param["method"]
    save_iter_root = param["save_iter_root"]
    lazy_greedy_mode = param["lazy_greedy_mode"]
    ####### Param ########

    if parallel:
        from ray.util.multiprocessing import Pool
        import ray
        pool = Pool()
    else:
        pool = None
    
    selected_vertices = OrderedSet()
    _coverage = set()
    FS = 0
    adj_mat, vertex_idx = graph_info

    if parallel: 
        _c =  mat.Cov(agent, subgoal_pos, total_area)
        param["All subgoals coverage"] = _c
        print("Total coverage:", _c)

    ################## graph type #####################
    if complete_G:
        # complete graph
        edge_idx = np.array([(i, j) for i in range(adj_mat.shape[0]) for j in range(i)])
    else:
        # incomplete graph
        edge_idx = np.array([(i, j) for i in range(adj_mat.shape[0]) for j in range(i)])
        T = np.arange(len(edge_idx))
        np.random.seed(10)
        t = np.random.choice(T, int(len(T)*.6), replace=False)
        edge_idx = edge_idx[t]
        edge_idx = edge_idx.tolist()
    print("Number of edges:", len(edge_idx))
    ################## choose method #####################
    if method == "MRSM":
        _type = "min"
        # _type = "random"
    elif method == "MRSIS-TSP":
        _type = "random"
    elif method == "MRSIS-MST":
        _type = "random"
    else:
        raise Exception("Invalid method.")
    
    selected_graph, V, edge_idx = mat.sample_edges(
                                        edge_idx, 
                                        subgoal_pos, 
                                        vertex_idx.inverse, 
                                        adj_mat, 
                                        n_clusters,
                                        _type=_type,
                                ) # This will modify edge_idx
    
    ################## initialization #####################
    selected_vertices = selected_vertices | V
    for i in V:
        _coverage = _coverage | get_fov_voxel(agent, i)
    
    # B, _, _, _ = mat.cal_B(selected_graph, adj_mat.shape[0])
    _, b1, b2, b3 = mat.cal_B(selected_graph, adj_mat.shape[0])
    B = b3 # test normalization
    lda = _lambda
    FS = (len(_coverage)/total_area) + (lda*B)

    record = {"cost":[], "total_coverage":[], "coverage":[], "B":[], "marginal":[], 
              "B*lmd":[], "lmd":[], "graph":[], "n_clusters":[]}
    start = time.time()
    iteration = 0
    lazy_greedy = {"idx": edge_idx.copy(), "B": [None]*len(edge_idx), "history":[], "selected":[], "cumu_B":[]}
    lg_control = 1; lg_count = 1
    try:
        while len(edge_idx) > 0:
            if verbose: print('===== Iteration', iteration, '=====') 
            ################## compute coverage func #####################
            s1=time.time()
            # follows edge_idx order    
            if parallel:
                cov_output = pool.map(mat.compute_coverage_fn_parallel, 
                                    zip(edge_idx, 
                                        [agent]*len(edge_idx), 
                                        [subgoal_pos]*len(edge_idx), 
                                        [vertex_idx.inverse]*len(edge_idx), 
                                        [_coverage]*len(edge_idx)))
            else:
                cov_output = map(mat.compute_coverage_fn, 
                                edge_idx, 
                                [agent]*len(edge_idx), 
                                [subgoal_pos]*len(edge_idx), 
                                [vertex_idx.inverse]*len(edge_idx), 
                                [_coverage]*len(edge_idx))
            cov_output = list(cov_output)
            coverage = np.array([i[0] for i in cov_output])
            if verbose: print('phase-1', (time.time()-s1)) 
            ################### Compute balancing func ####################
            s2=time.time()
            if lazy_greedy_mode:
                # _mask = np.any(np.isin(edge_idx, list(selected_graph.nodes)), axis=1)
                # print(edge_idx[_mask].shape[0])
                print("len of max", len(np.where(B==B.max())[0])) 
                ## test LG ## == 1, next will need to cal _mask, next cal all
                print("lg_control:", lg_control); _lg_control = int(lg_control)
                if lg_control == 0:
                    # skip, and remove balance edge
                    if len(np.where(B==B.max())[0]) == 1:
                        lg_control = 1
                elif lg_control == 1:
                    # update all
                    (_, is_indep, b1, b2, b3, r, nc) = mat.balancing_fn(edge_idx, 
                                                                    selected_graph, 
                                                                    adj_mat, 
                                                                    budget=budget, 
                                                                    n_clusters=n_clusters,
                                                                    pool=pool,
                                                                    method=method,
                                                                )        
                    B = b3-((nc-nc.min())/(nc.max()-nc.min())) if (nc.max()-nc.min()) != 0 else b3
                    ## todo
                    if lg_count >= 1:
                        lg_control = 0
                        lg_count = 0
                    else:
                        lg_count += 1

                if False: # debug use
                    _tm = []
                    for II in range(len(edge_idx)):
                        (_a, _b) = edge_idx[II]
                        IDX_lg = np.where((lazy_greedy["idx"][:,0]==_a)&(lazy_greedy["idx"][:,1]==_b))[0][0]
                        if lazy_greedy["B"][IDX_lg] != B[II]:
                            lazy_greedy["B"][IDX_lg] = B[II]
                            _tm.append((_a, _b))
                    # lazy_greedy["history"].append(_tm)
                    # lazy_greedy["cumu_B"].append(lazy_greedy["B"])
                    print("Ans:", len(_tm), len(edge_idx))
                    if not (_B == B).all():
                        print((_B == B).all())
                    if (len(_tm)>0) and (_lg_control ==0): 
                        print()
            else:
                (_, is_indep, b1, b2, b3, r, nc) = mat.balancing_fn(edge_idx, 
                                                                    selected_graph, 
                                                                    adj_mat, 
                                                                    budget=budget, 
                                                                    n_clusters=n_clusters,
                                                                    pool=pool,
                                                                    method=method,
                                                                    lg = lazy_greedy,
                                                                )        
                B = b3-((nc-nc.min())/(nc.max()-nc.min())) if (nc.max()-nc.min()) != 0 else b3 # test normalization
            
            lda = _lambda
            objective = (coverage/total_area) + (lda*B) - FS
            if verbose: print('phase-2', time.time()-s2)
            ########################################
            s3 = time.time()
            obj_max = objective.max()
            max_idx = np.where(objective==obj_max)[0]
            if obj_max < 0:
                print("Warning: Negative marginal gain.")
            
            if len(max_idx) > 1:
                best_edge_idx = np.random.choice(max_idx, 1)[0]
            else:
                best_edge_idx = max_idx[0]
            
            print("Number of clusters (best edge):", nc[best_edge_idx])
            if nc[best_edge_idx] > n_clusters:
                print("Terminating...")
                is_indep[best_edge_idx] = False

            ##### debug zone #####
            if False:
                print("Current:", edge_idx[best_edge_idx])
                for ii in np.where(B==np.sort(np.unique(B))[-2])[0]: print(edge_idx[ii])
                
                matplotlib.use("Agg")
                fig, ax = plt.subplots(nrows=6, figsize=(7, 9.6))
                ax[0].title.set_text('ent')
                ax[0].plot(b1)
                ax[1].title.set_text('nc')
                ax[1].plot(nc)
                ax[2].title.set_text('expected')
                ax[2].plot(b1-nc)
                ax[3].title.set_text('out')
                ax[3].plot(B)
                # ax[4].title.set_text('lmd')
                # ax[4].plot(record["lmd"])
                # ax[5].title.set_text('total_coverage')
                # ax[5].plot(record["total_coverage"])
                plt.tight_layout()
                plt.savefig("{}/test/{}.png".format(param["save_iter_root"], iteration))
            ##### debug zone #####

            if verbose: print('phase-3', time.time()-s3)
            ########################################
            pos1, pos2 = cov_output[best_edge_idx][1]

            # Finalize
            print("f(SUe)={}, B(SUe)={}, marginal gain={}, F(S)={}, _coverage={}".format(
                (coverage/total_area)[best_edge_idx], 
                B[best_edge_idx], 
                objective[best_edge_idx], 
                FS, 
                len(_coverage)/total_area
                )
            )
            lazy_greedy["selected"].append(edge_idx[best_edge_idx]) ## test LG ##

            s4 = time.time()
            if is_indep[best_edge_idx]:
                lg_control = 1
                (a, b) = edge_idx[best_edge_idx]
                selected_graph.add_edge(a, b, weight=adj_mat[a][b])
                selected_vertices = selected_vertices | {pos1} | {pos2}

                FS = objective[best_edge_idx] + FS

                _coverage = _coverage.union(mat.get_fov_voxel(agent, pos1))
                _coverage = _coverage.union(mat.get_fov_voxel(agent, pos2))

                ##### draw graph #####
                if draw_iter_graph:
                    import networkx as nx
                    matplotlib.use("Agg")
                    fig = plt.figure(figsize=(10,10))
                    nx.draw_networkx(selected_graph, ax=fig.add_subplot())
                    fig.savefig("{}/iter-{}.png".format(save_iter_root, iteration))
                ##### draw graph #####
                print("Number of clusters:", len(list(nx.connected_components(selected_graph))))

            record["coverage"].append((coverage/total_area)[best_edge_idx])
            record["B"].append(B[best_edge_idx])
            record["marginal"].append(objective[best_edge_idx])
            record["B*lmd"].append(lda*B[best_edge_idx])
            record["lmd"].append(lda)
            record["total_coverage"].append(round(len(_coverage)/total_area, 2))
            # record["graph"].append(selected_graph.copy())
            record["n_clusters"].append(nc[best_edge_idx])

            # remove angles or edges
            try:
                subgoal_pos[pos1[:3]].remove(pos1[3:])
                subgoal_pos[pos2[:3]].remove(pos2[3:])
            except:
                pass

            # edge_idx.pop(best_edge_idx)
            edge_idx = np.delete(edge_idx, best_edge_idx, axis=0)
            B = np.delete(B, best_edge_idx, axis=0) ## test LG ##
            is_indep = np.delete(is_indep, best_edge_idx, axis=0) ## test LG ##
            nc = np.delete(nc, best_edge_idx, axis=0) ## test LG ##

            if verbose: print('phase-4', time.time()-s4)
            iteration += 1
            if verbose: print('Iteration time:', time.time()-s1) 
     
    except KeyboardInterrupt:
        pass    
    
    if method == "MRSM":
        trajectories, r_costs = mat.generate_path(selected_graph, 
                                                  mst=False, 
                                                  return_routing_cost=True
                                                  )
    elif method == "MRSIS-TSP":
        trajectories, r_costs = mat.generate_path_TSP(selected_graph)
    elif method == "MRSIS-MST":
        trajectories, r_costs = mat.generate_path(selected_graph, 
                                                  mst=True, 
                                                  return_routing_cost=True
                                                  )

    end = time.time()
    print('== Coverage:', '{}%'.format(round(len(_coverage)*100/total_area, 2)))
    print('== Time elapsed:', '{}'.format(end-start))
    param["Coverage"] = "{}%".format(round(len(_coverage)*100/total_area, 2))
    param["Time elapsed"] = "{} sec.".format(round(end-start))
    
    # idx to coordinates
    traj_coord = mat.idx2coord(selected_vertices=selected_vertices, 
                               vidx_i=vertex_idx.inverse, 
                               trajectories=trajectories)
    
    # print("Routing costs:", mat.cal_routing(traj_coord)) # including rotation cost
    print("Routing costs:", r_costs)
    param["Routing costs"] = r_costs

    if True: # plot graph
        from matplotlib.lines import Line2D
        # matplotlib.use('TkAgg')
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        cls = list(nx.connected_components(selected_graph))
        vertex_idx_inv = vertex_idx.inverse
        c = ["g","b", "r"]; labels=["uav1","uav2","uav3"]
        for edge in [(u,v) for (u, v, d) in selected_graph.edges(data=True)]:
            for i in range(len(cls)):
                if edge[0] in cls[i]:
                    color = c[i]
                    label = labels[i]
                    break
            ax.plot([vertex_idx_inv[edge[0]][0], vertex_idx_inv[edge[1]][0]],
                    [vertex_idx_inv[edge[0]][1], vertex_idx_inv[edge[1]][1]],
                    [vertex_idx_inv[edge[0]][2], vertex_idx_inv[edge[1]][2]],color = color)
            ax.scatter((vertex_idx_inv[edge[0]][0], vertex_idx_inv[edge[1]][0]),
                    (vertex_idx_inv[edge[0]][1], vertex_idx_inv[edge[1]][1]),
                    (vertex_idx_inv[edge[0]][2], vertex_idx_inv[edge[1]][2]), 
                    marker='o', color=color)
        handles, _ = plt.gca().get_legend_handles_labels()
        handles.extend([Line2D([0], [0], label=l, color=c_) for (l, c_) in zip(labels, c)])
        plt.legend(handles=handles)
        plt.savefig("{}/Graph.png".format(param["save_iter_root"]))
        
        # import networkx as nx
        # msts = nx.minimum_spanning_tree(selected_graph, algorithm="prim")
        # nx.draw_networkx(msts)
        
    # save model parameters
    json_object = json.dumps(param, indent=4)
    with open("{}/{}_{}_{}_{}.json".format(param["save_iter_root"], 
                                           param["method"],
                                           param["n_robots"], 
                                           param["lambda"], 
                                           param["routing_budget"]), "w") as outfile:
        outfile.write(json_object)

    if draw_iter_graph:
        fig, ax = plt.subplots(nrows=6, figsize=(7, 9.6))
        ax[0].title.set_text('coverage')
        ax[0].plot(record["coverage"])
        ax[1].title.set_text('B')
        ax[1].plot(record["B"])
        ax[2].title.set_text('marginal')
        ax[2].plot(record["marginal"])
        ax[3].title.set_text('B*lmd')
        ax[3].plot(record["B*lmd"])
        ax[4].title.set_text('lmd')
        ax[4].plot(record["lmd"])
        ax[5].title.set_text('total_coverage')
        ax[5].plot(record["total_coverage"])
        plt.tight_layout()
        plt.savefig("{}/iteration.png".format(param["save_iter_root"]))

    return {
        "traj_index": trajectories,
        "vertex_idx": vertex_idx,
        "traj_coord": traj_coord,
        "selected_graph": selected_graph,
    }

class MatroidPlanner(pomdp_py.Planner):
    def __init__(self, env, param):
        simulation = param["simulation"]
        voxel_base = 15 # cm
        unit_length = 30 # cm
        f = lambda x: int(x*unit_length/voxel_base)
        self.param = param
        stride = param["stride"]

        w, h, l = env._gridworld.width, env._gridworld.height, env._gridworld.length
        self.total_area = w*h*l - len(env.object_poses) # eliminate obstacles.
        if simulation:
            self.total_area = w*h*l
            if param["hexagonal"]:
                self.subgoal_set = {i:mat.generate_subgoal_coord_uav() for i in mat.hexagonal_packing_3d(w, l, h, R=stride)}
            else:
                w_range, h_range, l_range = [i for i in range(0, w, stride)], \
                    [i for i in range(0, h, stride)], [i for i in range(1, l, stride)]
                self.subgoal_set = {i:mat.generate_subgoal_coord_uav() for i in product(w_range, h_range, l_range)}
        else:
            w_range, h_range, l_range = [f(5), f(21)], [f(11), f(22), f(33)], [6, 8]
            self.subgoal_set = {i:mat.generate_subgoal_coord_uav() for i in product(w_range, h_range, l_range)}

        print("Number of subgoals (without angles):", len(self.subgoal_set))

        vertexes = list(self.subgoal_set.keys())
        self.graph = (cdist(vertexes, vertexes, 'euclidean'), 
                      bidict({tuple(v):i for i, v in enumerate(vertexes)}))

        if False:
            fp = "/home/yanshuo/Documents/3D-MOS/mos3d/experiments/results/matroid/2023-05-16-20-45-04.pickle"
            fp = "/home/yanshuo/Documents/Multiuav/model/v2-B-5/2023-05-26-19-35-17.pickle"
            fp = "/home/yanshuo/Documents/Multiuav/GEB/model/MRSM/MRSM_2_0.9_99999.pickle"
            fp = "/home/yanshuo/Documents/Multiuav/model/matroid-test/iter/GEB/MRSM/3-c/MRSM_2_0.2_99999.pickle"
            fp = "/home/yanshuo/Documents/Multiuav/model/matroid-test/iter/GEB/MRSIS-TSP/MRSIS-TSP_2_0.2_99999.pickle"
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
        
        self.next_best_subgoal = None # (x,y,z,thx,thy,thz)
        self.action_queue = []
        self.cost_fn = cost_fn.main()
        self.step_counter = 0
        self.B = param["routing_budget"] 
        self.p = True
        self.paths = None
        self.is_detect = None
        self.detect = None

    def plan(self, agent, env, max_time):
        
        print("Running MRSM...")
        results = MRSM(agent, 
                      self.subgoal_set, 
                      self.graph, 
                      budget=self.B, 
                      total_area=self.total_area,
                      param = self.param
        )
        
        import pickle
        with open("{}/{}_{}_{}_{}.pickle".format(self.param["save_iter_root"],
                                                 self.param["method"],
                                                 self.param["n_robots"], 
                                                 self.param["lambda"], 
                                                 self.param["routing_budget"]),
                  'wb') as f:
            pickle.dump(results, f)
        with open("{}/{}_{}_{}_{}-agent.pickle".format(self.param["save_iter_root"],
                                                 self.param["method"],
                                                 self.param["n_robots"], 
                                                 self.param["lambda"], 
                                                 self.param["routing_budget"]),
                  'wb') as f:
            pickle.dump(agent, f)
        
        return 
        
'''
def calCoverage(agent, 
         subgoal_pos, 
         graph_info,
         budget, 
         total_area,
         param, 
         verbose=True):
    
    #################
    _lambda = param["lambda"]
    n_clusters = param["n_robots"]
    #################
    
    with open(param["traj"], 'rb') as file:
        results = pickle.load(file)

    # _coverage = set()
    # for traj in results["traj_coord"]:
    #     for t in traj:
    #         _coverage = _coverage | get_fov_voxel(agent, t)

    _coverage = set()
    coverage_l = []
    for traj_0, traj_1 in zip(results["traj_coord"][0], results["traj_coord"][1]):
        _coverage = _coverage | get_fov_voxel(agent, traj_0)
        _coverage = _coverage | get_fov_voxel(agent, traj_1)
        coverage_l.append(round(len(_coverage)*100/total_area, 2))
            
    print('== Coverage:', '{}%'.format(round(len(_coverage)*100/total_area, 2)))
    return
'''
import math, ast
def str2tuple(x):
    try:
        math.isnan(x)
        return x
    except:
        return ast.literal_eval(x)

# def calCoverage(agent, 
#          subgoal_pos, 
#          graph_info,
#          budget, 
#          total_area,
#          param, 
#          verbose=True):
    
#     #################
#     _lambda = param["lambda"]
#     n_clusters = param["n_robots"]
#     #################
    
#     # with open(param["traj"], 'rb') as file:
#     #     results = pickle.load(file)
#     results = pd.read_csv(param["traj"])

#     results["log_x"] = results["log_x"].apply(str2tuple)
#     results["log_y"] = results["log_y"].apply(str2tuple)

#     _coverage = set()
#     coverage_l = []
#     for _, row in results.iterrows():
#         try:
#             math.isnan(row["log_x"])
#         except:
#             _coverage = _coverage | get_fov_voxel(agent, row["log_x"])
        
#         try:
#             math.isnan(row["log_y"])
#         except:
#             _coverage = _coverage | get_fov_voxel(agent, row["log_y"])
        
#         coverage_l.append(round(len(_coverage)*100/total_area, 2))

#     results["coverage"] = coverage_l

#     print('== Coverage:', '{}%'.format(round(len(_coverage)*100/total_area, 2)))
#     return

def calCoverage(agent, 
         subgoal_pos, 
         graph_info,
         budget, 
         total_area,
         param, 
         verbose=True):
    
    #################
    _lambda = param["lambda"]
    n_clusters = param["n_robots"]
    #################
    
    with open("/home/yanshuo/Documents/Multiuav/sim/check/MRSIS-MST_3_0.2_60.pickle", 'rb') as file:
       results_1 = pickle.load(file) 

    with open("/home/yanshuo/Documents/Multiuav/sim/check/MRSM_3_0.2_60.pickle", 'rb') as file:
       results_2 = pickle.load(file)   

    cov = set()
    for group in results_1["traj_coord"]:
        for coord in group:
            fov = get_fov_voxel(agent, coord)
            cov = cov | fov
    
    cov_2 = set()
    for group in results_2["traj_coord"]:
       for coord in group:
            fov = get_fov_voxel(agent, coord)
            cov_2 = cov_2 | fov    

    mrsis = [j for i in results_1["traj_coord"] for j in i]
    mrsm = [j for i in results_2["traj_coord"] for j in i]

    count = 0
    for i in mrsis:
        if i in mrsm:
            count += 1
    print(count/len(mrsis))
    
    count = 0
    for i in mrsm:
        if i in mrsis:
            count += 1
    print(count/len(mrsm))

    cov = list(cov)
    cov_2 = list(cov_2)

    # save cov and cov_2 as pickle file
    with open("/home/yanshuo/Documents/Multiuav/sim/check/cov.pickle", 'wb') as file:
        pickle.dump(cov, file)
    with open("/home/yanshuo/Documents/Multiuav/sim/check/cov_2.pickle", 'wb') as file:
        pickle.dump(cov_2, file)

    print('== Coverage:', '{}%'.format(round(len(cov)*100/total_area, 2)))
    return

class MRPlanner(pomdp_py.Planner):
    def __init__(self, env, param, stride = 3):
        simulation = param["simulation"]
        voxel_base = 15 # cm
        unit_length = 30 # cm
        f = lambda x: int(x*unit_length/voxel_base)
        self.param = param

        w, h, l = env._gridworld.width, env._gridworld.height, env._gridworld.length
        self.total_area = w*h*l - len(env.object_poses) # eliminate obstacles.
        if simulation:
            self.total_area = w*h*l
            w_range, h_range, l_range = [i for i in range(0, w, stride)], \
                [i for i in range(0, h, stride)], [i for i in range(1, l, stride)]
            self.subgoal_set = {i:mat.generate_subgoal_coord_uav() for i in product(w_range, h_range, l_range)}
        else:
            w_range, h_range, l_range = [f(5), f(21)], [f(11), f(22), f(33)], [6, 8]
            self.subgoal_set = {i:mat.generate_subgoal_coord_uav() for i in product(w_range, h_range, l_range)}

        vertexes = list(self.subgoal_set.keys())
        self.graph = (cdist(vertexes, vertexes, 'euclidean'), 
                      bidict({tuple(v):i for i, v in enumerate(vertexes)}))
        
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
        
        print("Calculating coverage...")
        results = calCoverage(agent, 
                              self.subgoal_set, 
                              self.graph, 
                              budget=self.B, 
                              total_area=self.total_area,
                              param = self.param
                              )
                
        return 
        
