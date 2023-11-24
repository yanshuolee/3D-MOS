import sys 
import numpy as np
import copy
import time
from scipy.spatial.transform import Rotation as scipyR
from scipy.spatial.distance import cdist

def coverage_fn(X):
    return len(X)

def flatten(l):
    return [item for sublist in l for item in sublist]

from scipy.stats import entropy
def cal_B(S_prime, n_vertexes):
    connected_cp = list(nx.connected_components(S_prime))
    # pz = np.array([len(cls)/n_edges for cls in connected_cp])
    pz = np.array([len(cls)/n_vertexes for cls in connected_cp])
    # B = entropy(pz, base=2) - len(connected_cp)
    
    if np.log2(len(pz)) == 0: # prevent nan
        ent = 0
    else:
        ent = entropy(pz, base=2)/np.log2(len(pz))
    B = (ent) - (len(connected_cp)/(n_vertexes//3)) # Normalization
    # return B, (entropy(pz, base=2)-len(connected_cp))/np.log2(len(pz)), entropy(pz, base=2)/np.log2(len(pz)), len(connected_cp)
    return (
        B,
        entropy(pz, base=2),
        - len(connected_cp)/(n_vertexes//2),
        ent
    )

from ray.util.multiprocessing import Pool
pool = Pool()
def balancing_fn(E, S, adj, budget, n_clusters, parallel=False, method=None):
    n_vertexes = adj.shape[0]
    balance = []
    is_indep = []
    b1s, b2s, b3s = [], [], []
    r = []
    nc = []

    route = sum([d["weight"] for (u, v, d) in S.edges(data=True)])

    ##### serial version #####
    if not parallel:
        for a, b in E:
            S_prime = S.copy()
            # Get SUe
            S_prime.add_edge(a, b, weight=adj[a][b])

            # compute route
            r.append(route+adj[a][b])

            # check if SUe is in independent set
            if method == "MRSM":
                is_indep.append(in_indepSet(S_prime, budget, n_clusters))
            elif method == "MRSIS-TSP":
                is_indep.append(in_indepSet_MRSIS(S_prime, budget, n_clusters, method=method))
            elif method == "MRSIS-MST":
                is_indep.append(in_indepSet_MRSIS(S_prime, budget, n_clusters, method=method))

            # Compute balance
            B, b1, b2, b3 = cal_B(S_prime, n_vertexes)
            balance.append(B)
            # ent.append(B);part.append(len(pz))
            b1s.append(b1); b2s.append(b2); b3s.append(b3)
            # record nc
            nc.append(len(list(nx.connected_components(S_prime))))
    
    ##### parallel version #####
    else:
        def fn(inputs):
            (S_prime, (a, b)) = inputs
            # Get SUe
            S_prime.add_edge(a, b, weight=adj[a][b])
            # check if SUe is in independent set
            _is_indep = in_indepSet(S_prime, budget, n_clusters)
            # is_indep.append()
            # Compute balance
            B, b1, b2, b3 = cal_B(S_prime, n_vertexes)
            # balance.append(B)
            # # ent.append(B);part.append(len(pz))
            # b1s.append(b1); b2s.append(b2); b3s.append(b3)
            return _is_indep, B, b1, b2, b3
        
        results = pool.map(fn, zip(
            [S.copy() for _ in range(len(E))],
            E
        ))
        for items in results:
            is_indep.append(items[0])
            balance.append(items[1])
            b1s.append(items[2])
            b2s.append(items[3])
            b3s.append(items[4])
    
    return (
        np.array(balance), 
        np.array(is_indep), 
        np.array(b1s), 
        np.array(b2s), 
        np.array(b3s),
        np.array(r),
        np.array(nc),
    )

import networkx as nx
from .gcb_utils import OrderedSet
from random import sample
def sample_edges(E, subgoal_pos, vertex_idx, adj_mat, n, _type="random"):
    valid = False
    while not valid:
        if _type == "random":
            idx = np.random.choice(len(E)-1, size=n, replace=False)
        elif _type == "min":
            dist = [adj_mat[i][j] for (i,j) in E]
            idxs = np.argsort(dist)
            idx = idxs[:n] # Do not select edges close to border. Lead to negative marginal.

        G = nx.Graph()
        V = OrderedSet()
        for i in idx:
            (a, b) = E.pop(i)
            G.add_edge(a, b, weight=adj_mat[a][b])

            V = V | {vertex_idx[a]+sample(subgoal_pos[vertex_idx[a]], 1)[0]}
            V = V | {vertex_idx[b]+sample(subgoal_pos[vertex_idx[b]], 1)[0]}


        if len(list(nx.connected_components(G))) == n:
            valid = True

    return G, V

def get_fov_voxel(agent, pos):
    volume = agent.observation_model._gridworld.robot.camera_model.get_volume(pos)
    filtered_volume = {tuple(v) for v in volume if agent.observation_model._gridworld.in_boundary(v)}
    return filtered_volume

def coverage_fn(X):
    return len(X)

def compute_coverage_fn(x, agent, subgoal_pos, vertex_idx, _coverage):
    coverage = len(_coverage)
    pos = []
    # vertex 1
    for v1 in x:
        cov1 = [coverage_fn(_coverage.union(get_fov_voxel(agent, vertex_idx[v1]+i))) for i in subgoal_pos[vertex_idx[v1]]]
        if len(cov1) == 0:
            pos.append(vertex_idx[v1]+(0, 0, 0, 1))
            continue
        v1_idx = np.argmax(cov1)
        coverage += (cov1[v1_idx]-len(_coverage))
        pos1 = vertex_idx[v1]+subgoal_pos[vertex_idx[v1]][v1_idx]
        pos.append(pos1)
    
    return coverage, pos

def compute_coverage_fn_parallel(inputs):
    (x, agent, subgoal_pos, vertex_idx, _coverage) = inputs
    coverage = len(_coverage)
    pos = []
    # vertex 1
    for v1 in x:
        cov1 = [coverage_fn(_coverage.union(get_fov_voxel(agent, vertex_idx[v1]+i))) for i in subgoal_pos[vertex_idx[v1]]]
        if len(cov1) == 0:
            pos.append(vertex_idx[v1]+(0, 0, 0, 1))
            continue
        v1_idx = np.argmax(cov1)
        coverage += (cov1[v1_idx]-len(_coverage))
        pos1 = vertex_idx[v1]+subgoal_pos[vertex_idx[v1]][v1_idx]
        pos.append(pos1)
    
    return coverage, pos

def generate_subgoal_coord(c):
    return [
        (0, 0, 0, 1), # -x [0., 0., 0.]
        (0, 1, 0, 0), # +x [180.,   0., 180.]
        (0, 0, c, c), # -y [ 0.,  0., 90.]
        (0, 0, -c, c),# +y [  0.,   0., -90.]
        (0, c, 0, c), # +z [  0., -90.,   0.]
        (0, -c, 0, c) # -z
    ]

def generate_subgoal_coord_uav(angle=60):
    subg = []
    for ang in range(0, 360, angle):
        a = scipyR.from_euler('xyz', [0, 0, ang], degrees=True).as_quat()
        subg.append(tuple(a))
    return subg

def get_leaf(G):
    return [x for x in G.nodes() if G.degree(x)==1]

def traverse_dist(start, parent, sp, n_childs, paths):
    childs = list(nx.all_neighbors(sp, start))
    if parent in childs: childs.remove(parent)
    if len(childs) == 0:
        # print(start)
        paths.append(start)
        return start
    
    for child_idx in np.argsort([n_childs[i] if i in n_childs else 0 for i in childs]):
        child = childs[child_idx]
        # print(start)
        paths.append(start)
        node = traverse_dist(child, start, sp, n_childs, paths)
    # print(start)
    paths.append(start)

def generate_path(selected_graph, mst=True, return_routing_cost=False):
    paths = []
    route = []
    cls = list(nx.connected_components(selected_graph))
    
    if mst:
        msts = nx.minimum_spanning_tree(selected_graph, algorithm="prim")
        T = msts
    else:
        T = selected_graph
        
    for c in cls:
        sp = T.subgraph(c)
        leaves = get_leaf(sp)
        start = leaves[np.argmax([list(nx.shortest_path_length(sp, i).values())[-1]+1 for i in leaves])]
        leaves.remove(start)

        uniq = {i:0 for i in sp.nodes}
        tmp = [i for i in nx.bfs_edges(sp,start)]
        X = np.array([x for x,y in tmp])
        Y = np.array([y for x,y in tmp])
        for node in leaves:
            while True:
                node = np.where(node == Y)[0]
                if len(node) == 0:
                    break
                node = node[0]
                if uniq[X[node]] < uniq[Y[node]]:
                    uniq[X[node]] = uniq[Y[node]]
                    uniq[X[node]] += 1
                if (uniq[Y[node]] == 0) and (uniq[X[node]] == 0): 
                    uniq[X[node]] += 1
                node = X[node]

        # print("Routing")
        arr = []
        traverse_dist(start, -1, sp, uniq, arr)
        _max = -1
        for node in leaves:
            x = np.where(node == np.array(arr))[0][0]
            if x > _max:
                _max = x
        paths.append(arr[:_max+1])

        if return_routing_cost:
            R = 0
            path = arr[:_max+1]
            for j in range(len(path)-1):
                R += sp.adj[path[j]][path[j+1]]["weight"]
            route.append(R)

    if return_routing_cost:
        return paths, route
    else: 
        return paths

def generate_path_TSP(selected_graph):
    paths = []
    route = []
    tsp = nx.approximation.traveling_salesman_problem
    cls = list(nx.connected_components(selected_graph))

    for c in cls:
        path = tsp(selected_graph, cycle=False, nodes=c)
        paths.append(path)

        R = 0
        for j in range(len(path)-1):
            R += selected_graph.adj[path[j]][path[j+1]]["weight"]
        route.append(R)

    return paths, route

def idx2coord(selected_vertices, vidx_i, trajectories):
    selected_vertices_ = {}
    for i in selected_vertices:
        if i[:3] not in selected_vertices_:
            selected_vertices_[i[:3]] = [i[3:]]
        else:
            selected_vertices_[i[:3]].append(i[3:]) 

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
    return traj_coord

def cal_routing(trajs):
    routing_len = []
    for traj in trajs:
        dist = 0
        for i in range(1,len(traj)):
            dist += cdist([traj[i-1]], [traj[i]], 'euclidean')[0][0]
        routing_len.append(dist)
    return routing_len

# MRSM version
def in_indepSet(SUe, B, k): 
    # check N >= k
    connected_cp = list(nx.connected_components(SUe))
    if len(connected_cp) < k:
        # print(connected_cp, "<k")
        return False

    # check cycle
    try:
        nx.find_cycle(SUe)
        return False
    except:
        pass

    # check routing
    _, routing = generate_path(SUe, mst=False, return_routing_cost=True)
    routing = np.array(routing)

    if (routing < B).all():
        return True
    else:
        # print(connected_cp, "routing", routing)
        return False

# MRSIS version
def in_indepSet_MRSIS(SUe, B, k, method): 
    # check N >= k
    connected_cp = list(nx.connected_components(SUe))
    if len(connected_cp) < k:
        # print(connected_cp, "<k")
        return False

    # check routing
    if method == "MRSIS-TSP":
        tsp = nx.approximation.traveling_salesman_problem
        routing = np.zeros(len(connected_cp))
        for i in range(len(routing)):
            path = tsp(SUe, cycle=False, nodes=connected_cp[i])
            for j in range(len(path)-1):
                routing[i] += SUe.adj[path[j]][path[j+1]]["weight"]
    elif method == "MRSIS-MST":
        T = nx.minimum_spanning_tree(SUe, algorithm="prim")
        routing = np.zeros(len(connected_cp))
        for u, v, d in T.edges(data=True):
            for idx, c in enumerate(connected_cp):
                if (u in c) or (v in c):
                    routing[idx] += d["weight"]
                    break
        routing *= 2

    if (routing < B).all():
        return True
    else:
        # print(connected_cp, "routing", routing)
        return False

if __name__ == '__main__':
    # graph = [[0, 2, 0, 6, 0],
    #         [2, 0, 3, 8, 5],
    #         [0, 3, 0, 0, 7],
    #         [6, 8, 0, 0, 9],
    #         [0, 5, 7, 9, 0]]
    # g = Graph(graph)
    # mst = g.primMST()

    # import numpy as np
    # graph = np.random.rand(10,10)
    # g = Graph(graph)
    # g.primMST()

    import numpy as np
    import time
    from itertools import product
    
    import matplotlib.pyplot as plt
    subgoal_set = []
    w, h, l, stride = 8,8,8, 3
    w_range, h_range, l_range = [i for i in range(0, w, stride)], \
                [i for i in range(0, h, stride)], [i for i in range(0, l, stride)]
    for i in product(w_range, h_range, l_range ):
        subgoal_set.append(list(i))

    subgoal_set[0], subgoal_set[15] = subgoal_set[15], subgoal_set[0]

    graph = cdist(subgoal_set, subgoal_set, 'cityblock') # cityblock euclidean

    g = Graph(graph)
    mst = g.primMST()
    g.traverse(4)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    subgoal_set = np.array(subgoal_set)

    for start, end, length in mst:
        ax.plot([subgoal_set[int(start)][0], subgoal_set[int(end)][0]],
                [subgoal_set[int(start)][1], subgoal_set[int(end)][1]],
                [subgoal_set[int(start)][2], subgoal_set[int(end)][2]],color = 'g')

    ax.scatter(subgoal_set[:,0],subgoal_set[:,1],subgoal_set[:,2], marker='o')
    ax.set(xlabel='x', ylabel='y', zlabel='z')

    plt.show()
    print()

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # for start, end, length in mst:
    #     ax.plot([vertexes[int(start)][0], vertexes[int(end)][0]],
    #             [vertexes[int(start)][1], vertexes[int(end)][1]],
    #             [vertexes[int(start)][2], vertexes[int(end)][2]],color = 'g')

    # ax.scatter(vertexes[:,0],vertexes[:,1],vertexes[:,2], marker='o')
    # ax.set(xlabel='x', ylabel='y', zlabel='z')

    # plt.show()