import sys 
import numpy as np
import copy
import time
from scipy.spatial.transform import Rotation as scipyR
from scipy.spatial.distance import cdist
import gc
import psutil

def coverage_fn(X):
    return len(X)

def flatten(l):
    return [item for sublist in l for item in sublist]

def hexagonal_packing_3d(width, height, depth, R = 1, plot=False):
    # Constants for hexagon
    SQRT3 = np.sqrt(3)

    # Function to calculate hexagon vertices
    def hexagon(x, y, z):
        return [
            (x + R * np.cos(np.pi * i / 3), y + R * np.sin(np.pi * i / 3), z)
            for i in range(6)
        ]

    # Calculate the number of hexagons needed in X, Y, and Z directions
    hexagons_in_x = int(width / (R * SQRT3))
    hexagons_in_y = int(height / (R * 1.5))
    hexagons_in_z = int(depth / R)

    # Adjust width, height, and depth based on the maximum number of hexagons
    width = hexagons_in_x * R * SQRT3
    height = hexagons_in_y * R * 1.5
    depth = hexagons_in_z * R

    centroids = []  # List to store the centroids

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    for i in range(hexagons_in_x):
        for j in range(hexagons_in_y):
            for k in range(hexagons_in_z):
                # Calculate position of each hexagon
                x = i * R * SQRT3
                y = j * R * 1.5
                z = k * R

                if i % 2 == 1:
                    y += R * 0.75
                
                x, y, z = x+R, y+R, z+R
                # Append the centroid to the list
                centroids.append((x, y, z))
                
                hexagon_coords = hexagon(x, y, z)
                for (x, y, z) in hexagon_coords:
                    centroids.append((x, y, z))
                hexagon_coords.append(hexagon_coords[0])
                hexagon_x, hexagon_y, hexagon_z = zip(*hexagon_coords)

                # Plot hexagon
                if plot:
                    ax.plot(hexagon_x, hexagon_y, hexagon_z, color="b")

    if plot:    
        # Set limits
        ax.set_xlim([0, width])
        ax.set_ylim([0, height])
        ax.set_zlim([0, depth])
    
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
        plt.show()

    return centroids

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
    B = (ent) - (len(connected_cp)/(n_vertexes//2)) # Normalization
    # return B, (entropy(pz, base=2)-len(connected_cp))/np.log2(len(pz)), entropy(pz, base=2)/np.log2(len(pz)), len(connected_cp)
    return (
        B,
        entropy(pz, base=2),
        - len(connected_cp)/(n_vertexes//2),
        ent
    )

def auto_garbage_collect(pct=80.0):
    if psutil.virtual_memory().percent >= pct:
        gc.collect()
    return

def hexagonal_packing_3d(_X, _Y, _Z, R=1, plot=False):
    (width, height, depth) = (_X, _Y, _Z)
    if plot:
        import matplotlib.pyplot as plt
    
    # Constants for hexagon
    SQRT3 = np.sqrt(3)

    # Function to calculate hexagon vertices
    def hexagon(x, y, z):
        return [
            (x + R * np.cos(np.pi * i / 3), y + R * np.sin(np.pi * i / 3), z)
            for i in range(6)
        ]

    # Calculate the number of hexagons needed in X, Y, and Z directions
    hexagons_in_x = int(width / (R * SQRT3))
    hexagons_in_y = int(height / (R * 1.5))
    hexagons_in_z = int(depth / R)

    # Adjust width, height, and depth based on the maximum number of hexagons
    width = hexagons_in_x * R * SQRT3
    height = hexagons_in_y * R * 1.5
    depth = hexagons_in_z * R

    centroids = []  # List to store the centroids

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    for i in range(hexagons_in_x):
        for j in range(hexagons_in_y):
            for k in range(hexagons_in_z):
                # Calculate position of each hexagon
                x = i * R * SQRT3
                y = j * R * 1.5
                z = k * R

                if i % 2 == 1:
                    y += R * 0.75
                
                x, y, z = x+R, y+R, z+R
                # Append the centroid to the list
                centroids.append((x, y, z))
                
                hexagon_coords = hexagon(x, y, z)
                for (x, y, z) in hexagon_coords:
                    centroids.append((x, y, z))
                hexagon_coords.append(hexagon_coords[0])
                hexagon_x, hexagon_y, hexagon_z = zip(*hexagon_coords)

                # Plot hexagon
                if plot:
                    ax.plot(hexagon_x, hexagon_y, hexagon_z, color="b")

    if plot:    
        # Set limits
        ax.set_xlim([0, width])
        ax.set_ylim([0, height])
        ax.set_zlim([0, depth])
    
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
        plt.show()

    return centroids

def Cov(agent, subgoal_pos, total_area):
    _coverage = set()
    for i in subgoal_pos:
        for j in subgoal_pos[i]:
            # print(i+j)
            _coverage = _coverage | get_fov_voxel(agent, i+j)
    return len(_coverage) / total_area

def balancing_fn(E, S, adj, budget, n_clusters, pool=None, method=None):
    n_vertexes = adj.shape[0]
    balance = []
    is_indep = []
    b1s, b2s, b3s = [], [], []
    r = []
    nc = []

    route = sum([d["weight"] for (u, v, d) in S.edges(data=True)])

    ##### serial version #####
    if not pool:
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
            (_S, (a, b)) = inputs
            S_prime = _S.copy()
            # Get SUe
            S_prime.add_edge(a, b, weight=adj[a][b])
            
            # compute route
            _r = route+adj[a][b]

            # check if SUe is in independent set
            if method == "MRSM":
                _is_indep = in_indepSet(S_prime, budget, n_clusters)
            elif method == "MRSIS-TSP":
                _is_indep = in_indepSet_MRSIS(S_prime, budget, n_clusters, method=method)
            elif method == "MRSIS-MST":
                _is_indep = in_indepSet_MRSIS(S_prime, budget, n_clusters, method=method)
            
            # Compute balance
            B, b1, b2, b3 = cal_B(S_prime, n_vertexes)
            nc = len(list(nx.connected_components(S_prime)))
            
            del S_prime
            auto_garbage_collect()

            return (
                _is_indep, 
                B, b1, b2, b3, _r, 
                nc, 
            )
        
        results = pool.map(fn, 
                           zip([S]*len(E), 
                               E,
                        )
        )

        results = np.array(results)
        is_indep = results[:, 0].astype(bool)
        balance = results[:, 1]
        b1s = results[:, 2]
        b2s = results[:, 3]
        b3s = results[:, 4]
        r = results[:, 5]
        nc = results[:, 6]

        del results 

        # for items in results:
        #     is_indep.append(items[0])
        #     balance.append(items[1])
        #     b1s.append(items[2])
        #     b2s.append(items[3])
        #     b3s.append(items[4])
        #     r.append(items[5])
        #     nc.append(items[6])

    
    auto_garbage_collect()

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
            idx = idxs[:n] 
            
            # Check connected_components
            remove_list = []
            s = set()
            for i in idx:
                (a, b) = E[i]
                N = len(s)
                s = s | {a,b}
                if len(s) - N < 2:
                    remove_list.append(i)
            counter = n
            for i in remove_list:
                idx = np.delete(idx, np.where(idx==i)[0][0])
                while True:
                    assert counter < len(idxs), "Too many clusters. Try to reduce number of robots."
                    edge_id = idxs[counter]
                    counter += 1
                    (a, b) = E[edge_id]
                    if (a not in s) and (b not in s):
                        idx = np.append(idx, edge_id)
                        s = s | {a,b}
                        break

        G = nx.Graph()
        V = OrderedSet()
        for i in idx:
            (a, b) = E[i]
            G.add_edge(a, b, weight=adj_mat[a][b]) 

            V = V | {vertex_idx[a]+sample(subgoal_pos[vertex_idx[a]], 1)[0]}
            V = V | {vertex_idx[b]+sample(subgoal_pos[vertex_idx[b]], 1)[0]}

        # if valid is false, E is not the original one.
        if len(list(nx.connected_components(G))) == n:
            valid = True

    # for i in idx: print(E[i])
    E = np.delete(E, idx, axis=0)

    return G, V, E

def get_fov_voxel(agent, pos):
    volume = agent.observation_model._gridworld.robot.camera_model.get_volume(pos)
    # filtered_volume = {tuple(v) for v in volume if agent.observation_model._gridworld.in_boundary(v)}
    bool_idx =  (volume[:,0] >= 0) & (volume[:,0] < agent.observation_model._gridworld.width) & \
                (volume[:,1] >= 0) & (volume[:,1] < agent.observation_model._gridworld.length) & \
                (volume[:,2] >= 0) & (volume[:,2] < agent.observation_model._gridworld.height)
    filtered_volume = {tuple(v) for v in volume[bool_idx]}
    return filtered_volume

def coverage_fn(X):
    return len(X)

def compute_coverage_fn(x, agent, subgoal_pos, vertex_idx, _coverage):
    coverage = len(_coverage)
    pos = []
    # vertex 1
    for v1 in x:
        cov1 = [coverage_fn(_coverage | get_fov_voxel(agent, vertex_idx[v1]+i)) for i in subgoal_pos[vertex_idx[v1]]]
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
        cov1 = [coverage_fn(_coverage | get_fov_voxel(agent, vertex_idx[v1]+i)) for i in subgoal_pos[vertex_idx[v1]]]
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