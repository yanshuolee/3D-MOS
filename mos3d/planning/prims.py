# A Python program for Prim's Minimum Spanning Tree (MST) algorithm.

# The program is for adjacency matrix representation of the graph

import sys # Library for INT_MAX
import numpy as np
import copy
import time

class Graph():

    def __init__(self, graph):
        self.graph = graph
        self.V = len(graph)
        self.mst = None

    def traverse(self, dst, g):
        cost = 0
        path = [dst]
        
        while (dst != 0):
            idx = int(dst - 1)
            parent = self.mst[idx][0]
            # cost += self.mst[idx][2]
            dst = parent
            path.append(int(dst))
        
        g = g|set(path)
        # g.extend(path)
        # f=time.time()
        # idx = np.unique(g)
        # print(time.time()-f)

        # f=time.time()
        idx = np.array(list(g)) #np.array(list(set(g)))
        # print(time.time()-f)

        idx -= 1
        idx = idx[idx>=0]
        cost = self.mst[idx, 2].sum()
        
        return cost, path

    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minKey(self, key, mstSet):

        # Initilaize min value
        min = sys.maxsize

        for v in range(self.V):
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                min_index = v

        return min_index

    # Function to construct and print MST for a graph
    # represented using adjacency matrix representation
    def primMST(self):

        # Key values used to pick minimum weight edge in cut
        key = [sys.maxsize] * self.V
        parent = [None] * self.V # Array to store constructed MST
        # Make key 0 so that this vertex is picked as first vertex
        key[0] = 0
        mstSet = [False] * self.V

        parent[0] = -1 # First node is always the root of

        for cout in range(self.V):

            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minKey(key, mstSet)

            # Put the minimum distance vertex in
            # the shortest path tree
            mstSet[u] = True

            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shotest path tree
            for v in range(self.V):

                # graph[u][v] is non zero only for adjacent vertices of m
                # mstSet[v] is false for vertices not yet included in MST
                # Update the key only if graph[u][v] is smaller than key[v]
                if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
                        key[v] = self.graph[u][v]
                        parent[v] = u

        mst = np.array([[parent[i], i, self.graph[i][ parent[i] ]] for i in range(1, self.V)])
        self.mst = mst
        return mst
        
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
    from scipy.spatial.distance import cdist
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