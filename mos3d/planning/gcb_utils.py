import sys
import collections
from scipy.spatial.distance import cdist

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import numpy as np
from scipy import ndimage

def calcDistVec(coord, offset):
    n = coord.shape[0]
    dist = np.zeros((n-offset+1, 1))
    temp = coord[:-offset, :] - coord[offset:, :]
    dist[1:] = np.sqrt(np.sum(temp**2, axis=1)).reshape((len(temp), 1))
    return dist

def solveTSP(vertex):
    N = len(vertex)
    itt = 0
    maxItt = min(20*N, 1e5)
    noChange = 0
    order = np.arange(N)

    while (itt < maxItt) and (noChange < N):
        dist = calcDistVec(vertex[order, :], 1)
        flip = np.mod(itt, N-3) + 2

        untie = dist[:N-flip] + dist[flip:]
        shuffledDist = calcDistVec(vertex[order, :], flip)
        connect = shuffledDist[:-1] + shuffledDist[1:]
        benefit = connect - untie

        localMin = ndimage.grey_erosion(benefit, footprint=np.ones((2*flip+1,1)))
        minimasInd = np.where(localMin == benefit)[0]
        reqFlips = minimasInd[(benefit[minimasInd] < -np.finfo(np.float64).eps).flatten()]

        prevOrd = order.copy() 
        for n in range(len(reqFlips)):
            order[reqFlips[n]:reqFlips[n]+flip] = order[reqFlips[n]:reqFlips[n]+flip][::-1]

        if (order == prevOrd).all():
            noChange = noChange + 1
        else:
            noChange = 0

        itt += 1

    return vertex[order, :], sum(dist)[0]

class Graph():

    def __init__(self, vertices):
        self.V = len(vertices) # type: set
        v = [list(i) for i in vertices]
        self.graph = cdist(v, v, 'euclidean')

	# A utility function to find the vertex with
	# minimum distance value, from the set of vertices
	# not yet included in shortest path tree
    def minDistance(self, dist, sptSet):

        # Initilaize minimum distance for next node
        min = sys.maxsize

        # Search not nearest vertex not in the
        # shortest path tree
        for v in range(self.V):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v

        return min_index

	# Funtion that implements Dijkstra's single source
	# shortest path algorithm for a graph represented
	# using adjacency matrix representation
    def dijkstra(self, src):

        dist = [sys.maxsize] * self.V
        dist[src] = 0
        sptSet = [False] * self.V

        for cout in range(self.V):

            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minDistance(dist, sptSet)

            # Put the minimum distance vertex in the
            # shotest path tree
            sptSet[u] = True

            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shotest path tree
            for v in range(self.V):
                if self.graph[u][v] > 0 and \
                sptSet[v] == False and \
                dist[v] > dist[u] + self.graph[u][v]:
                    dist[v] = dist[u] + self.graph[u][v]

        return sum(dist)

class OrderedSet(collections.MutableSet):
    """From https://code.activestate.com/recipes/576694/"""
    
    def __init__(self, iterable=None):
        self.end = end = [] 
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:        
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)
