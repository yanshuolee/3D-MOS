from mos3d.planning import gcb_utils

def objective_fn(vertex): #  start_idx=0
    """Dijkstra's Algorithm"""
    if len(vertex) == 0:
        return 0

    # version 1
    # s1=time.time()
    # g = gcb_utils.Graph(vertex)
    # print('g', time.time()-s1)
    # s2=time.time()
    # cost = g.dijkstra(start_idx)
    # print('co', time.time()-s2)
    # del g

    # version 2
    # v = [list(i) for i in vertex]
    # g = cdist(v, v, 'euclidean')
    # g = gcb_utils.csr_matrix(g)
    # dist_matrix, predecessors = gcb_utils.dijkstra(csgraph=g, directed=False, indices=0, return_predecessors=True)
    # cost = dist_matrix.sum()

    # version 3: approximation
    ordered_coord, cost = gcb_utils.solveTSP(vertex)

    return cost

def approximation_fn(vertex):
    if len(vertex) == 0:
        return 0
    
    ordered_coord, cost = gcb_utils.solveTSP(vertex)
    return cost

def main():
    fn = approximation_fn

    return fn