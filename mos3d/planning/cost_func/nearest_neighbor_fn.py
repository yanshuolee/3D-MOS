import numpy as np

""" Reference: https://jupyter.brynmawr.edu/services/public/dblank/jupyter.cs/FLAIRS-2015/TSPv3.ipynb """
def distance(A, B): 
    "The distance between two points."
    total = 0
    for x, y in zip(A, B):
        total = total + (x-y)**2
    return np.sqrt(total)

def first(collection):
    "Start iterating over collection, and return the first element."
    return next(iter(collection))

def nn_tsp(cities):
    """Start the tour at the first city; at each step extend the tour 
    by moving from the previous city to its nearest neighbor 
    that has not yet been visited."""
    start = first(cities)
    tour = [start]
    unvisited = set(cities - {start})
    total_dist = 0
    while unvisited:
        C, dist = nearest_neighbor(tour[-1], unvisited)
        total_dist += dist
        tour.append(C)
        unvisited.remove(C)
    return tour, total_dist

def nearest_neighbor(A, cities):
    "Find the city in cities that is nearest to city A."
    min_coord = min(cities, key=lambda c: distance(c, A))
    return min_coord, distance(min_coord, A)
