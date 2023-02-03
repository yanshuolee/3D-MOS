import sys
import collections
from scipy.spatial.distance import cdist
import numpy as np
from scipy.spatial.transform import Rotation as scipyR

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])[0]

def get_fov_voxel(agent, pos):
    volume = agent.observation_model._gridworld.robot.camera_model.get_volume(pos)
    filtered_volume = {tuple(v) for v in volume if agent.observation_model._gridworld.in_boundary(v)}
    return filtered_volume

def coverage_fn(X):
    return len(X)

def compute_coverage_fn(agent, x, _coverage, current_cov):
    voxels = get_fov_voxel(agent, x)
    del_f = coverage_fn(_coverage.union(voxels)) - current_cov
    return del_f

def generate_subgoal_coord(xyz, c):
    x, y, z = xyz
    return [
        (x, y, z, 0, 0, 0, 1), # -x [0., 0., 0.]
        (x, y, z, 0, 1, 0, 0), # +x [180.,   0., 180.]
        (x, y, z, 0, 0, c, c), # -y [ 0.,  0., 90.]
        (x, y, z, 0, 0, -c, c),# +y [  0.,   0., -90.]
        (x, y, z, 0, c, 0, c), # +z [  0., -90.,   0.]
        (x, y, z, 0, -c, 0, c) # -z
    ]

def generate_subgoal_coord_uav(xyz, angle=60):
    x, y, z = xyz
    subg = []
    for ang in range(0, 360, angle):
        a = [x, y, z]
        a.extend(scipyR.from_euler('xyz', [0, 0, ang], degrees=True).as_quat())
        subg.append(tuple(a))
    return subg

def generate_subgoal_union(s1, s2):
    union = s1|{s2[:3]}
    # v = np.array([list(i) for i in union])
    # return v
    return union

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
