from itertools import product
import random
from mos3d.planning import matroid_utils as mat
import pickle

def generate_random_traj(seed, w, h, l, stride, n_cls):
    random.seed(seed)
    w_range, h_range, l_range = [i for i in range(0, w, stride)], \
        [i for i in range(0, h, stride)], [i for i in range(1, l, stride)]
    subgoal_set = [i+j for i in product(w_range, h_range, l_range) for j in mat.generate_subgoal_coord_uav()]

    random.shuffle(subgoal_set)
    traj = []
    for i in range(0, len(subgoal_set), len(subgoal_set)//n_cls):
        traj.append(subgoal_set[i:i+len(subgoal_set)//n_cls])

    return traj


if __name__ == "__main__":
    seeds = [random.randint(1, 1000000) for i in range(100)]
    for idx, seed in enumerate(seeds):
        traj = generate_random_traj(seed=seed, w=12, h=12, l=12, stride=3, n_cls=3)
        results = {"traj_coord":traj}
        with open("/home/yanshuo/Documents/3D-MOS/mos3d/experiments/results/matroid/rand-{}.pickle".format(idx),
                  'wb') as f:
            pickle.dump(results, f)
