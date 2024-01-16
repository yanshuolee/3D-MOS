class Config:
    scenarios = [(8, 1, 4, 10, 3.0, 500, 240)]
    # scenarios = [(12, 2, 3, 10, 3.0, 500, 240)]

    VIZ = False
    simulation = True
    _lambda = 0.9 #.9 / .9*3
    n_robots = 5
    budget = 20 #50
    method = "MRSM"
    # method = "MRSIS-TSP"
    # method = "MRSIS-MST"
    parallel = False
    save_iter_root = "/home/yanshuo/Documents/Multiuav/model/matroid-test/iter"
    scenario_txt_file = ""
    # scenario_txt_file = "/home/rsl/ysl/3D-MOS/mos3d/experiments/env_50x50x20.txt" # if specified, it will be loaded for simulation
    stride = 3

