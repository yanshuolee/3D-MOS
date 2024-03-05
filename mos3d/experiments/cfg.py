class Config:
    ########### Simulation parameters ###########
    # scenarios = [(8, 1, 4, 10, 3.0, 500, 240)]
    scenarios = [(12, 2, 3, 10, 3.0, 500, 240)]
    # scenarios = [(24, 2, 3, 10, 3.0, 500, 240)]
    VIZ = False
    simulation = True
    _lambda = 0.9 #.9 / .9*3
    n_robots = 3
    budget = 20
    method = "MRSM"
    # method = "MRSIS-TSP"
    # method = "MRSIS-MST"
    parallel = 0
    save_iter_root = "/home/yanshuo/Documents/Multiuav/model/matroid-test/iter"
    scenario_txt_file = ""
    # scenario_txt_file = "/home/rsl/ysl/3D-MOS/mos3d/experiments/env_50x50x20.txt" # if specified, it will be loaded for simulation
    stride = 5
    hexagonal = True
    lazy_greedy_mode = False
    ########### Simulation parameters ###########

    ########### GEB parameters ###########
    # scenarios = [(8, 1, 4, 10, 3.0, 500, 240)]
    # VIZ = False
    # simulation = False
    # _lambda = 0.5 
    # n_robots = 2
    # budget = 99999
    # method = "MRSM"
    # # method = "MRSIS-TSP"
    # # method = "MRSIS-MST"
    # parallel = False
    # save_iter_root = "/home/yanshuo/Documents/Multiuav/model/matroid-test/iter/GEB"
    # scenario_txt_file = ""
    # stride = -1
    # hexagonal = False
    # lazy_greedy_mode = True
    ########### GEB parameters ###########

