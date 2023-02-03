# import time
# from concurrent.futures import ThreadPoolExecutor

# # global status
# status = False

# def test(n):
#     global status
#     for i in range(n):
#         # print(i, end=' ')
#         if i%3==0:
#             status = True
#             print('set to', status)
#         else:
#             status = False
#         time.sleep(1)

# executor = ThreadPoolExecutor()  # 設定一個執行 Thread 的啟動器

# def start():
#     a = executor.submit(test, 5)     # 啟動第一個 test 函式

# for i in range(10):
#     print('out:', i)
#     time.sleep(0.2)

# executor.shutdown()
world = "\n64\n88\n13\n\n" # to voxel

voxel_base = 15 # cm
unit_length = 30 # cm
f = lambda x: int(x*unit_length/voxel_base)

# Build obstacle
# for x in range(0, 25):
#     for y in range(8, 56):
#         # print("cube {} {} {} obstacle".format(f(y), f(x), 0)) # 3dmos coord
#         world += "cube {} {} {} obstacle\n".format(f(y), f(x), 0)

# for x in range(33, 63):
#     for y in range(8, 56):
#         # print("cube {} {} {} obstacle".format(f(y), f(x), 0)) # 3dmos coord
#         world += "cube {} {} {} obstacle\n".format(f(y), f(x), 0)

# for x in range(63, 74): # Build the arch
#     for y in range(8, 49):
#         # print("cube {} {} {} obstacle".format(f(y), f(x), 0)) # 3dmos coord
#         world += "cube {} {} {} obstacle\n".format(f(y), f(x), 0)

# for x in range(74, 105):
#     for y in range(8, 56):
#         # print("cube {} {} {} obstacle".format(f(y), f(x), 0)) # 3dmos coord
#         world += "cube {} {} {} obstacle\n".format(f(y), f(x), 0)

# for x in range(113, 150):
#     for y in range(8, 56):
#         # print("cube {} {} {} obstacle".format(f(y), f(x), 0)) # 3dmos coord
#         world += "cube {} {} {} obstacle\n".format(f(y), f(x), 0)

for x in range(f(22), f(33), 1):
    for y in range(f(34), f(45), 1):
        for z in range(13):
            world += "cube {} {} {} obstacle\n".format(x, y, z)

# Build obstacle
# world += "\n---\nrobot 8 58 0 0 0 0 occlusion 45 1.0 0.1 20\n"
world += "\n---\nrobot 10 10 0 0 0 0 occlusion 45 1.0 0.1 20\n" # to voxel

with open("./GEB.txt", "w") as f:
    f.write(world)

print