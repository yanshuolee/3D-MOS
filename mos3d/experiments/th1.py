# import th
# import time
# # th.start()

# print('My program')
# s=time.time()
# while 1:
#     # print(th.status)
#     # time.sleep(1)
#     if time.time()-s > 3:
#         break

voxel_base = 15 # cm
unit_length = 30 # cm
f = lambda x: x*unit_length/voxel_base

def unit2voxel(x, y):
    _x = f(x)
    _y = f(y)
    return _x, _y

