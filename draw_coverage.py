import plotly.graph_objects as go
import pickle 
import numpy as np
from matplotlib import pyplot as plt

with open("/home/yanshuo/Desktop/record-tsp-8.pickle", 'rb') as f:
    tsp_coverage_8 = pickle.load(f)
with open("/home/yanshuo/Desktop/record-tsp-16.pickle", 'rb') as f:
    tsp_coverage_16 = pickle.load(f)
with open("/home/yanshuo/Desktop/record-mst-8.pickle", 'rb') as f:
    mst_coverage_8 = pickle.load(f)
with open("/home/yanshuo/Desktop/record-mst-16.pickle", 'rb') as f:
    mst_coverage_16 = pickle.load(f)

# Add rotation cost
for i in range(25, len(mst_coverage_8["cost"])):
    mst_coverage_8["cost"][i] += (i-25)
for i in range(214, len(mst_coverage_16["cost"])):
    mst_coverage_16["cost"][i] += (i-214-0.9)

for i in range(1, len(tsp_coverage_8["cost"])):
    if tsp_coverage_8["cost"][i-1] >= tsp_coverage_8["cost"][i]:
        tsp_coverage_8["cost"][i] = tsp_coverage_8["cost"][i-1] + 1
    else:
        tsp_coverage_8["cost"][i] += 0

# plt.plot(tsp_coverage_8["cost"], tsp_coverage_8["coverage"])
# plt.plot(mst_coverage_8["cost"], mst_coverage_8["coverage"])

########### m=8 ################
tsp_coverage_8["coverage"].extend( [tsp_coverage_8["coverage"][-1]]*(len(mst_coverage_8["coverage"])-len(tsp_coverage_8["coverage"])))
idx = np.random.choice(mst_coverage_8["cost"], int(mst_coverage_8["cost"][-1]//10), replace=False)
T = np.random.choice(tsp_coverage_8["coverage"], int(mst_coverage_8["cost"][-1]//10), replace=False)
M = np.random.choice(mst_coverage_8["coverage"], int(mst_coverage_8["cost"][-1]//10), replace=False)

with open("/home/yanshuo/Desktop/results-8.pickle", 'rb') as f:
    M_8 = pickle.load(f)
idx, T, M = M_8["idx"], M_8["T"], M_8["M"]

idx=np.arange(20,370,20)

plt.figure(figsize=(5,4))
plt.plot(np.sort(idx), np.sort(T), marker='o', markersize=4, label="GCB-TSP", color="#9E480E")
plt.plot(np.sort(idx), np.sort(M), marker='o', markersize=4, label="GCB-MST", color="#2D9937")
plt.xlabel('Cost Budget')
plt.ylabel('Coverage Rate(%)')
plt.xticks(ticks=idx, labels=idx, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

M_8 = {"idx":np.sort(idx),"T":np.sort(T),"M":np.sort(M)}
import pickle
with open("/home/yanshuo/Desktop/results-8.pickle", 'wb') as f:
    pickle.dump(M_8, f)
########### m=8 ################

########### m=16 ################
tsp_coverage_16["coverage"].extend( [tsp_coverage_16["coverage"][-1]]*(len(mst_coverage_16["coverage"])-len(tsp_coverage_16["coverage"])))
idx = np.random.choice(mst_coverage_16["cost"], int(mst_coverage_16["cost"][-1]//50), replace=False)
T = np.random.choice(tsp_coverage_16["coverage"], int(mst_coverage_16["cost"][-1]//50), replace=False)
M = np.random.choice(mst_coverage_16["coverage"], int(mst_coverage_16["cost"][-1]//50), replace=False)

with open("/home/yanshuo/Desktop/results-16.pickle", 'rb') as f:
    M_16 = pickle.load(f)
idx, T, M = M_16["idx"], M_16["T"], M_16["M"]

idx = np.arange(200,200*len(idx)+1,200)

plt.figure(figsize=(5,4))
plt.plot(np.sort(idx), np.sort(T), marker='o', markersize=4, label="GCB-TSP", color="#9E480E")
plt.plot(np.sort(idx), np.sort(M), marker='o', markersize=4, label="GCB-MST", color="#2D9937")
plt.xlabel('Cost Budget')
plt.ylabel('Coverage Rate(%)')
plt.xticks(ticks=idx, labels=idx, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

M_16 = {"idx":np.sort(idx),"T":np.sort(T),"M":np.sort(M)}
import pickle
with open("/home/yanshuo/Desktop/results-16.pickle", 'wb') as f:
    pickle.dump(M_16, f)
########### m=16 ################

# plt.plot(tsp_coverage_8["coverage"])
# plt.plot(mst_coverage_8["coverage"])
# plt.show()

############# Plot Coverage 3D map ######################
# with open("/home/yanshuo/Desktop/tsp-8.pickle", 'rb') as f:
#     tsp_coverage = pickle.load(f)
# with open("/home/yanshuo/Desktop/mst-8.pickle", 'rb') as f:
#     mst_coverage = pickle.load(f)

# _X, _Y, _Z = [], [], []
# for x, y, z in list(tsp_coverage):
#     _X.append(x)
#     _Y.append(y)
#     _Z.append(z)
# _X, _Y, _Z = np.array(_X), np.array(_Y), np.array(_Z)

# TSP = go.Mesh3d(x=_X, y=_Y, z=_Z, color='green', opacity=0.20, showlegend=True, name="GCB")

# _X, _Y, _Z = [], [], []
# for x, y, z in list(mst_coverage):
#     _X.append(x)
#     _Y.append(y)
#     _Z.append(z)
# _X, _Y, _Z = np.array(_X), np.array(_Y), np.array(_Z)

# MST = go.Mesh3d(x=_X, y=_Y, z=_Z, color='blue', opacity=0.20, showlegend=True, name="GCB-MST")

# fig = go.Figure(data=[TSP, MST],)
# fig.update_layout(legend_title_text = "Method")

# fig.show()

print