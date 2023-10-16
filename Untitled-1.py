
list(nx.connected_components(S))

import networkx as nx
list(nx.connected_components(selected_graph))

[(u,v,d) for (u, v, d) in selected_graph.edges(data=True)]

[(u,v,d) for (u, v, d) in T.edges(data=True)]


from itertools import combinations
list(combinations(subgoal_pos, 2))


T = nx.minimum_spanning_tree(G, algorithm="prim")
nx.draw_networkx(T)

# Set margins for the axes so that nodes aren't clipped
ax = plt.gca()
ax.margins(0.20)
plt.axis("off")
plt.show()

------

selected_graph
selected_vertices

list(nx.connected_components(sp))

nx.draw_networkx(msts)

nx.shortest_path_length(sp, 8).values()


import networkx as nx
T = nx.minimum_spanning_tree(resuls["selected_graph"], algorithm="prim")
nx.draw_networkx(T)



from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


# plot subgoals
from itertools import combinations
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
subgoal_set = np.array(vertexes)
ax.scatter(subgoal_set[:,0],subgoal_set[:,1],subgoal_set[:,2], marker='o', s=50)
arrow_len = 4
for pos in subgoal_set:
    for ang in [0, 60, 120, 180, 240, 300]:
        print(arrow_len*np.cos(ang * np.pi / 180.),
              arrow_len*np.sin(ang * np.pi / 180.),
              np.sqrt((arrow_len*np.cos(ang * np.pi / 180.))**2 + (arrow_len*np.sin(ang * np.pi / 180.))**2))
        a = Arrow3D([pos[0], pos[0]+arrow_len*np.cos(ang * np.pi / 180.)], 
                    [pos[1], pos[1]+arrow_len*np.sin(ang * np.pi / 180.)], 
                    [pos[2], pos[2]], mutation_scale=20, 
                    lw=2, arrowstyle="-|>", color="tab:orange")
        ax.add_artist(a)

# ax.scatter(root[0],root[1],root[2], marker='*', s=300)
ax.set(xlabel='x', ylabel='y', zlabel='z')
plt.show()



from matplotlib import pyplot as plt
fig = plt.figure(figsize=(100,100))
ax = fig.add_subplot(projection='3d')
subgoal_set = np.array(vertexes)
for start, end, length in self.mst.primMST():
    ax.plot([subgoal_set[int(start)][0], subgoal_set[int(end)][0]],
            [subgoal_set[int(start)][1], subgoal_set[int(end)][1]],
            [subgoal_set[int(start)][2], subgoal_set[int(end)][2]],color = 'g')
ax.scatter(subgoal_set[:,0],subgoal_set[:,1],subgoal_set[:,2], marker='o', s=70)
arrow_len = 6
# for pos in subgoal_set:
#     for ang in [0, 60, 120, 180, 240, 300]:
#         print(arrow_len*np.cos(ang * np.pi / 180.),
#               arrow_len*np.sin(ang * np.pi / 180.),
#               np.sqrt((arrow_len*np.cos(ang * np.pi / 180.))**2 + (arrow_len*np.sin(ang * np.pi / 180.))**2))
#         a = Arrow3D([pos[0], pos[0]+arrow_len*np.cos(ang * np.pi / 180.)], 
#                     [pos[1], pos[1]+arrow_len*np.sin(ang * np.pi / 180.)], 
#                     [pos[2], pos[2]], mutation_scale=10, 
#                     lw=2, arrowstyle="-|>", color="tab:orange")
#         ax.add_artist(a)
for x, y, z, q1, q2, q3, q4 in np.unique(self.paths, axis=0):
    ang = R.from_quat([q1, q2, q3, q4]).as_euler('xyz', degrees=True)[2]
    a = Arrow3D([x, x+arrow_len*np.cos(ang * np.pi / 180.)], 
                [y, y+arrow_len*np.sin(ang * np.pi / 180.)], 
                [z, z], mutation_scale=30, 
                lw=4, arrowstyle="-|>", color="lightcoral") #lightcoral
    ax.add_artist(a)

# ax.scatter(root[0],root[1],root[2], marker='*', s=300)
ax.set(xlabel='x', ylabel='y', zlabel='z')
plt.show()


from matplotlib import pyplot as plt
plt.figure()
plt.subplot(611)
plt.plot(record["coverage"])
plt.subplot(612)
plt.plot(record["B"])
plt.subplot(613)
plt.plot(record["marginal"])
plt.subplot(614)
plt.plot(record["B*lmd"])
plt.subplot(615)
plt.plot(record["lmd"])
plt.subplot(616)
plt.plot(record["total_coverage"])


from matplotlib import pyplot as plt
plt.figure()
plt.subplot(411)
plt.plot(B)
plt.subplot(412)
plt.plot(b1)
plt.subplot(413)
plt.plot(b2)
plt.subplot(414)
plt.plot(b3+b2)


import networkx as nx
msts = nx.minimum_spanning_tree(record["graph"][83], algorithm="prim")
nx.draw_networkx(msts)


G = msts
pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility
# nodes
nx.draw_networkx_nodes(G, pos, node_size=100)
# edges
nx.draw_networkx_edges(G, pos, width=3)
# node labels
nx.draw_networkx_labels(G, pos, font_size=5, font_family="sans-serif")
# edge weight labels
edge_labels = dict([((u,v,), f"{d['weight']:.2f}") for u,v,d in G.edges(data=True)])
# edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels)
ax = plt.gca()
ax.margins(0.08)
plt.axis("off")
plt.tight_layout()
plt.show()




from scipy.spatial.transform import Rotation as R
fig = plt.figure(figsize=(100,100))
ax = fig.add_subplot(projection='3d')
subgoal_set = np.array(vertexes)
ax.scatter(subgoal_set[:,0],subgoal_set[:,1],subgoal_set[:,2], marker='o', s=70)
arrow_len = 6
for X in results["traj_coord"]:
    for x, y, z, q1, q2, q3, q4 in X:
        ang = R.from_quat([q1, q2, q3, q4]).as_euler('xyz', degrees=True)[2]
        a = Arrow3D([x, x+arrow_len*np.cos(ang * np.pi / 180.)], 
                    [y, y+arrow_len*np.sin(ang * np.pi / 180.)], 
                    [z, z], mutation_scale=30, 
                    lw=4, arrowstyle="-|>", color="lightcoral") #lightcoral
        ax.add_artist(a)

# ax.scatter(root[0],root[1],root[2], marker='*', s=300)
ax.set(xlabel='x', ylabel='y', zlabel='z')
plt.show()