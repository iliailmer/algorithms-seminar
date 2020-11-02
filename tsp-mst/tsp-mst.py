import networkx as nx
from matplotlib import pyplot as plt
import numpy as np

from concorde.tsp import TSPSolver

size = 1024
xs = np.array([x for x in np.random.randn(size)])
ys = np.array([x for x in np.random.randn(size)])

g = nx.Graph()
for i in range(len(xs)):
    g.add_node(i, pos=(xs[i], ys[i]))
for i in range(len(xs)):
    for j in range(len(ys)):
        if i != j:
            g.add_edge(
                i,
                j,
                weight=np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2),
            )

mst = nx.algorithms.minimum_spanning_tree(g)
edge_list_mst = np.array([np.array([x, y]) for (x, y) in mst.edges])

solver = TSPSolver.from_data(xs, ys, norm="GEOM")
tour = solver.solve()

edge_list_tsp = np.array(
    [
        np.array([x, y])
        for (x, y) in list(zip(tour.tour, tour.tour[1:]))
        + [(tour.tour[-1], tour.tour[0])]
    ]
)

common = []
mst_list = [(x, y) for (x, y) in mst.edges]
for tsp_edge in edge_list_tsp:
    if tuple(tsp_edge) in mst_list or tuple(tsp_edge[::-1]) in mst_list:
        common.append(tsp_edge)
common = np.array(common)
print(len(common) / len(edge_list_tsp))

fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(16, 4))

for i in range(3):
    ax[i].scatter(xs, ys, s=40, color="r")


ax[0].plot(
    [xs[edge_list_mst[:, 0]], xs[edge_list_mst[:, 1]]],
    [ys[edge_list_mst[:, 0]], ys[edge_list_mst[:, 1]]],
    color="k",
)
ax[0].set_title("MST")

ax[1].plot(
    [xs[edge_list_tsp[:, 0]], xs[edge_list_tsp[:, 1]]],
    [ys[edge_list_tsp[:, 0]], ys[edge_list_tsp[:, 1]]],
    color="b",
)
ax[1].set_title("TSP")

ax[2].plot(
    [xs[common[:, 0]], xs[common[:, 1]]],
    [ys[common[:, 0]], ys[common[:, 1]]],
    color="g",
    linewidth=4,
)
ax[2].set_title(f"Intersection, ratio: {len(common)/len(edge_list_tsp)}")

plt.tight_layout()
plt.show()
plt.savefig("figure")