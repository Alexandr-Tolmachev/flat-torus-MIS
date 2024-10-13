import numpy as np
import networkx as nx

def is_perfectly_periodical(l1, l2, alpha):
    if l1 * np.sin(alpha) >= 2 and l2 >= 2:
        return True
    if l2 * np.sin(alpha) >= 2 and l1 >= 2:
        return True
    return False


def calculate_radius(l1, l2, alpha, N, M):
    d1, d2 = l1 / N, l2 / M  # distances on the triangle grid
    c = np.sqrt(d1 * d1 + d2 * d2 - 2 * d1 * d2 * np.cos(alpha)) # compute third triangle length
    r = c / (2 * np.sin(alpha)) # compute the radius of the circumscribed circle
    return r


def compute_torus_distances(point1, point2, l1, l2, alpha):
    x1, y1 = point1
    x2, y2 = point2

    if l1 > l2:
        l1, l2 = l2, l1 # without loss of generality suppose that l1 <= l2
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    v1 = np.array([l1, 0])
    v2 = np.array([l2 * np.cos(alpha), l2 * np.sin(alpha)])

    def f(k):
        return -(x2 - x1 + ((y2 - y1 + k) * l2 * np.cos(alpha) / l1))

    K = int(np.ceil(1 / (1 - np.cos(alpha) ** 2)) + 1)
    min_dist = +np.infty

    best_n, best_m = 0, 0

    for n in range(-K, K + 1):
        for m in [np.floor(f(n)), np.ceil(f(n))]:  #np.floor(f(n)) + 1]:
            curr_dist = np.linalg.norm((x2 - x1 + m) * v1 + (y2 - y1 + n) * v2)
            if curr_dist < min_dist:
                best_n = n
                best_m = m
                min_dist = curr_dist

    return min_dist, best_n, best_m


def get_offset_list(l1, l2, alpha, N, M):

    v1 = np.array([l1, 0])
    v2 = np.array([l2 * np.cos(alpha), l2 * np.sin(alpha)])

    r = calculate_radius(l1, l2, alpha, N, M)
    assert 2 * r < 1, "Grid isn't an appropriate for the graph construction"

    offset_list = []
    DELTA = 1e-10

    for i in range(N):
        for j in range(M):
            curr_dist, _, _ = compute_torus_distances((0, 0), (i/N, j/M), l1, l2, alpha)
            if curr_dist > 1 - 2 * r - DELTA and curr_dist < 1 + 2 * r + DELTA:  # for numerical stability
                offset_list.append([i, j])

    return offset_list


def get_torus_graph(l1, l2, alpha, N, M):
    #assert is_perfectly_periodical(l1, l2, alpha), "Torus is not a perfectly periodical"
    offset_list = get_offset_list(l1, l2, alpha, N, M)

    g = nx.Graph()
    g.add_nodes_from(np.arange(N * M))

    for i in range(N):
        for j in range(M):
            vertex_from = M * i + j # + 1

            for offset in offset_list:
                x0 = (i + offset[0]) % N
                y0 = (j + offset[1]) % M

                vertex_to = M * x0 + y0  #+ 1
                g.add_edge(vertex_from, vertex_to)

    assert len(g.edges) == N * M * len(offset_list) // 2
    
    return g
