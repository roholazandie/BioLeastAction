import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from numba import jit, prange
import matplotlib.colors as mcolors
import scipy.sparse as sp
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from plotly_visualize import visualize_graph

def balanced_forman_curvature(A, C=None):
    """
    Compute the balanced Forman curvature of a graph represented by adjacency matrix A.

    Parameters:
        A (numpy.ndarray): Adjacency matrix of the graph.
        C (numpy.ndarray, optional): Output matrix to store the curvature values.
                                      If None, a new matrix is created.

    Returns:
        numpy.ndarray: Matrix of balanced Forman curvatures.
    """
    N = A.shape[0]
    d_in = A.sum(axis=0)
    d_out = A.sum(axis=1)
    if C is None:
        C = np.zeros((N, N))

    A2 = np.matmul(A, A)
    for i in range(N):
        for j in range(N):
            if A[i, j] == 0:
                C[i, j] = 0
                continue

            d_max = max(d_in[i], d_out[j])
            d_min = min(d_in[i], d_out[j])

            if d_max * d_min == 0:
                C[i, j] = 0
                continue

            sharp_ij = 0
            lambda_ij = 0

            # Compute sharp_ij and lambda_ij
            for k in range(N):
                if A[i, k] == 1 and k != j:
                    for l in range(N):
                        if A[j, l] == 1 and l != i:
                            if A[k, l] == 1:
                                sharp_ij += 1
                                if A[i, l] == 0 and A[j, k] == 0:
                                    lambda_ij += 1

            C[i, j] = (
                (2 / d_max)
                + (2 / d_min)
                - 2
                + (2 / d_max + 1 / d_min) * A2[i, j] * A[i, j]
            )
            if lambda_ij > 0:
                C[i, j] += sharp_ij / (d_max * lambda_ij)

    return C



def balanced_forman_curvature_fast(A):
    N = A.shape[0]
    d_in = A.sum(axis=0)
    d_out = A.sum(axis=1)
    C = np.zeros((N, N))

    # Precompute neighbor lists
    neighbor_list = [set(np.nonzero(A[i])[0]) for i in range(N)]

    A2 = A @ A  # Compute A squared

    for i in range(N):
        for j in range(N):
            if A[i, j] == 0:
                continue  # Skip non-adjacent nodes

            d_max = max(d_in[i], d_out[j])
            d_min = min(d_in[i], d_out[j])

            if d_max * d_min == 0:
                continue  # Avoid division by zero

            N_i = neighbor_list[i] - {j}
            N_j = neighbor_list[j] - {i}

            if not N_i or not N_j:
                sharp_ij = 0
                lambda_ij = 0
            else:
                N_i_list = list(N_i)
                N_j_list = list(N_j)

                # Submatrix between N_i and N_j
                sub_A = A[np.ix_(N_i_list, N_j_list)]

                # Number of edges between N_i and N_j
                sharp_ij = np.count_nonzero(sub_A)

                # Indices where sub_A == 1
                k_indices, l_indices = np.nonzero(sub_A)

                # Map back to node indices
                ks = np.array(N_i_list)[k_indices]
                ls = np.array(N_j_list)[l_indices]

                # Check A[i, l] == 0 and A[j, k] == 0
                A_i_l = A[i, ls]
                A_j_k = A[j, ks]

                mask = (A_i_l == 0) & (A_j_k == 0)
                lambda_ij = np.count_nonzero(mask)

            # Compute curvature
            C[i, j] = (
                (2 / d_max)
                + (2 / d_min)
                - 2
                + (2 / d_max + 1 / d_min) * A2[i, j] * A[i, j]
            )

            if lambda_ij > 0:
                C[i, j] += sharp_ij / (d_max * lambda_ij)
            # else:
            #     The A2 term is already included above

    return C


def balanced_forman_curvature_sparse(G, C=None):
    """
    Compute the balanced Forman curvature of a sparse graph.

    Parameters:
        G (networkx.Graph): Input graph.
        C (scipy.sparse matrix, optional): Output matrix to store the curvature values.
                                           If None, a new sparse matrix is created.

    Returns:
        scipy.sparse.csr_matrix: Sparse matrix of balanced Forman curvatures.
    """
    N = max(G.nodes()) + 1  # N is the largest node label plus one
    nodelist = list(range(N))  # Create a nodelist from 0 to N-1

    # Create a copy of G and add missing nodes
    missing_nodes = set(nodelist) - set(G.nodes())
    G_complete = G.copy()
    if missing_nodes:
        G_complete.add_nodes_from(missing_nodes)

    # Get the adjacency matrix as a sparse CSR matrix with the specified nodelist
    A = nx.adjacency_matrix(G_complete, nodelist=nodelist).tocsr()

    # Compute A squared (paths of length 2)
    A2 = A @ A  # A2 is also a sparse matrix

    # Compute in-degrees and out-degrees
    d_in = np.array(A.sum(axis=0)).flatten()
    d_out = np.array(A.sum(axis=1)).flatten()

    if C is None:
        # Initialize C as a sparse LIL matrix for efficient assignment
        C = sp.lil_matrix((N, N))

    # Precompute neighbor lists for each node
    neighbor_list = [set(A.indices[A.indptr[i]:A.indptr[i + 1]]) for i in range(N)]

    # Initialize a set to keep track of processed edges (i, j)
    processed_edges = set()

    # Iterate over the non-zero elements of A (i.e., the edges)
    total_edges = A.nnz
    A_coo = A.tocoo()
    for idx in tqdm(range(total_edges), desc="Computing curvature"):
        i = A_coo.row[idx]
        j = A_coo.col[idx]

        # Since the graph is undirected, process each edge only once
        if (i, j) in processed_edges or (j, i) in processed_edges:
            continue
        processed_edges.add((i, j))

        # Degrees of nodes
        d_in_i = d_in[i]
        d_out_j = d_out[j]

        # Compute d_max and d_min
        d_max = max(d_in_i, d_out_j)
        d_min = min(d_in_i, d_out_j)

        if d_max * d_min == 0:
            continue  # Avoid division by zero

        # Compute sharp_ij and lambda_ij
        N_i = neighbor_list[i] - {j}
        N_j = neighbor_list[j] - {i}

        sharp_ij = 0
        lambda_ij = 0

        for k in N_i:
            neighbors_k = neighbor_list[k]
            l_set = neighbors_k & N_j  # Common neighbors between k and N_j

            for l in l_set:
                sharp_ij += 1
                # Check if internal diagonals are absent
                if l not in neighbor_list[i] and k not in neighbor_list[j]:
                    lambda_ij += 1

        # Value of A2[i, j] and A[i, j]
        A2_ij = A2[i, j]
        A_ij = A[i, j]

        # Compute curvature C[i, j]
        C_ij = (
            (2 / d_max)
            + (2 / d_min)
            - 2
            + (2 / d_max + 1 / d_min) * A2_ij * A_ij
        )

        if lambda_ij > 0:
            C_ij += sharp_ij / (d_max * lambda_ij)

        # Assign curvature value to C (since the graph is undirected)
        C[i, j] = C_ij
        C[j, i] = C_ij

    # Convert C to CSR format after computation for efficient arithmetic operations
    return C.tocsr()



if __name__ == "__main__":

    # # 1. Complete graph of size N=10
    # G_complete = nx.complete_graph(10)
    # A_complete = nx.adjacency_matrix(G_complete).todense()
    # A_complete = np.array(A_complete)
    # C_complete = balanced_forman_curvature(A_complete)
    # plot_graph_with_curvature(G_complete, C_complete, 'Complete Graph with Balanced Forman Ricci Curvature')

    # # 2. Grid graph
    # G = nx.grid_2d_graph(10, 10)
    # G = nx.convert_node_labels_to_integers(G)
    # C = balanced_forman_curvature(G)
    # edge_curvatures = {}
    # for u, v in G.edges():
    #     # For undirected graphs, ensure consistent ordering
    #     curvature = C[u, v] if (u, v) in G.edges else C[v, u]
    #     edge_curvatures[(u, v)] = curvature
    #
    # visualize_graph(G, edge_curvatures, node_sizes=[15]*len(G), layout="graphviz", title="G", filename="graph.html")


    # # 3. Tree-like graph with a central node
    # G = nx.Graph()
    # G.add_edges_from([
    #     (0, 1), (0, 2), (0, 3),
    #     (1, 4), (1, 5),
    #     (2, 6),
    #     (3, 7), (3, 8), (3, 9),
    #     (9, 7), (9, 8), (9, 5),
    #     (3, 1), (6, 10), (6, 11), (6, 12), (6, 13), (10, 11)
    # ])
    #
    # C = balanced_forman_curvature(G)
    #
    # edge_curvatures = {}
    # for u, v in G.edges():
    #     # For undirected graphs, ensure consistent ordering
    #     curvature = C[u, v] if (u, v) in G.edges else C[v, u]
    #     edge_curvatures[(u, v)] = curvature
    #
    # visualize_graph(G, edge_curvatures, node_sizes=[15]*len(G), layout="graphviz", title='Tree-like Graph with Balanced Forman Ricci Curvature')


    # # Create two complete graphs
    # n1 = 10  # Number of nodes in the first complete graph
    # n2 = 10  # Number of nodes in the second complete graph
    # G1 = nx.complete_graph(n1)
    # G2 = nx.complete_graph(n2)
    #
    # # Relabel nodes in G2 to avoid overlap with G1
    # mapping = {i: i + n1 for i in G2.nodes()}
    # G2 = nx.relabel_nodes(G2, mapping)
    #
    # # Combine the two graphs
    # G = nx.compose(G1, G2)
    #
    # # Add more random edges between the two graphs
    # num_additional_edges = 1  # Number of additional edges to add
    # for _ in range(num_additional_edges):
    #     node_from_G1 = random.choice(list(G1.nodes()))
    #     node_from_G2 = random.choice(list(G2.nodes()))
    #     G.add_edge(node_from_G1, node_from_G2)
    #
    # C = balanced_forman_curvature(G)
    #
    # # Assume G is your graph and C is the curvature matrix
    #
    # # Create a dictionary of edge curvatures
    # edge_curvatures = {}
    # for u, v in G.edges():
    #     # For undirected graphs, ensure consistent ordering
    #     curvature = C[u, v] if (u, v) in G.edges else C[v, u]
    #     edge_curvatures[(u, v)] = curvature
    #
    # # Visualize the graph
    # visualize_graph(G, edge_curvatures, node_sizes=[15]*len(G), layout="graphviz", title="Graph with Edge Curvatures", filename="graph.html")


    G = nx.Graph()

    # a simple graph
    # G.add_edges_from([
    #     (0, 1), (0, 3), (0, 2), (0, 6), (0, 4),
    #     (2, 5), (3, 5), (5, 1),
    #     (4, 6), (6, 1)
    # ])

    # a cycle
    # G.add_edges_from([
    #     (0, 1), (1, 2), (2, 3), (3, 0)
    # ])


    # G.add_edges_from([
    #     (0, 2), (0, 1), (0, 5),
    #     (1, 2), (1, 3), (1, 4), (1, 5),
    #     (2, 6), (3, 6), (4, 6), (5, 6),
    #     (2, 7), (5, 7), (6, 7),
    # ]
    # )


    # a bottleneck
    G.add_edges_from([
            (0, 3), (1, 3), (2, 3),
            (3, 4),
            (4, 5), (4, 6), (4, 7)
        ])

    # make bottleneck less bottlenecky
    # G.add_edges_from([
    #     (0, 5), (1, 5), (2, 7),
    # ])

    # a diamond (multiple paths)
    # G.add_edges_from([
    #     (0, 1), (0, 2), (0, 3), (0, 4),
    #     (1, 5), (2, 5), (3, 5), (4, 5),
    #     ])




    C = balanced_forman_curvature_sparse(G)


    edge_curvatures = {}
    for u, v in G.edges():
        # For undirected graphs, ensure consistent ordering
        curvature = C[u, v] if (u, v) in G.edges else C[v, u]
        edge_curvatures[(u, v)] = curvature
        print(f"Curvature between nodes {u} and {v}: {curvature}")

    # Visualize the graph
    visualize_graph(G, edge_curvatures, node_size=30, layout="graphviz", title="Graph with Edge Curvatures", filename="graph.html")


