import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import issparse, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits
from typing import Tuple, List
from scipy.stats import entropy
import logging
import scvelo as scv
from scanpy.neighbors import _get_indices_distances_from_sparse_matrix

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


def get_symmetric_matrix(csr_mat: "csr_matrix") -> "csr_matrix":
    tp_mat = csr_mat.transpose().tocsr()
    sym_mat = csr_mat + tp_mat
    sym_mat.sort_indices()

    idx_mat = (csr_mat != 0).astype(int) + (tp_mat != 0).astype(int)
    idx_mat.sort_indices()
    idx = idx_mat.data == 2

    sym_mat.data[idx] /= 2.0
    return sym_mat


def calculate_affinity_matrix(
        indices: List[int],
        distances: List[float]
) -> "csr_matrix":
    nsample = indices.shape[0]
    K = indices.shape[1]
    # calculate sigma, important to use median here!
    sigmas = np.median(distances, axis=1)
    sigmas_sq = np.square(sigmas)

    # calculate local-scaled kernel
    normed_dist = np.zeros((nsample, K), dtype=np.float32)
    for i in range(nsample):
        numers = 2.0 * sigmas[i] * sigmas[indices[i, :]]
        denoms = sigmas_sq[i] + sigmas_sq[indices[i, :]]
        normed_dist[i, :] = np.sqrt(numers / denoms) * np.exp(
            -np.square(distances[i, :]) / denoms
        )

    W = csr_matrix(
        (normed_dist.ravel(), (np.repeat(range(nsample), K), indices.ravel())),
        shape=(nsample, nsample),
    )
    W = get_symmetric_matrix(W)

    # density normalization
    z = W.sum(axis=1).A1
    W = W.tocoo()
    W.data /= z[W.row]
    W.data /= z[W.col]
    W = W.tocsr()
    W.eliminate_zeros()

    return W


def calculate_normalized_affinity(
        W: csr_matrix
) -> Tuple[csr_matrix, np.array, np.array]:
    diag = W.sum(axis=1).A1
    diag_half = np.sqrt(diag)
    W_norm = W.tocoo(copy=True)
    W_norm.data /= diag_half[W_norm.row]
    W_norm.data /= diag_half[W_norm.col]
    W_norm = W_norm.tocsr()

    return W_norm, diag, diag_half


def calc_von_neumann_entropy(lambdas: List[float], t: float) -> float:
    etas = 1.0 - lambdas ** t
    etas = etas / etas.sum()
    return entropy(etas)


def eff_n_jobs(n_jobs: int) -> int:
    """ If n_jobs < 0, set it as the number of physical cores _cpu_count """
    if n_jobs > 0:
        return n_jobs

    import psutil
    _cpu_count = psutil.cpu_count(logical=False)
    if _cpu_count is None:
        _cpu_count = psutil.cpu_count(logical=True)
    return _cpu_count


def find_knee_point(x: List[float], y: List[float]) -> int:
    """ Return the knee point, which is defined as the point furthest from line between two end points
    """
    p1 = np.array((x[0], y[0]))
    p2 = np.array((x[-1], y[-1]))
    length_p12 = np.linalg.norm(p2 - p1)

    max_dis = 0.0
    knee = 0
    for cand_knee in range(1, len(x) - 1):
        p3 = np.array((x[cand_knee], y[cand_knee]))
        dis = np.linalg.norm(np.cross(p2 - p1, p3 - p1)) / length_p12
        if max_dis < dis:
            max_dis = dis
            knee = cand_knee

    return knee


def calculate_diffusion_map(
        W: csr_matrix,
        n_components: int,
        solver: str,
        max_t: int,
        n_jobs: int,
        random_state: int,
) -> Tuple[np.array, np.array, np.array]:
    assert issparse(W)

    # nc, labels = connected_components(W, directed=True, connection="strong")
    # logger.info("Calculating connected components is done.")
    # assert nc == 1

    W_norm, diag, diag_half = calculate_normalized_affinity(
        W.astype(np.float64))  # use double precision to guarantee reproducibility
    logger.info("Calculating normalized affinity matrix is done.")

    n_jobs = eff_n_jobs(n_jobs)
    with threadpool_limits(limits=n_jobs):
        if solver == "eigsh":
            np.random.seed(random_state)
            v0 = np.random.uniform(-1.0, 1.0, W_norm.shape[0])
            lambda_, U = eigsh(W_norm, k=n_components, v0=v0)
            lambda_ = lambda_[::-1]
            U = U[:, ::-1]
        else:
            assert solver == "randomized"
            U, S, VT = randomized_svd(
                W_norm, n_components=n_components, random_state=random_state
            )
            signs = np.sign((U * VT.transpose()).sum(axis=0))  # get eigenvalue signs
            lambda_ = signs * S  # get eigenvalues

    # remove the first eigen value and vector
    lambda_ = lambda_[1:]
    U = U[:, 1:]
    phi = U / diag_half[:, np.newaxis]

    if max_t == -1:
        lambda_new = lambda_ / (1.0 - lambda_)
    else:
        # Find the knee point
        x = np.array(range(1, max_t + 1), dtype=float)
        y = np.array([calc_von_neumann_entropy(lambda_, t) for t in x])
        t = x[find_knee_point(x, y)]
        logger.info("Detected knee point at t = {:.0f}.".format(t))

        lambda_new = lambda_ * ((1.0 - lambda_ ** t) / (1.0 - lambda_))
    phi_point = phi * lambda_new  # asym pseudo component

    return phi_point, lambda_, phi  # , U_df, W_norm