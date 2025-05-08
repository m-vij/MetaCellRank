# Import Required Libraries
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

import cellrank as cr
import scvelo as scv
import SEACells

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import nnls
from sklearn.linear_model import Lasso
from sklearn.multioutput import MultiOutputRegressor
from scipy.sparse import hstack

%matplotlib inline

sns.set_style('ticks')
matplotlib.rcParams['figure.figsize'] = [4, 4]
matplotlib.rcParams['figure.dpi'] = 100

# Define Methods to Run Pipeline
def preprocess_for_SEACells(adata, normalize=True, log_transform=True, n_top_genes=2000, use_highly_variable=False, n_pcs=30, n_neighbors=30, random_state=0, ):
    """
    Preprocesses an AnnData object for use with SEACells by performing standard
    single-cell RNA-seq data processing steps.

    Parameters:
    -----------
    adata : AnnData
        The annotated data matrix (cells x genes) to be processed.
    normalize : bool, default=True
        Whether to normalize counts per cell to a target sum (1e4).
    log_transform : bool, default=True
        Whether to apply log1p transformation to normalized data.
    n_top_genes : int, default=2000
        Number of top highly variable genes to select (used only if use_highly_variable=True).
    use_highly_variable : bool, default=False
        Whether to subset the data to the top highly variable genes.
    n_pcs : int, default=30
        Number of principal components to compute and use for neighborhood graph construction.
    n_neighbors : int, default=30
        Number of nearest neighbors for computing the neighborhood graph.
    random_state : int, default=0
        Seed for reproducibility in PCA and UMAP.

    Returns:
    --------
    adata : AnnData
        The processed AnnData object with PCA, neighbors, and UMAP embeddings computed.
    """
    if normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)

    if log_transform:
        sc.pp.log1p(adata)

    if use_highly_variable:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        adata = adata[:, adata.var.highly_variable]

    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors, random_state=random_state)
    sc.tl.umap(adata, random_state=random_state)

    return adata

def run_SEACells(adata, n_SEACells, build_kernel_on, n_waypoint_eigs, convergence_epsilon, min_iter, max_iter):
    """
    Runs the SEACells algorithm on a preprocessed AnnData object to identify metacells
    (SEACells) that represent underlying cellular states.

    Parameters:
    -----------
    adata : AnnData
        Preprocessed AnnData object containing single-cell data. Should be PCA-transformed
        and have a neighborhood graph computed.
    n_SEACells : int
        The number of SEACells (metacells) to identify.
    build_kernel_on : str
        The attribute of `adata.obsm` on which to construct the kernel (e.g., 'X_pca' or 'X_umap').
    n_waypoint_eigs : int
        Number of eigenvectors to use when constructing waypoints.
    convergence_epsilon : float
        Threshold for convergence; lower values require tighter convergence of the model.
    min_iter : int
        Minimum number of iterations to run the SEACells optimization.
    max_iter : int
        Maximum number of iterations to run the SEACells optimization.

    Returns:
    --------
    model : SEACells.core.SEACells
        A fitted SEACells model object containing the archetype assignments and kernel matrix.
    """
    model = SEACells.core.SEACells(adata,
                      build_kernel_on=build_kernel_on,
                      n_SEACells=n_SEACells,
                      n_waypoint_eigs=n_waypoint_eigs,
                      convergence_epsilon = convergence_epsilon)
    model.construct_kernel_matrix()
    M = model.kernel_matrix
    model.initialize_archetypes()
    SEACells.plot.plot_initialization(adata, model) # Plot the initilization to ensure they are spread across phenotypic space
    model.fit(min_iter=min_iter, max_iter=max_iter)
    return model

def run_cellrank(SEACell_combined, n_pcs, n_neighbors, filter=False):
    """
    Runs RNA velocity and computes CellRank transition matrices using a combination
    of VelocityKernel and ConnectivityKernel on SEACell-level data.

    This function optionally filters and normalizes the data, recovers gene dynamics,
    computes velocities, and then constructs a transition matrix that integrates
    both velocity-derived and graph-based cell transitions.

    Parameters:
    -----------
    SEACell_combined : AnnData
        An AnnData object containing combined (spliced + unspliced) SEACell data,
        with spliced and unspliced counts in `.layers`.
    n_pcs : int
        Number of principal components (PCs) to use in the preprocessing pipeline.
        (Note: This argument is not currently used inside the function but may be
        relevant for prior steps.)
    n_neighbors : int
        Number of neighbors to use when building the connectivity graph.
        (Note: This argument is not used in this function but may be relevant if
        neighborhood graphs are recomputed externally.)
    filter : bool, optional (default: False)
        If True, applies `scv.pp.filter_and_normalize` to the data before
        velocity computation.

    Returns:
    --------
    combined_kernel : cellrank.kernels.CompoundKernel
        A CellRank kernel that combines the VelocityKernel and ConnectivityKernel
        with weights 0.8 and 0.2 respectively.
    vk : cellrank.kernels.VelocityKernel
        The velocity kernel computed from RNA velocity.
    ck : cellrank.kernels.ConnectivityKernel
        The connectivity kernel based on neighborhood graph structure.
    """
    if filter:
        scv.pp.filter_and_normalize(
            SEACell_combined,
            min_shared_counts=20,
            n_top_genes=2000,
            subset_highly_variable=False
        )

    scv.tl.recover_dynamics(SEACell_combined, n_jobs=8)
    scv.tl.velocity(SEACell_combined, mode="dynamical")

    vk = cr.kernels.VelocityKernel(SEACell_combined)
    vk.compute_transition_matrix()

    ck = cr.kernels.ConnectivityKernel(SEACell_combined)
    ck.compute_transition_matrix()

    combined_kernel = 0.8 * vk + 0.2 * ck
    return combined_kernel, vk, ck

def visualize_trajectory(SEACell_combined, vk, color, density, number_cells=50):
    """
    Computes UMAP embedding and visualizes RNA velocity trajectories
    on SEACell data using the VelocityKernel.

    This function performs PCA and neighborhood graph construction
    based on the 'spliced' layer, followed by UMAP projection.
    It then overlays the RNA velocity vectors onto the UMAP using
    `velocityKernel.plot_projection`.

    Parameters:
    -----------
    SEACell_combined : AnnData
        AnnData object containing SEACell-level combined data with a 'spliced' layer.
    vk : cellrank.kernels.VelocityKernel
        A precomputed VelocityKernel object from CellRank.
    color : str
        Column name in `SEACell_combined.obs` to use for coloring cells in the plot.
    density : float
        Controls the density of velocity streamlines in the plot.
    """
    sc.pp.pca(SEACell_combined, layer='spliced')
    sc.pp.neighbors(SEACell_combined, use_rep='X_pca')
    sc.tl.umap(SEACell_combined)
    vk.plot_projection(color=color, density=density, size=5000)


# Combines spliced and unspliced into a single layer
def concat_spliced_unspliced_for_SEACells(adata):
    """
    Assigns cluster labels from the original single-cell AnnData object
    to the SEACell-level metacell representation based on index matching.

    This is useful for comparing metacell assignments to existing cell-type
    annotations or clusters.

    Parameters:
    -----------
    combined_contribution : AnnData
        AnnData object whose `.obs_names` are aligned with a subset of the original adata.
    adata : AnnData
        The original single-cell AnnData object containing cluster labels in `.obs`.
    cluster_key : str, optional (default: 'clusters')
        The column in `adata.obs` that holds cluster or cell-type labels.

    Returns:
    --------
    pandas.Series
        A Series of cluster labels aligned to `combined_contribution.obs_names`.
    """
    concatenated_counts = hstack([adata.layers['spliced'], adata.layers['unspliced']])

    # 2. Prepare updated gene names (var)
    var_spliced = adata.var.copy()
    var_unspliced = adata.var.copy()

    var_spliced.index = var_spliced.index + "_spliced"
    var_unspliced.index = var_unspliced.index + "_unspliced"

    var_combined = pd.concat([var_spliced, var_unspliced], axis=0)

    # 3. Create a new AnnData object
    adata_combined = AnnData(X=concatenated_counts)

    # 4. Transfer obs and new var
    adata_combined.obs = adata.obs.copy()
    adata_combined.var = var_combined
    adata_combined.raw = adata_combined.copy()

    return adata_combined


def restack_layers(concatenated_metacell_adata, adata):
    """
    Reconstructs an AnnData object with separated 'spliced' and 'unspliced' layers
    from a concatenated input where spliced and unspliced data were stacked side-by-side.

    Parameters:
    -----------
    concatenated_metacell_adata : AnnData
        AnnData object where the count matrix X has spliced and unspliced counts concatenated horizontally.
        Shape: (n_cells, n_genes * 2)
        
    adata : AnnData
        A reference AnnData object (e.g. the original before concatenation) to restore variable metadata.

    Returns:
    --------
    new_adata : AnnData
        A new AnnData object with `X` as the combined (spliced + unspliced) counts and
        `.layers["spliced"]`, `.layers["unspliced"]` appropriately restored.
    """


    concatenated_counts = concatenated_metacell_adata.X

    n_spliced = int(concatenated_counts.shape[1] / 2)
    spliced = concatenated_counts[:, :n_spliced]
    unspliced = concatenated_counts[:, n_spliced:]


    combined_counts = spliced + unspliced
    new_adata = AnnData(X=combined_counts)
    new_adata.layers["spliced"] = spliced
    new_adata.layers["unspliced"] = unspliced
    new_adata.obs = concatenated_metacell_adata.obs.copy()
    new_adata.var = adata.var.copy()

    return new_adata


# Load pancreas dataset
adata = scv.datasets.pancreas()

# Define Approach 2 Parameters
# Sea Cells Paramaters
n_SEACells = 100
build_kernel_on = 'X_pca' # key in ad.obsm to use for computing metacells. This would be replaced by 'X_svd' for ATAC data.
n_waypoint_eigs = 10 # Number of eigenvalues to consider when initializing metacells
convergence_epsilon = 1e-5 # Convergence threshold
min_iter = 10 # SEACells minimum iteratons for convergence
max_iter = 500 # SEACells maximum iteratons for convergence
minimum_weight = 0.05 # SEACells weight for soft assignments
celltype_label = None # Can optionally provide celltypes for metacells

# Cell Rank parameters
n_pcs = 50 # number of principal componenets for cell rank
n_neighbors = 10 # number of nearest neighbors for KNN graph in cell rank


# Run Approach 2 Pipeline With Specified Parameters

# Run SEACells
concatenated = concat_spliced_unspliced_for_SEACells(adata)
concatenated = preprocess_for_SEACells(concatenated)
model = run_SEACells(concatenated, n_SEACells, build_kernel_on, n_waypoint_eigs, convergence_epsilon, min_iter, max_iter)

# Get SEACell gene expression and split matrix back into two layers
SEACell_soft_ad = SEACells.core.summarize_by_soft_SEACell(concatenated, model.A_, celltype_label='clusters',summarize_layer='raw', minimum_weight=0.05)
metacell_combined = restack_layers(SEACell_soft_ad, adata)

# Run Cell Rank
combined_kernel, vk, ck = run_cellrank(metacell_combined, n_pcs, n_neighbors, filter=True) # combined_kernel is a string formula
visualize_trajectory(metacell_combined, vk, 'celltype', density=3)
