from dolfin import *
import numpy as np
from scipy.spatial import cKDTree
import os
import sys
from fenics_functions import *
from mpi4py import MPI
import dolfin as df
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
interface_id = 103

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.abspath(os.environ.get("FLOW_RUN_DIR", CODE_DIR))
MESH_DIR = os.path.join(CODE_DIR, "meshes")
VASCULAR_MESH_H5 = os.path.join(MESH_DIR, "vascular_mesh.h5")
INTERSTITIUM_MESH_H5 = os.path.join(MESH_DIR, "interstitial_mesh.h5")
STOKES_PRESSURE_XDMF = os.path.join(RUN_DIR, "blood_flow", "stokes_pressure.xdmf")
OUTPUT_DIR = os.path.join(RUN_DIR, "perivascular_coordinate")

df.parameters["mesh_partitioner"] = "ParMETIS"
df.parameters["form_compiler"]["cpp_optimize"]  = True

mesh_v = Mesh()
with HDF5File(mesh_v.mpi_comm(), VASCULAR_MESH_H5, "r") as h5:
    h5.read(mesh_v, "/mesh", False)
    tdim_v = mesh_v.topology().dim()
    facets_v = MeshFunction("size_t", mesh_v, tdim_v - 1, 0)
    h5.read(facets_v, "/boundaries")

mesh_i = Mesh()
with HDF5File(mesh_i.mpi_comm(), INTERSTITIUM_MESH_H5, "r") as h5:
    h5.read(mesh_i, "/mesh", False)
    tdim_i = mesh_i.topology().dim()
    facets_i = MeshFunction("size_t", mesh_i, tdim_i - 1, 0)
    h5.read(facets_i, "/boundaries")

def main():
    base_folder = OUTPUT_DIR

    if rank == 0:
        os.makedirs(base_folder, exist_ok=True)
        start_logging(base_folder)
    comm.Barrier()
    file = XDMFFile(comm, os.path.join(base_folder, "solution.xdmf"))
    file.parameters["flush_output"] = True
    file.parameters["functions_share_mesh"] = True
    V_v = FunctionSpace(mesh_v, "CG", 1)
    # Load the vascular pressure checkpoint.
    p_v = Function(V_v)

    stokes_pressure_file = XDMFFile(comm, STOKES_PRESSURE_XDMF)
    stokes_pressure_file.read_checkpoint(p_v, "p_v", -1)

    file.write(p_v, 0.0)
    # --- 1) Sample Stokes pressure on vascular boundary vertices (local on each rank) ---
    bm_v = BoundaryMesh(mesh_v, "exterior", False)   # local piece per rank
    X_loc = bm_v.coordinates()                       # (n_loc, 3)
    print(f"Rank {rank}: sampled {len(X_loc)} vascular boundary vertices")
    p_loc = np.empty(len(X_loc), dtype=float)
    buf = np.zeros(1, dtype=float)

    # robust point evaluation (works for any CG degree)
    for i, x in enumerate(X_loc):
        p_v.eval(buf, x)
        p_loc[i] = buf[0]

    # --- 2) Gather to rank 0, concatenate, and 3) broadcast back to all ranks ---
    X_list = comm.gather(X_loc, root=0)
    p_list = comm.gather(p_loc, root=0)

    if rank == 0:
        X_src = np.vstack(X_list) if len(X_list) else np.empty((0, mesh_v.geometry().dim()))
        p_src = np.concatenate(p_list) if len(p_list) else np.empty((0,))
    else:
        X_src = None
        p_src = None

    X_src = comm.bcast(X_src, root=0)
    p_src = comm.bcast(p_src, root=0)
    print(f"Rank {rank}: received {len(X_src)} vascular boundary vertices total")
    # --- 4) Build KD-tree on every rank ---
    tree = cKDTree(X_src)
    V = FunctionSpace(mesh_i, "CG", 1)     # Darcy pressure space


    # --- Find the Darcy boundary DOFs ONCE ---
    # Use a zero-valued boundary condition to identify boundary degrees of freedom.
    gdim = mesh_i.geometry().dim()
    # Identify local boundary degrees of freedom.
    bc_probe = DirichletBC(V, Constant(0.0), facets_i, interface_id, method='topological')

    # Local boundary dofs returned by BC (may include ghosts)
    bdofs_local_all = np.array(list(bc_probe.get_boundary_values().keys()), dtype=np.int64)

    # Keep only OWNED local indices

    # Now it’s safe to index coordinates and fill the vector
    gdim = mesh_i.geometry().dim()
    Xdof_owned = V.tabulate_dof_coordinates().reshape((-1, gdim))   # shape = (local_owned_dim, gdim)
    bdofs_local_owned = bdofs_local_all[bdofs_local_all < len(Xdof_owned)]

    Xq = Xdof_owned[bdofs_local_owned]

    # --- Batch kNN on this rank (tree is identical on all ranks) ---
    k = 3
    d, idx = tree.query(Xq, k=k, workers=-1)
    d   = np.atleast_2d(d)
    idx = np.atleast_2d(idx)

    # Vectorized IDW with exact-hit handling
    zero_hit = (d == 0.0)
    vals = np.empty(Xq.shape[0], dtype=float)
    rows_zh = np.any(zero_hit, axis=1)
    print(f"Rank {rank}: interpolating {len(vals)} Darcy BC DOFs, of which {np.sum(rows_zh)} are exact hits")
    if np.any(rows_zh):
        j0 = zero_hit[rows_zh].argmax(axis=1)
        vals[rows_zh] = p_src[idx[rows_zh, j0]]
    rows_nzh = ~rows_zh
    print(f"Rank {rank}: doing IDW for {np.sum(rows_nzh)} Darcy BC DOFs")
    if np.any(rows_nzh):
        w = 1.0 / d[rows_nzh]
        w /= w.sum(axis=1, keepdims=True)
        vals[rows_nzh] = np.einsum('ij,ij->i', w, p_src[idx[rows_nzh]])

    # --- Fill ONLY local boundary DOFs in g (local indices!) ---
    g = Function(V)
    vec = g.vector().get_local()
    vec[bdofs_local_owned] = vals
    g.vector().set_local(vec)
    g.vector().apply('insert')
    darcy_file = df.XDMFFile(comm, os.path.join(base_folder, "darcy_pressure.xdmf"))
    darcy_file.parameters["flush_output"] = True
    darcy_file.parameters["functions_share_mesh"] = True





    p_darcy = Function(V)
    pd = TrialFunction(V);
    qd = TestFunction(V)
    # Darcy equations
    # Parameters
    mu = 1.2E-3 # viscosity in kg/(m*s)

    dj = 20 # nm
    tau = 1.0
    k = 5.2E-14/ (tau**2) # permeability in m²
    Lp = 2.7E-9 * (dj/70)**3 # Transvascular permeability from Starlings law in m/(s Pa)
    ds = Measure('ds', domain=mesh_i, subdomain_data=facets_i)
    a = inner(k/mu * grad(pd), grad(qd)) * dx
    L = df.Constant(0)* qd * dx
    A = assemble(a)
    b = assemble(L)

    # Apply dirichlet BC - set pd equals g on interface
    bc = DirichletBC(V, g, facets_i, interface_id)
    bc.apply(A, b)

    solve(A, p_darcy.vector(), b)


    p_darcy.rename("p_darcy", "Darcy pressure")
    darcy_file.write(p_darcy, 0.0)

    # save dracy pressure as a readable XDMF file
    output_file = XDMFFile(comm, os.path.join(base_folder, "darcy_pressure_output.xdmf"))
    output_file.write_checkpoint(p_darcy, "p_darcy", 0.0, XDMFFile.Encoding.HDF5, True)


if __name__ == '__main__':
    # Log time
    start_time = MPI.Wtime()
    main()
    end_time = MPI.Wtime()
    if rank == 0:
        print(f"Execution time: {end_time - start_time:.2f} seconds")
