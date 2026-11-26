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
PERIVASCULAR_COORDINATE_XDMF = os.path.join(
    RUN_DIR, "perivascular_coordinate", "darcy_pressure_output.xdmf"
)
DARCY_OUTPUT_DIR = os.path.join(RUN_DIR, "interstitial_flow")
X0_CUT = 0.004771099337748345

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
    base_folder = DARCY_OUTPUT_DIR

    if rank == 0:
        os.makedirs(base_folder, exist_ok=True)
        start_logging(base_folder)
        save_current_script(base_folder)
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

    g.rename("g_interpolated", "Interpolated Darcy BC from Stokes")
    darcy_file.write(g, 0.0)



    p_darcy = Function(V)
    pd = TrialFunction(V);
    qd = TestFunction(V)
    # Darcy equations
    # Parameters

    mu = 1.2E-3  # kg/(m*s)

    dj = 70      # nm
    tau = 1.0
    k0  = 5.2E-14 / (tau**2)                 # baseline permeability (m^2)
    Lp0 = 2.7E-9 * (dj/70.0)**3              # baseline hydraulic conductivity (m/(s Pa))

    x0_cut = X0_CUT

    # --- Spatially varying "switch-off" expressions ---
    class KExpr(UserExpression):
        def __init__(self, k_base, x_cut, **kwargs):
            super().__init__(**kwargs)
            self.k_base = float(k_base)
            self.x_cut  = float(x_cut)

        def eval(self, values, x):
            values[0] = self.k_base/10000 if x[0] > self.x_cut else self.k_base

        def value_shape(self):
            return ()

    class LpExpr(UserExpression):
        def __init__(self, Lp_base, x_cut, **kwargs):
            super().__init__(**kwargs)
            self.Lp_base = float(Lp_base)
            self.x_cut   = float(x_cut)

        def eval(self, values, x):
            values[0] = self.Lp_base/10000 if x[0] > self.x_cut else self.Lp_base

        def value_shape(self):
            return ()

    # Degree matters for quadrature; 0 is piecewise-constant.
    k  = KExpr(k0,  x0_cut, degree=0)
    Lp = LpExpr(Lp0, x0_cut, degree=0)

    # Darcy weak form with spatially varying permeability and wall conductivity.
    ds = Measure("ds", domain=mesh_i, subdomain_data=facets_i)

    p_darcy = Function(V)
    pd = TrialFunction(V)
    qd = TestFunction(V)

    psi_path = PERIVASCULAR_COORDINATE_XDMF

    psi = Function(V)  # scalar CG space matching p_darcy
    xf = XDMFFile(psi_path)
    xf.read_checkpoint(psi, "p_darcy", -1)

    psi.rename("psi", "perivascular coordinate")
    a = inner((k/mu) * grad(pd), grad(qd)) * dx + Lp * pd * qd * ds(interface_id)
    L = Lp * g * qd * ds(interface_id)

    A = assemble(a)
    b = assemble(L)


    solve(A, p_darcy.vector(), b, "gmres", "amg")


    p_darcy.rename("p_darcy", "Darcy pressure")
    darcy_file.write(p_darcy, 0.0)

    # Compute Darcy velocity
    W = VectorFunctionSpace(mesh_i, "CG", 1)
    u_darcy = project(-k/mu*grad(p_darcy), W, solver_type="gmres", preconditioner_type="amg")
    u_darcy.rename("u_darcy", "Darcy velocity")
    darcy_file.write(u_darcy, 0.0)


    # Save Darcy velocity as readable XDMF file
    output_file = XDMFFile(comm, os.path.join(base_folder, "darcy_velocity.xdmf"))
    output_file.write_checkpoint(u_darcy, "u_darcy", 0.0, XDMFFile.Encoding.HDF5, True)

    # save dracy pressure as a readable XDMF file
    output_file = XDMFFile(comm, os.path.join(base_folder, "darcy_pressure_output.xdmf"))
    output_file.write_checkpoint(p_darcy, "p_darcy", 0.0, XDMFFile.Encoding.HDF5, True)

    # save ginterpol
    output_file = XDMFFile(comm, os.path.join(base_folder, "stokes_pressure_output.xdmf"))
    output_file.write_checkpoint(g, "g", 0.0, XDMFFile.Encoding.HDF5, True)
    def log_planarity_ratio(u_sol, logfile="stokes_planarity.log", tag=""):
        """
        R_z = (∫ u_z^2 dV) / (∫ ||u||^2 dV)
        Appends to logfile (rank 0 only).
        """
        mesh = u_sol.function_space().mesh()
        dxm  = Measure("dx", domain=mesh)

        if u_sol.ufl_shape[0] < 3:
            raise ValueError(f"u_sol has shape {u_sol.ufl_shape}, cannot access z-component.")

        uz = u_sol[2]
        num = assemble(uz*uz * dxm)
        den = assemble(inner(u_sol, u_sol) * dxm)
        R   = num/den if den > 0.0 else 0.0


        if rank == 0:
            with open(logfile, "a") as f:
                f.write(f"{tag}: num=int(uz^2)={num:.16e}, den=int(|u|^2)={den:.16e}, R_z={R:.16e}\n")

        return R, num, den


    def log_interface_surface_metrics(u_darcy, p_darcy, p_stokes_on_interface, Lp, ds, interface_id,
                                    logfile="stokes_planarity.log", tag="Darcy"):
        """
        Computes on Γ = ds(interface_id):
        1) surface avg of |u_T| where u_T = Lp (p_S - p_D)
        2) surface avg of ||u_D|| (u_D = u_darcy)
        3) ratio = avg(||u_D||) / avg(|u_T|)
        Appends to logfile (rank 0 only) and returns the values.
        """


        dsi = ds(interface_id)
        area = assemble(Constant(1.0) * dsi)

        # u_T = Lp (p_S - p_D)
        uT = Lp * (p_stokes_on_interface - p_darcy)

        avg_abs_uT = assemble(abs(uT) * dsi) / area if area > 0.0 else 0.0
        avg_uD     = assemble(sqrt(inner(u_darcy, u_darcy)) * dsi) / area if area > 0.0 else 0.0
        ratio      = (avg_uD/ avg_abs_uT ) if avg_abs_uT > 0.0 else 0.0

        if rank == 0:
            with open(logfile, "a") as f:
                f.write(
                    f"{tag}: area(Γ)={area:.16e}, "
                    f"avg|u_T|={avg_abs_uT:.16e}, avg||u_D||={avg_uD:.16e}, "
                    f"avg||u_D||/avg|u_T|={ratio:.16e}\n"
                )

        return avg_abs_uT, avg_uD, ratio, area





    dxm  = Measure("dx", domain=mesh_i)

    gradpsi = grad(psi)
    u_norm  = sqrt(inner(u_darcy, u_darcy))
    g_norm  = sqrt(inner(gradpsi, gradpsi))

    eps = Constant(1e-16)  # avoid division by zero
    cos_theta = inner(u_darcy, gradpsi) / (u_norm * g_norm + eps)

    darcy_vol = assemble(Constant(1)* dxm)
    angle_metric = assemble(cos_theta * dxm) / darcy_vol


        # ---- metrics ----
    logfile = os.path.join(base_folder, "planarity_and_interface_metrics.log")


    R_D_z, numDz, denD = log_planarity_ratio(u_darcy, logfile=logfile, tag="Darcy (volume)")
    avg_abs_uT, avg_uD, ratio_uT_uD, areaGamma = log_interface_surface_metrics(
        u_darcy=u_darcy,
        p_darcy=p_darcy,
        p_stokes_on_interface=g,   # transferred vascular pressure
        Lp=Lp,
        ds=ds,
        interface_id=interface_id,
        logfile=logfile,
        tag="Interface (surface)"
    )

    if rank == 0:
        print(f"[Darcy] R_D,z = {R_D_z:.6e}")
        print(f"[Interface] avg|u_T| = {avg_abs_uT:.6e}, avg||u_D|| = {avg_uD:.6e}, ratio = {ratio_uT_uD:.6e}")
        print(f"Wrote metrics to: {logfile}")
    if rank == 0:
        with open(logfile, "a") as f:
            f.write(f"Darcy (volume): angle_metric=<cos(theta)>={angle_metric:.16e}\n")
        print(f"[Darcy] angle_metric = {angle_metric:.6e}")
if __name__ == '__main__':
    # Log time
    start_time = MPI.Wtime()
    main()
    end_time = MPI.Wtime()
    if rank == 0:
        print(f"Execution time: {end_time - start_time:.2f} seconds")
