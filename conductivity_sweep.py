from dolfin import *
import dolfin as df
import numpy as np
from scipy.spatial import cKDTree
import csv
import os
import sys
from pathlib import Path
from mpi4py import MPI

from fenics_functions import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.abspath(os.environ.get("FLOW_RUN_DIR", CODE_DIR))

interface_id = 103
MESH_V_FILE = os.path.join(CODE_DIR, "meshes/vascular_mesh.h5")
MESH_I_FILE = os.path.join(CODE_DIR, "meshes/interstitial_mesh.h5")
STOKES_P_FILE = os.path.join(RUN_DIR, "blood_flow", "stokes_pressure.xdmf")

mu = 1.2e-3
k0_ref = 5.2e-14
Lp_ref = 2.7e-9
tw = 3.0e-6
Rc_ref = k0_ref / (mu * Lp_ref * tw)

n_Rc_values = 40
Rc_min = 1.0e-3 * Rc_ref
Rc_max = 1.0e3 * Rc_ref
Rc_target_list = np.logspace(np.log10(Rc_min), np.log10(Rc_max), n_Rc_values)

x0_cut = 0.004771099337748345
switch_scale = 10000.0

ROOT_OUT = os.path.join(RUN_DIR, "conductivity_sweep")
MASTER_CSV = os.path.join(ROOT_OUT, "sweep_summary.csv")
PROVENANCE_DIR = os.path.join(ROOT_OUT, "provenance")
PROVENANCE_SCRIPT = "conductivity_sweep.py"
WORKFLOW_VERSION = "conductivity_sweep_v1"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the conductivity-ratio sweep.")
    parser.add_argument("--case-indices", nargs="+", type=int, choices=range(n_Rc_values),
                        help="Run selected original indices (0-39); default: all 40.")
    args = parser.parse_args()

df.parameters["mesh_partitioner"] = "ParMETIS"
df.parameters["form_compiler"]["cpp_optimize"] = True

mesh_v = Mesh()
with HDF5File(mesh_v.mpi_comm(), MESH_V_FILE, "r") as h5:
    h5.read(mesh_v, "/mesh", False)
    facets_v = MeshFunction("size_t", mesh_v, mesh_v.topology().dim() - 1, 0)
    h5.read(facets_v, "/boundaries")

mesh_i = Mesh()
with HDF5File(mesh_i.mpi_comm(), MESH_I_FILE, "r") as h5:
    h5.read(mesh_i, "/mesh", False)
    facets_i = MeshFunction("size_t", mesh_i, mesh_i.topology().dim() - 1, 0)
    h5.read(facets_i, "/boundaries")


def write_case_log(case_folder, lines):
    if rank != 0:
        return
    with open(os.path.join(case_folder, "metrics.log"), "w") as f:
        for s in lines:
            f.write(s.rstrip() + "\n")


def case_is_done(case_folder):
    done = False
    if rank == 0:
        done = os.path.exists(os.path.join(case_folder, "done.flag"))
    return comm.bcast(done, root=0)


def mark_case_done(case_folder):
    if rank != 0:
        return
    with open(os.path.join(case_folder, "done.flag"), "w") as f:
        f.write("done\n")


def ensure_master_csv():
    if rank != 0:
        return
    os.makedirs(ROOT_OUT, exist_ok=True)
    fieldnames = [
        "case_folder",
        "Rc_target",
        "Rc_index",
        "Lp_base_m_per_s_Pa",
        "k_base_m2",
        "case_kind",
        "is_baseline",
        "reference_label",
        "Rc_factor_from_ref",
        "provenance_script",
        "workflow_version",
    ]
    if not os.path.exists(MASTER_CSV):
        with open(MASTER_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return
    with open(MASTER_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        existing_fields = list(reader.fieldnames or [])
        rows = list(reader)
    changed = False
    for col in fieldnames:
        if col not in existing_fields:
            existing_fields.append(col)
            changed = True
    if not changed:
        return
    tmp = MASTER_CSV + ".tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=existing_fields)
        writer.writeheader()
        for row in rows:
            for col in existing_fields:
                row.setdefault(col, "")
            writer.writerow(row)
    os.replace(tmp, MASTER_CSV)


def append_or_replace_row(row_dict):
    if rank != 0:
        return
    with open(MASTER_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for col in row_dict:
        if col not in fieldnames:
            fieldnames.append(col)
    case_folder = row_dict["case_folder"]
    rows = [r for r in rows if r.get("case_folder") != case_folder]
    rows.append(row_dict)
    tmp = MASTER_CSV + ".tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for col in fieldnames:
                row.setdefault(col, "")
            writer.writerow(row)
    os.replace(tmp, MASTER_CSV)


def build_interface_stokes_bc_g(mesh_v, p_v, mesh_i, facets_i, interface_id):
    V = FunctionSpace(mesh_i, "CG", 1)
    gdim = mesh_i.geometry().dim()

    bm_v = BoundaryMesh(mesh_v, "exterior", False)
    X_loc = bm_v.coordinates()
    p_loc = np.empty(len(X_loc), dtype=float)
    buf = np.zeros(1, dtype=float)
    for i, x in enumerate(X_loc):
        p_v.eval(buf, x)
        p_loc[i] = buf[0]

    X_list = comm.gather(X_loc, root=0)
    p_list = comm.gather(p_loc, root=0)
    if rank == 0:
        X_src = np.vstack(X_list) if len(X_list) else np.empty((0, gdim))
        p_src = np.concatenate(p_list) if len(p_list) else np.empty((0,))
    else:
        X_src = None
        p_src = None
    X_src = comm.bcast(X_src, root=0)
    p_src = comm.bcast(p_src, root=0)

    tree = cKDTree(X_src)
    bc_probe = DirichletBC(V, Constant(0.0), facets_i, interface_id, method="topological")
    bdofs_local_all = np.array(list(bc_probe.get_boundary_values().keys()), dtype=np.int64)
    Xdof_local = V.tabulate_dof_coordinates().reshape((-1, gdim))
    bdofs_local_owned = bdofs_local_all[bdofs_local_all < len(Xdof_local)]
    Xq = Xdof_local[bdofs_local_owned]

    d, idx = tree.query(Xq, k=3, workers=-1)
    d = np.atleast_2d(d)
    idx = np.atleast_2d(idx)
    zero_hit = (d == 0.0)
    vals = np.empty(Xq.shape[0], dtype=float)
    rows_zh = np.any(zero_hit, axis=1)
    if np.any(rows_zh):
        j0 = zero_hit[rows_zh].argmax(axis=1)
        vals[rows_zh] = p_src[idx[rows_zh, j0]]
    rows_nzh = ~rows_zh
    if np.any(rows_nzh):
        w = 1.0 / d[rows_nzh]
        w /= w.sum(axis=1, keepdims=True)
        vals[rows_nzh] = np.einsum("ij,ij->i", w, p_src[idx[rows_nzh]])

    g = Function(V)
    vec = g.vector().get_local()
    vec[bdofs_local_owned] = vals
    g.vector().set_local(vec)
    g.vector().apply("insert")
    return V, g


class KExpr(UserExpression):
    def __init__(self, k_base, x_cut, scale, **kwargs):
        super().__init__(**kwargs)
        self.k_base = float(k_base)
        self.x_cut = float(x_cut)
        self.scale = float(scale)

    def eval(self, values, x):
        values[0] = self.k_base / self.scale if x[0] > self.x_cut else self.k_base

    def value_shape(self):
        return ()


class LpExpr(UserExpression):
    def __init__(self, Lp_base, x_cut, scale, **kwargs):
        super().__init__(**kwargs)
        self.Lp_base = float(Lp_base)
        self.x_cut = float(x_cut)
        self.scale = float(scale)

    def eval(self, values, x):
        values[0] = self.Lp_base / self.scale if x[0] > self.x_cut else self.Lp_base

    def value_shape(self):
        return ()


def main(case_indices=None):
    if rank == 0:
        os.makedirs(ROOT_OUT, exist_ok=True)
        os.makedirs(PROVENANCE_DIR, exist_ok=True)
        start_logging(ROOT_OUT)
        save_current_script(PROVENANCE_DIR)
        print(f"[info] Rc_ref = {Rc_ref:.16e}")
        print(f"[info] Rc_min = {Rc_min:.16e}")
        print(f"[info] Rc_max = {Rc_max:.16e}")
        print(f"[info] n_Rc_values = {n_Rc_values}")
        print(f"[info] workflow_version = {WORKFLOW_VERSION}")
    comm.Barrier()

    ensure_master_csv()
    comm.Barrier()

    V_v = FunctionSpace(mesh_v, "CG", 1)
    p_v = Function(V_v)
    stokes_pressure_file = XDMFFile(STOKES_P_FILE)
    stokes_pressure_file.read_checkpoint(p_v, "p_v", -1)

    V, g = build_interface_stokes_bc_g(mesh_v, p_v, mesh_i, facets_i, interface_id)

    g_out = df.XDMFFile(comm, os.path.join(ROOT_OUT, "g_boundary.xdmf"))
    g_out.parameters["flush_output"] = True
    g_out.parameters["functions_share_mesh"] = True
    g.rename("g_interpolated", "Interpolated Darcy BC from Stokes")
    g_out.write(g, 0.0)
    g_out.close()

    W = VectorFunctionSpace(mesh_i, "CG", 1)
    ds = Measure("ds", domain=mesh_i, subdomain_data=facets_i)
    pd = TrialFunction(V)
    qd = TestFunction(V)
    k_base = k0_ref

    for i_Rc, Rc_target in enumerate(Rc_target_list):
        if case_indices is not None and i_Rc not in case_indices:
            continue
        Lp_base = k_base / (mu * tw * Rc_target)
        case_folder = os.path.join(ROOT_OUT, f"case_{i_Rc:02d}")

        if rank == 0:
            os.makedirs(case_folder, exist_ok=True)
        comm.Barrier()

        if case_is_done(case_folder):
            if rank == 0:
                print(f"[skip] already done -> {case_folder}")
            continue

        k_expr = KExpr(k_base, x0_cut, switch_scale, degree=0)
        Lp_expr = LpExpr(Lp_base, x0_cut, switch_scale, degree=0)
        a = inner((k_expr / mu) * grad(pd), grad(qd)) * dx + Lp_expr * pd * qd * ds(interface_id)
        L = Lp_expr * g * qd * ds(interface_id)

        p_darcy = Function(V)
        A = assemble(a)
        b = assemble(L)
        solve(A, p_darcy.vector(), b, "gmres", "amg")

        u_darcy = project(
            -(k_expr / mu) * grad(p_darcy),
            W,
            solver_type="gmres",
            preconditioner_type="amg",
        )

        sol = df.XDMFFile(comm, os.path.join(case_folder, "darcy_solution.xdmf"))
        sol.parameters["flush_output"] = True
        sol.parameters["functions_share_mesh"] = True
        p_darcy.rename("p_darcy", "Darcy pressure")
        u_darcy.rename("u_darcy", "Darcy velocity")
        sol.write(p_darcy, 0.0)
        sol.write(u_darcy, 0.0)
        sol.close()

        outp = XDMFFile(comm, os.path.join(case_folder, "darcy_pressure_output.xdmf"))
        outp.write_checkpoint(p_darcy, "p_darcy", 0.0, XDMFFile.Encoding.HDF5, True)
        outp.close()

        outu = XDMFFile(comm, os.path.join(case_folder, "darcy_velocity_output.xdmf"))
        outu.write_checkpoint(u_darcy, "u_darcy", 0.0, XDMFFile.Encoding.HDF5, True)
        outu.close()

        outg = XDMFFile(comm, os.path.join(case_folder, "g_output.xdmf"))
        outg.write_checkpoint(g, "g", 0.0, XDMFFile.Encoding.HDF5, True)
        outg.close()

        lines = [
            f"workflow_version = {WORKFLOW_VERSION}",
            f"provenance_script = {PROVENANCE_SCRIPT}",
            f"Rc_ref = {Rc_ref:.16e}",
            f"Rc_min = {Rc_min:.16e}",
            f"Rc_max = {Rc_max:.16e}",
            f"Rc_index = {i_Rc}",
            f"Rc_target = {Rc_target:.16e}",
            f"Rc_factor_from_ref = {Rc_target / Rc_ref:.16e}",
            f"mu [kg/(m*s)] = {mu:.16e}",
            f"tw [m] = {tw:.16e}",
            f"k_base [m^2] = {k_base:.16e}",
            f"Lp_base [m/(s Pa)] = {Lp_base:.16e}",
            "case_kind = logspace",
            "is_baseline = 0",
            "reference_label = ",
        ]
        write_case_log(case_folder, lines)

        row = {
            "case_folder": case_folder,
            "Rc_target": f"{Rc_target:.16e}",
            "Rc_index": str(i_Rc),
            "Lp_base_m_per_s_Pa": f"{Lp_base:.16e}",
            "k_base_m2": f"{k_base:.16e}",
            "case_kind": "logspace",
            "is_baseline": "0",
            "reference_label": "",
            "Rc_factor_from_ref": f"{Rc_target / Rc_ref:.16e}",
            "provenance_script": PROVENANCE_SCRIPT,
            "workflow_version": WORKFLOW_VERSION,
        }
        append_or_replace_row(row)
        mark_case_done(case_folder)
        comm.Barrier()

        if rank == 0:
            print(f"[done] Rc_idx={i_Rc:02d}, Rc={Rc_target:.6e} -> {case_folder}")

    if rank == 0:
        print(f"[summary] wrote master CSV: {MASTER_CSV}")


if __name__ == "__main__":
    t0 = MPI.Wtime()
    main(args.case_indices)
    t1 = MPI.Wtime()
    if rank == 0:
        print(f"Execution time: {t1 - t0:.2f} s")
