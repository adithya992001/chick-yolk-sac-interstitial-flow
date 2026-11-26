from dolfin import *
import dolfin as df
import numpy as np
from scipy.spatial import cKDTree
import csv
import os
import sys
from mpi4py import MPI

from fenics_functions import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.environ.get("E7_RUN_DIR", CODE_DIR))

interface_id = 103
MESH_V_FILE = os.path.join(CODE_DIR, "mesh_original_vasculature_corrected_scale/vasculature_decimated_255K_0,0001.h5")
MESH_I_FILE = os.path.join(CODE_DIR, "mesh_original_vasculature_corrected_scale/interstitium_0,00008.h5")
STOKES_P_FILE = os.path.join(STUDY_DIR, "sharp_Stokes_3D", "stokes_pressure.xdmf")

mu = 1.2e-3
k0_ref = 5.2e-14
Lp_ref = 2.7e-9
tw = 3.0e-6
Rp_ref = k0_ref / (mu * Lp_ref * tw)

x0_cut = 0.004771099337748345
switch_scale = 10000.0

ROOT_OUT = os.path.join(STUDY_DIR, "Darcy_flow_parametric_sensitivity_updated_Rp_0.001_to_1000")
MASTER_CSV = os.path.join(ROOT_OUT, "sweep_summary.csv")
PROVENANCE_DIR = os.path.join(ROOT_OUT, "provenance")
PROVENANCE_SCRIPT = "./darcy_flow_parametric_sensitivity_updated_Rp_0p001_to_1000_special_cases.py"
WORKFLOW_VERSION = "rp_only_updated_0p001_to_1000_v1"

SPECIAL_CASES = [
    {
        "case_name": f"Rp_baseline_Rpref={Rp_ref:.6e}",
        "case_kind": "baseline",
        "special_label": "baseline",
        "is_baseline": "1",
        "Rp_index": "-1",
        "Rp_factor_from_ref": 1.0,
    },
    {
        "case_name": "Lp_high_Rpfac=1.000000e-03",
        "case_kind": "lp_high",
        "special_label": "Lp_high",
        "is_baseline": "0",
        "Rp_index": "-2",
        "Rp_factor_from_ref": 1.0e-3,
    },
]

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
        "Rp_target",
        "Rp_index",
        "Lp_base_m_per_s_Pa",
        "k_base_m2",
        "case_kind",
        "is_baseline",
        "special_label",
        "Rp_factor_from_ref",
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


def main():
    if rank == 0:
        os.makedirs(ROOT_OUT, exist_ok=True)
        os.makedirs(PROVENANCE_DIR, exist_ok=True)
        save_current_script(PROVENANCE_DIR)
        print(f"[info] Rp_ref = {Rp_ref:.16e}")
        for cfg in SPECIAL_CASES:
            print(f"[info] special case {cfg['special_label']}: Rp/Rp_ref = {cfg['Rp_factor_from_ref']:.6e}")
    comm.Barrier()

    ensure_master_csv()
    comm.Barrier()

    V_v = FunctionSpace(mesh_v, "CG", 1)
    p_v = Function(V_v)
    stokes_pressure_file = XDMFFile(STOKES_P_FILE)
    stokes_pressure_file.read_checkpoint(p_v, "p_v", -1)

    V, g = build_interface_stokes_bc_g(mesh_v, p_v, mesh_i, facets_i, interface_id)
    W = VectorFunctionSpace(mesh_i, "CG", 1)
    ds = Measure("ds", domain=mesh_i, subdomain_data=facets_i)
    pd = TrialFunction(V)
    qd = TestFunction(V)
    k_base = k0_ref

    for cfg in SPECIAL_CASES:
        Rp_target = Rp_ref * float(cfg["Rp_factor_from_ref"])
        Lp_base = k_base / (mu * tw * Rp_target)
        case_folder = os.path.join(ROOT_OUT, cfg["case_name"])

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
        u_darcy = project(-(k_expr / mu) * grad(p_darcy), W, solver_type="gmres", preconditioner_type="amg")

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
            f"case_kind = {cfg['case_kind']}",
            f"special_label = {cfg['special_label']}",
            f"is_baseline = {cfg['is_baseline']}",
            f"Rp_ref = {Rp_ref:.16e}",
            f"Rp_target = {Rp_target:.16e}",
            f"Rp_factor_from_ref = {float(cfg['Rp_factor_from_ref']):.16e}",
            f"mu [kg/(m*s)] = {mu:.16e}",
            f"tw [m] = {tw:.16e}",
            f"k_base [m^2] = {k_base:.16e}",
            f"Lp_base [m/(s Pa)] = {Lp_base:.16e}",
        ]
        write_case_log(case_folder, lines)

        row = {
            "case_folder": case_folder,
            "Rp_target": f"{Rp_target:.16e}",
            "Rp_index": cfg["Rp_index"],
            "Lp_base_m_per_s_Pa": f"{Lp_base:.16e}",
            "k_base_m2": f"{k_base:.16e}",
            "case_kind": cfg["case_kind"],
            "is_baseline": cfg["is_baseline"],
            "special_label": cfg["special_label"],
            "Rp_factor_from_ref": f"{float(cfg['Rp_factor_from_ref']):.16e}",
            "provenance_script": PROVENANCE_SCRIPT,
            "workflow_version": WORKFLOW_VERSION,
        }
        append_or_replace_row(row)
        mark_case_done(case_folder)
        comm.Barrier()

        if rank == 0:
            print(f"[done] {cfg['special_label']} (Rp/Rp_ref={float(cfg['Rp_factor_from_ref']):.6e}) -> {case_folder}")


if __name__ == "__main__":
    t0 = MPI.Wtime()
    main()
    t1 = MPI.Wtime()
    if rank == 0:
        print(f"Execution time: {t1 - t0:.2f} s")
