import numpy as np
from fenics import *
import os
import sys
from petsc4py import PETSc
import ufl as ufl
from fenics_functions import *
from dolfin import PETScOptions

from mpi4py import MPI
import dolfin as df
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
df.parameters["mesh_partitioner"] = "ParMETIS"
df.parameters["form_compiler"]["cpp_optimize"]  = True
if rank != 0:
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.abspath(os.environ.get("FLOW_RUN_DIR", CODE_DIR))
MESH_DIR = os.path.join(CODE_DIR, "meshes")
VASCULAR_MESH_H5 = os.path.join(MESH_DIR, "vascular_mesh.h5")

# Inlet/outlet cuts and the reference pressure point, in metres.
INLET_OUTLET_X_CUT = 0.004771099337748345
INLET_OUTLET_Y_SPLIT = 0.0074172185430463585
PRESSURE_POINT = (0.004982781456953643, 0.00857086092715232, -1.7049006622516557e-05)

def main():
    base_folder = os.path.join(RUN_DIR, "blood_flow")
    if rank == 0:
        os.makedirs(base_folder, exist_ok=True)

    if rank == 0:
        start_logging(base_folder)
        save_current_script(base_folder)
    comm.Barrier()
    file = XDMFFile(comm, os.path.join(base_folder, "solution.xdmf"))
    file.parameters["flush_output"] = True
    file.parameters["functions_share_mesh"] = True

    h5path = VASCULAR_MESH_H5
    mesh = Mesh()
    with HDF5File(MPI.COMM_WORLD, h5path, "r") as h5:
        h5.read(mesh, "/mesh", False)
        tdim = mesh.topology().dim()
        facets = MeshFunction("size_t", mesh, tdim - 1, 0)
    ds = Measure("ds", domain=mesh, subdomain_data=facets)

    # ----------------------- finite-element spaces ---------------------------
    P2 = df.VectorElement("Lagrange", mesh.ufl_cell(), 1)
    P1 = df.FiniteElement ("Lagrange", mesh.ufl_cell(), 1)
    comm.Barrier()
    ME = df.MixedElement([P2, P1])
    W  = df.FunctionSpace(mesh, ME)

    # ------------------------- variational formulation -----------------------
    nu = Constant(2.8E-6)
    rho = 1060                  # density [kg/m^3]
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    class Walls(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1e-8
            return on_boundary

    walls = Walls(); walls.mark(facets, 116)

    class Inlet(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] < INLET_OUTLET_Y_SPLIT and x[0] > (INLET_OUTLET_X_CUT - 1E-7)
    inlet = Inlet()
    inlet.mark(facets, 117)
    class Outlet(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] > INLET_OUTLET_Y_SPLIT and x[0] > (INLET_OUTLET_X_CUT - 1E-7)
    outlet = Outlet()
    outlet.mark(facets, 118)

    f = Constant((0.0, 0.0, 0.0))  # body force (optional, set to zero)
    def eps(w):   return sym(grad(w))
    a = (inner(2*nu*eps(u), eps(v)) + div(u)*q/rho - div(v)*p/rho ) * dx

    # Apply pressure traction at the inlet (tag 117).
    n = FacetNormal(mesh)
    t_in = Constant(65.32/rho)                # [Pa]
    F = -dot(t_in*n, v) * ds(117)          # Outlet (tag 118) has zero traction.

    #Adding GLS stabilization
    h = CellDiameter(mesh)
    tau_momentum = h**2/(4*nu)
    gls_lhs = ( tau_momentum*(inner((grad(p)/rho - div(2*nu*eps(u))),(grad(q)/rho - div(2*nu*eps(v))))))* dx
    gls_rhs = ( tau_momentum*(inner(f,(grad(q)/rho - div(2*nu*eps(v))))))* dx
    a += gls_lhs
    F += gls_rhs
    # ---------------------------- solve --------------------------------------
    zeroV = Constant((0.0, 0.0, 0.0))
    bc_walls = DirichletBC(W.sub(0), zeroV, facets, 116)
    pressure_location = Point(*PRESSURE_POINT)
    def pressure_point(x, on_boundary):
        tol = 1E-7
        check = pressure_location[0] - tol < x[0] < pressure_location[0] + tol and \
                pressure_location[1] - tol < x[1] < pressure_location[1] + tol and \
                pressure_location[2] - tol < x[2] < pressure_location[2] + tol
        if check:
            print("Found pressure point on rank", comm.rank)
        return check

    bc_p0 = DirichletBC(W.sub(1), Constant(0), pressure_point, method="pointwise")

    inlet_velocity = Constant((-1E-3, 0.0, 0.0))
    bc_p1 = DirichletBC(W.sub(0), inlet_velocity, facets, 117)
    # ---------------------------- solve --------------------------------------
    w = Function(W, name="solution")
    print("Solving...")
    A = assemble(a)
    b = assemble(F)
    bc_walls.apply(A, b)

    PETScOptions.set("pc_type", "asm")
    PETScOptions.set("pc_asm_overlap", 1) # number of layers of overlap
    PETScOptions.set("sub_pc_type", "lu")
    PETScOptions.set("ksp_max_it", 20)
    PETScOptions.set("ksp_rtol", 1e-2)
    PETScOptions.set("ksp_atol", 1e-6)
    PETScOptions.set("ksp_monitor")
    PETScOptions.set("ksp_converged_reason")
    PETScOptions.set("monitor_true_residual")
    PETScOptions.set("ksp_type", "gmres")
    solver = PETScKrylovSolver()
    solver.set_operator(A)
    solver.set_from_options()
    # Solve
    w = Function(W)

    solve(A, w.vector(), b, "mumps")
    print("Solved.")
    u_sol, p_sol = w.split(deepcopy=True)
    u_sol.rename("velocity", "")
    p_sol.rename("pressure", "")

    # ---------------------------- output -------------------------------------

    file.write(u_sol, 0.0)
    file.write(p_sol, 0.0)

    # Compute flow rate at outlet and inlet and check they are the same
    n = FacetNormal(mesh)
    flow_inlet = assemble(dot(u_sol, n)*ds(117))
    flow_outlet = assemble(dot(u_sol, n)*ds(118))
    if rank == 0:
        print(f"Flow rate at inlet: {flow_inlet:.6e} m^3/s")
        print(f"Flow rate at outlet: {flow_outlet:.6e} m^3/s")
        print(f"Difference %: {(flow_inlet + flow_outlet)/flow_inlet:.6e}")

    # Save the pressure solution


    save_file = df.XDMFFile(comm, os.path.join(base_folder, "stokes_pressure.xdmf"))
    save_file.write_checkpoint(p_sol, "p_v", 0.0, df.XDMFFile.Encoding.HDF5, True)


    def log_planarity_ratio(u_sol, logfile="stokes_planarity.log", tag="Stokes"):
        """
        Computes  R_{S,z} = (∫ u_z^2 dV) / (∫ ||u||^2 dV)  and appends it to a log file.
        Writes only from rank 0.
        """
        mesh = u_sol.function_space().mesh()
        dxm  = Measure("dx", domain=mesh)

        # safety: make sure we're in 3D velocity
        if u_sol.ufl_shape[0] < 3:
            raise ValueError(f"u_sol has shape {u_sol.ufl_shape}, cannot access z-component.")

        uz = u_sol[2]

        num = assemble(uz*uz * dxm)
        den = assemble(inner(u_sol, u_sol) * dxm)

        R = num/den if den > 0.0 else 0.0


        if comm.rank == 0:
            with open(logfile, "a") as f:
                f.write(f"{tag}: num=int(uz^2)={num:.16e}, den=int(|u|^2)={den:.16e}, R_S_z={R:.16e}\n")

        return R, num, den
    logfile = os.path.join(base_folder, "stokes_planarity.log")
    R_S_z, num, den = log_planarity_ratio(u_sol, logfile=logfile, tag="Blood")
    if rank == 0:
        print(f"R_S_z = {R_S_z:.6e} (written to stokes_planarity.log)")




if __name__ == '__main__':
    # Log time
    start_time = MPI.Wtime()
    main()
    end_time = MPI.Wtime()
    if rank == 0:
        print(f"Execution time: {end_time - start_time:.2f} seconds")
