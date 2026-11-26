# Interstitial flow in the chick yolk sac exhibits organ-scale patterns driven by segregated leakage and drainage

## Abstract

Interstitial flow plays a key role in drug delivery, angiogenesis, cancer, and edema. Recent studies show that the interstitium can provide connected pathways capable of supporting long-distance material transport. However, these studies do not establish whether physiological interstitial flow is organized over long distances. A connected interstitial space could support local-scale flow patterns, with short streamlines determined primarily by nearby vascular filtration, or large-scale patterns, with long streamlines where local flow is influenced by filtration from distant vessels. Here, we resolve this question by computing interstitial flow across the chick embryo yolk sac, a bounded organ-scale interstitial region that allows large-scale flow patterns to be identified within the complete simulated domain, where the largest spatial scale of flow patterns is organ-scale. We find that physiological interstitial flow in the yolk sac exhibits organ-scale patterns, where interstitial flow speed is substantially larger than transvascular flow speed. Sensitivity analysis identifies an interstitial-to-vessel-wall hydraulic conductivity ratio that controls the transition from local-scale to organ-scale flow. Mechanistically, the organ-scale patterns arise from the spatial segregation of the leakage and drainage of interstitial fluid from the vessels. These findings have important implications for biological transport by interstitial flow, suggesting that flow-mediated cues may be sensed locally but generated nonlocally.

[Read the paper on arXiv](https://arxiv.org/abs/2609.13249).

## Requirements and meshes

The solvers require:

- **Legacy FEniCS/DOLFIN 2019.1.0**, using the `fenics`/`dolfin` API.
- NumPy and SciPy.
- mpi4py, petsc4py, and an MPI runtime.
- MPI-enabled HDF5 and PETSc, with ParMETIS, MUMPS, and AMG support.

Place the two required meshes in `meshes/`:

- `vascular_mesh.h5` — vascular volume mesh.
- `interstitial_mesh.h5` — interstitial volume mesh and boundary tags.

Mesh files will be provided as GitHub Release assets. File sizes and checksums are listed in [manifest.json](meshes/manifest.json).

## Run

From the repository directory, run the commands below in order. Set `NPROCS` to the number of MPI processes available on your system; the example uses 128.

```bash
export NPROCS=128
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mpirun -np "$NPROCS" python stokes_flow.py
mpirun -np "$NPROCS" python perivascular_coordinate.py
mpirun -np "$NPROCS" python darcy_flow.py
mpirun -np "$NPROCS" python conductivity_sweep.py
mpirun -np "$NPROCS" python reference_cases.py
```

## Paper results

The generated solutions correspond to these results in the paper:

| Generated solution | Paper results |
|---|---|
| Blood pressure and velocity | Section 2.2, Figure 2 |
| Baseline and local-scale interstitial pressure and velocity | Sections 2.3–2.4, Figure 3 |
| Conductivity-ratio sweep | Section 2.4, Figure 4 and supplementary Figure 9 |

These scripts generate the solution fields. Postprocessing code will be uploaded soon.

## License

This code is released under the [MIT License](LICENSE).
