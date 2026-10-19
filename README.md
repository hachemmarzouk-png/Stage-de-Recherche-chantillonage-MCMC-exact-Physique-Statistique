# Exact Sampling for the Solid-on-Solid Model

**Coupling from the Past (Propp–Wilson) · Metropolis dynamics · Probability & statistical mechanics**

**Research internship — May–July 2026**  
Hachem Marzouk · M1 Applied Mathematics, Institut Polytechnique de Paris / Université Paris-Saclay  
**Host:** Laboratoire de Probabilités, Statistique et Modélisation (LPSM)  
**Supervisor:** Benoît Laslier

## Research question

How does a random interface behave at **low but finite temperature** under mixed boundary conditions? The integer-valued **Solid-on-Solid (SOS)** model is defined on a square grid, with heights fixed at the bottom and top and free lateral boundaries.

At zero temperature, energy minimizers are constant along horizontal rows; their heights form a monotone staircase with a random-walk representation. At finite inverse temperature `β`, the question is whether the interface still admits a useful one-dimensional description.

## Implementation

- **Local Metropolis dynamics** for the finite, height-truncated SOS model.
- **Monotone Coupling from the Past (CFTP)** with common randomness, minimal and maximal configurations, and doubling look-back windows that preserve all previously generated random updates.
- Numerical visualization of sampled configurations and height profiles.
- **Exploratory line-update dynamics**, built from repeated local moves on selected rows; not claimed as a validated speed-up.

The exact-sampling claim concerns the **finite height-truncated stationary distribution**, not an asymptotic statement about the finite-temperature scaling limit.

## Numerical observations

The research report documents progressively flatter rows as `β` increases, while the coalescence horizon grows rapidly. For example, the reported **median CFTP horizons for N = 6** were:

| Inverse temperature β | Median horizon (local updates) |
|---|---:|
| 0.5 | 292,864 |
| 1 | 292,864 |
| 2 | 1,171,456 |
| 3 | 7,028,736 |

These computational limitations motivated an analysis of more collective updates and two one-dimensional toy models for mixing-time estimates.

## Quickstart

Requires Python 3.10+.

```bash
python -m pip install -r requirements.txt
python scripts/run_cftp.py --N 2 --beta 1.0 --seed 0 --plot
```

The command saves a sample matrix (`sample.npy`) and metadata (`run.json`) in `outputs_demo/`. For the experimental line-update variant:

```bash
python scripts/run_cftp.py --N 2 --beta 1.0 --seed 0 --mode line --k 20
```

Run lightweight checks:

```bash
python -m unittest discover -s tests -v
```

**Caution:** Large N and large β may require lengthy runs and significant memory. The small example is for demonstration rather than reproducing the full experimental study.

## Files

- `src/sos_cftp.py` — original research implementation of Metropolis/CFTP and exploratory line updates.
- `scripts/run_cftp.py` — reproducible CLI for small experiments.
- `tests/test_cftp.py` — basic boundary, energy and reproducibility tests.
- `report/rapport_stage_final.pdf` — original French report (**PDF upload pending**).

The original report is available in the accompanying downloadable project package. It will be linked here once its binary file is uploaded to the repository.

**Keywords:** Markov chains · perfect simulation · Gibbs measures · stochastic processes · numerical experiments.
