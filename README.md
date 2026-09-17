# ACIT4610: Job Shop Scheduling Problem Using Genetic Algorithms

Group 2
Course: ACIT4610

## Overview

This project implements a Genetic Algorithm (GA) to solve the Job Shop Scheduling Problem (JSSP), a classic combinatorial optimization problem in manufacturing. The goal is to minimize the makespan (total completion time) by optimally scheduling jobs on machines.

## Setup

1. Download or clone the repository.

2. Create a virtual environment (recommended):

**On Windows:**
```bash
python -m venv venv
```

**On macOS/Linux:**
```bash
python3 -m venv venv
```

3. Activate the virtual environment:

**On Windows:**
```bash
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
ACIT4610-A1-Group2/
├── ga_core.py              # Genetic Algorithm implementation
├── decoder.py              # Schedule Building Algorithm & instance parsing
├── experiments.py          # Experiment runner & plotting functions
├── config.yaml             # Configuration (parameters, instances, output paths)
├── requirements.txt        # Python dependencies
├── util/
│   ├── __init__.py         # Marks util as a package
│   └── LoadConfig.py       # Configuration loader
├── data/                   # JSPLib benchmark instances
│   ├── la01.txt (small)
│   ├── la03.txt (small)
│   ├── la16.txt (medium)
│   ├── la18.txt (medium)
│   ├── la31.txt (large)
│   └── la33.txt (large)
└── results/                # Generated outputs (created on first run)
```

## How the Modules Are Connected

`config.yaml` holds every configurable value used across the project, including the dataset paths, the GA parameter sets, the tournament size, and the output filenames. Nothing in the project hardcodes these values directly.

The `util` package contains a single module, `LoadConfig.py`, which reads `config.yaml` once and exposes its values as named constants (for example `DATA_DIR`, `PARAMETER_SETS`, `RESULTS_DIR`). `decoder.py`, `ga_core.py`, and `experiments.py` each import the specific constants they need from `LoadConfig.py`, rather than reading `config.yaml` directly themselves.

`decoder.py` is responsible for parsing a benchmark instance file into a Python dictionary, and for decoding a chromosome into a feasible schedule through its `decode` function. This module has no dependency on the other two, it can be run and tested on its own.

`ga_core.py` implements the Genetic Algorithm itself, including population initialization, selection, crossover, mutation, and the main evolutionary loop inside `run_ga`. This module does not import anything from `decoder.py`. Instead, `run_ga` accepts a `decode_function` as an argument, so the caller decides which decoding logic to use when evaluating each chromosome's fitness.

`experiments.py` is the module that connects the other two together. It imports `parse_instance` and `decode` from `decoder.py`, and `run_ga` from `ga_core.py`. For every benchmark instance and every parameter set, it calls `run_ga`, passing `decode` in as the `decode_function`, so the Genetic Algorithm and the Schedule Building Algorithm work together without either module needing to know the internal details of the other.

Both `decoder.py` and `experiments.py` write their output into the `results` folder, which is created automatically at runtime if it does not already exist. Running `decoder.py` produces the Gantt chart images and prints the schedule table used in the report's decoding example. Running `experiments.py` produces `experiment_results.csv`, along with the convergence comparison and category comparison plots. The contents of this folder are the exact figures and tables referenced throughout the report.

## Running the Experiments

`experiments.py` runs the full experiment sweep by default, testing every benchmark instance with every parameter set for `NUMBER_OF_RUNS` independent runs each:

```bash
python3 experiments.py
```

This produces `experiment_results.csv` and the convergence and category comparison plots, and can take a few minutes depending on the values set in `config.yaml`. A faster, 3-run pipeline check also exists in the file but is disabled by default, wrapped in triple quotes to prevent it from running.


## Authors

Group 2

Course: ACIT4610