# ACIT 4610: Job Shop Scheduling Problem Using Genetic Algorithms

Group 2 Assignment - Mid-Term Portfolio Project 2026

## Overview

This project implements a Genetic Algorithm (GA) to solve the Job Shop Scheduling Problem (JSSP), a classic combinatorial optimization problem in manufacturing. The goal is to minimize the makespan (total completion time) by optimally scheduling jobs on machines.

## Features

- **Genetic Algorithm Core**: Tournament selection, order crossover, swap mutation
- **Schedule Building Algorithm (SBA)**: Semi-active schedule building with precedence and machine-capacity constraints
- **Benchmark Instances**: 6 JSPLib instances across 3 problem categories (small, medium, large)
- **Statistical Analysis**: Min, max, average makespan, standard deviation, execution time, convergence tracking
- **Parameter Tuning**: 3 different GA parameter sets to compare exploration vs. exploitation
- **Visualization**: Gantt charts, convergence curves, category comparison plots

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. Clone or extract this repository:
```bash
cd ACIT4610-A1-Group2-main
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install pyyaml matplotlib numpy
```

## Usage

### Run the Decoder Demo
This generates example Gantt charts for a tiny toy problem and a real benchmark instance (la01):

```bash
python3 decoder.py
```

**Output:**
- `results/example_gantt.png` — Gantt chart for 2-job, 2-machine toy example
- `results/example_la01_gantt.png` — Gantt chart for real la01 instance
- Console output showing schedule table with job, operation, machine, start, and end times

### Run the Full Experiment
This runs the GA on all 6 benchmark instances with all 3 parameter sets, 20 independent runs each. **Warning: takes 5-15 minutes.**

```bash
python3 experiments.py
```

**Output:**
- `results/experiment_results.csv` — Complete results table with all statistics
- `results/convergence_comparison.png` — Convergence curves for 3 parameter sets (showing early vs. late-stage effects)
- `results/category_comparison.png` — Bar chart comparing average makespan across problem sizes
- `results/single_convergence_la31.png` — Single convergence curve for representative instance (la31)
- Console table with formatted results

## Project Structure

```
ACIT4610-A1-Group2-main/
├── ga_core.py              # Genetic Algorithm implementation
├── decoder.py              # Schedule Building Algorithm & instance parsing
├── experiments.py          # Experiment runner & plotting functions
├── config.yaml             # Configuration (parameters, instances, output paths)
├── requirements.txt        # Python dependencies
├── util/
│   └── LoadConfig.py       # Configuration loader
├── data/                   # JSPLib benchmark instances
│   ├── la01.txt (small)
│   ├── la03.txt (small)
│   ├── la16.txt (medium)
│   ├── la18.txt (medium)
│   ├── la31.txt (large)
│   └── la33.txt (large)
└── results/                # Generated outputs (created on first run)
    ├── example_gantt.png
    ├── example_la01_gantt.png
    ├── experiment_results.csv
    ├── convergence_comparison.png
    ├── category_comparison.png
    └── single_convergence_la31.png
```

## Configuration

Edit `config.yaml` to customize:
- **Dataset instances**: Which benchmark files to test
- **GA parameters**: Population size, generations, crossover/mutation rates
- **Number of runs**: How many independent runs per combination (default: 20)
- **Output directory**: Where to save results

### Parameter Sets

Three GA parameter sets are defined for comparison:

1. **Set 1 (Baseline)**: Population=50, Generations=200, Crossover=0.8, Mutation=0.1
   - Standard recommended settings
   
2. **Set 2 (Exploration-heavy)**: Population=100, Generations=200, Crossover=0.7, Mutation=0.3
   - Larger population and higher mutation for genetic diversity
   - Best for avoiding local optima
   
3. **Set 3 (Exploitation-heavy)**: Population=30, Generations=300, Crossover=0.95, Mutation=0.05
   - Smaller population, high crossover for aggressive recombination
   - Risk: premature convergence

## Algorithm Details

### Chromosome Representation
A chromosome is a list of job numbers, where each job appears once per operation it has. Reading left-to-right, the k-th appearance of a job number refers to that job's k-th operation.

Example: `[0, 1, 0, 1]` means: job 0 op 1, job 1 op 1, job 0 op 2, job 1 op 2

### Schedule Building Algorithm (SBA)
Uses a **semi-active** strategy:
1. Process chromosome left-to-right
2. For each operation, schedule it as early as possible respecting:
   - **Precedence constraint**: can't start before previous operation of same job finishes
   - **Machine capacity**: can't start before machine is free
3. Calculate makespan as maximum end time across all operations

### Genetic Operators
- **Selection**: Tournament selection (size=3) — picks chromosome with best (lowest) makespan
- **Crossover**: Order crossover — maintains valid job operation counts
- **Mutation**: Swap mutation — exchanges two random genes
- **Elitism**: Best chromosome automatically copied to next generation

## Results Interpretation

### experiment_results.csv Columns
- `category`: Problem size (small, medium, large)
- `instance`: Benchmark instance filename
- `parameter_set`: Which GA parameter set was used
- `best_makespan`: Best solution found (minimum)
- `worst_makespan`: Worst solution found (maximum)
- `average_makespan`: Mean makespan across 20 runs
- `std_dev_makespan`: Standard deviation (stability indicator)
- `average_execution_time`: Mean runtime per run (seconds)
- `total_execution_time`: Total time for all 20 runs
- `average_convergence_generation`: Generation where best solution was found (earlier = faster convergence)

### How to Analyze Results

1. **Best solutions**: Compare `best_makespan` across parameter sets
2. **Stability**: Lower `std_dev_makespan` indicates more stable solutions
3. **Efficiency**: Compare `average_execution_time` for speed
4. **Convergence speed**: Lower `average_convergence_generation` means faster convergence
5. **Scalability**: Compare results across `category` (small → medium → large)

## Output Files

- **Gantt Charts** (.png): Visual schedule showing jobs (colored bars) on machines (rows) over time
- **Convergence Comparison Plot**: Line graph showing best fitness per generation for each parameter set (reveals early vs. late-stage behavior)
- **Category Comparison Plot**: Bar chart showing average makespan for each problem size
- **Single Convergence Plot**: Single line showing how quickly one instance converges
- **CSV Results**: Tabular data for further analysis

## Libraries Used

- **random**: Chromosome initialization and genetic operators
- **math**: Statistical calculations (standard deviation)
- **matplotlib.pyplot**: Gantt charts and convergence plots
- **numpy**: Not directly used, available for future enhancements
- **yaml**: Configuration file parsing
- **os**: File and directory operations
- **time**: Execution time measurement

## Notes

- Each experiment run is independent (uses different random seeds)
- Results may vary slightly between runs due to stochastic nature of GA
- For large instances (la31, la33), computation may take 1-2 minutes per run
- Gantt charts open automatically in default image viewer; close to continue execution

## Report Requirements Coverage

This code supports all report requirements:

1. ✅ GA design description (see `ga_core.py` and Algorithm Details above)
2. ✅ Decoder pseudocode (see `decoder.py` with step-by-step comments)
3. ✅ Objective/fitness function (makespan minimization, documented in `ga_core.py`)
4. ✅ Example chromosome decoding (decoder.py demo produces this)
5. ✅ Metrics comparison (experiment_results.csv contains all required statistics)
6. ✅ Time table (see `average_execution_time` column in CSV)
7. ✅ Parameter correlation analysis (convergence_comparison.png and category_comparison.png)
8. ✅ Early vs. late-stage effects (convergence_comparison.png shows parameter impact across generations)

## Troubleshooting

**ImportError: No module named 'matplotlib'**
```bash
pip install matplotlib
```

**ImportError: dlopen(...): architecture mismatch**
- Recreate virtual environment: `rm -rf venv` then repeat Installation steps

**Plots not showing**
- On macOS/Linux, plots should open automatically; close each window to continue
- If not, check `results/` folder for saved PNG files

**Memory error on large instances**
- Reduce `NUMBER_OF_RUNS` in `config.yaml`
- Or increase system virtual memory

## Authors

ACIT 4610 Group 2 - 2026

## License

Academic use only (OsloMet Assignment)
