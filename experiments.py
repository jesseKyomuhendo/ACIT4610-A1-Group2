# Experiment runner for the Job Shop Scheduling Problem (ACIT 4610).
#
# This file runs the GA (from ga_core.py) against the real benchmark
# instances (loaded through decoder.py), repeats each combination of
# instance + parameter setting several independent times, and aggregates
# the results into a table and a few comparison plots.
#
# random, math, numpy, and matplotlib.pyplot are used for the actual
# JSSP/GA logic, plus the standard library's os and time modules for file
# handling and timing.

import os
import time
import math
import matplotlib.pyplot as plt

from util.LoadConfig import (
    DATA_DIR,
    INSTANCE_CATEGORIES,
    RESULTS_DIR,
    NUMBER_OF_RUNS,
    PARAMETER_SETS,
    EXPERIMENT_RESULTS_CSV,
    CONVERGENCE_COMPARISON_FILENAME,
    CATEGORY_COMPARISON_FILENAME
)

from decoder import parse_instance, decode
from ga_core import run_ga


# ---------------------------------------------------------------------------
# 1. Running one (instance, parameter setting) combination several times
# ---------------------------------------------------------------------------

def run_one_combination(instance, parameter_setting, number_of_runs, progress_label):
    """
    Runs the GA `number_of_runs` independent times on the same instance with
    the same parameter setting. Returns a dictionary with the aggregated
    statistics the report needs, plus one representative convergence curve
    (the best-per-generation list from the FIRST run only, so we don't have
    to store every run's full generation history).
    """
    makespans = []
    execution_times = []
    convergence_generations = []
    first_run_best_per_generation = None

    run_number = 0
    while run_number < number_of_runs:
        print(progress_label, "- run", run_number + 1, "of", number_of_runs, "...")

        start_time = time.time()

        result = run_ga(
            instance,
            decode,
            population_size=parameter_setting["population_size"],
            num_generations=parameter_setting["num_generations"],
            crossover_rate=parameter_setting["crossover_rate"],
            mutation_rate=parameter_setting["mutation_rate"],
            patience=parameter_setting["patience"]
        )

        end_time = time.time()
        elapsed_time = end_time - start_time

        makespans.append(result["best_makespan"])
        execution_times.append(elapsed_time)
        convergence_generations.append(result["convergence_generation"])

        if run_number == 0:
            first_run_best_per_generation = result["best_per_generation"]

        run_number = run_number + 1

    # --- Aggregate statistics across all runs of this combination -----------
    best_makespan = min(makespans)
    worst_makespan = max(makespans)

    total_makespan = 0
    for value in makespans:
        total_makespan = total_makespan + value
    average_makespan = total_makespan / len(makespans)

    variance_total = 0
    for value in makespans:
        variance_total = variance_total + (value - average_makespan) ** 2
    std_dev_makespan = math.sqrt(variance_total / len(makespans))

    total_time = 0
    for value in execution_times:
        total_time = total_time + value
    average_execution_time = total_time / len(execution_times)

    total_convergence = 0
    for value in convergence_generations:
        total_convergence = total_convergence + value
    average_convergence_generation = total_convergence / len(convergence_generations)

    combination_result = {
        "best_makespan": best_makespan,
        "worst_makespan": worst_makespan,
        "average_makespan": average_makespan,
        "std_dev_makespan": std_dev_makespan,
        "average_execution_time": average_execution_time,
        "total_execution_time": total_time,
        "average_convergence_generation": average_convergence_generation,
        "best_per_generation_first_run": first_run_best_per_generation
    }
    return combination_result


# ---------------------------------------------------------------------------
# 2. Running the full sweep: every category x every instance x every
#    parameter setting
# ---------------------------------------------------------------------------

def run_experiment(instance_files_by_category, parameter_settings, number_of_runs):
    """
    Runs the full experiment sweep and returns a list of dictionaries, one
    per (instance, parameter setting) combination. Each dictionary is one
    row of the final results table, tagged with its size category so
    small/medium/large can be compared directly later.
    """
    all_results = []

    for category in instance_files_by_category:
        instance_filenames = instance_files_by_category[category]

        for instance_filename in instance_filenames:
            instance_path = os.path.join(DATA_DIR, instance_filename)
            instance = parse_instance(instance_path)

            parameter_index = 0
            while parameter_index < len(parameter_settings):
                parameter_setting = parameter_settings[parameter_index]

                progress_label = (
                    instance_filename + " [" + category + "], "
                    "parameter set " + parameter_setting["name"]
                )

                combination_result = run_one_combination(
                    instance, parameter_setting, number_of_runs, progress_label
                )

                row = {
                    "category": category,
                    "instance": instance_filename,
                    "parameter_set": parameter_setting["name"],
                    "best_makespan": combination_result["best_makespan"],
                    "worst_makespan": combination_result["worst_makespan"],
                    "average_makespan": combination_result["average_makespan"],
                    "std_dev_makespan": combination_result["std_dev_makespan"],
                    "average_execution_time": combination_result["average_execution_time"],
                    "total_execution_time": combination_result["total_execution_time"],
                    "average_convergence_generation": combination_result["average_convergence_generation"],
                    "best_per_generation_first_run": combination_result["best_per_generation_first_run"]
                }

                all_results.append(row)

                parameter_index = parameter_index + 1

    return all_results


# ---------------------------------------------------------------------------
# 3. Saving results to a CSV file by hand
# ---------------------------------------------------------------------------

def save_results_to_csv(results, filepath):
    """
    Writes the list of result dictionaries to a plain CSV file: one header
    line with column names, then one line per result row. Note that the
    per-generation convergence curve is left out of the CSV on purpose —
    it's a list of numbers, not a single value, so it doesn't belong in a
    table row. Use plot_convergence / plot_convergence_comparison to turn
    that data into a chart instead.
    """
    column_names = [
        "category",
        "instance",
        "parameter_set",
        "best_makespan",
        "worst_makespan",
        "average_makespan",
        "std_dev_makespan",
        "average_execution_time",
        "total_execution_time",
        "average_convergence_generation"
    ]

    file = open(filepath, "w")

    # Write the header line.
    header_line = ""
    column_index = 0
    while column_index < len(column_names):
        header_line = header_line + column_names[column_index]
        if column_index < len(column_names) - 1:
            header_line = header_line + ","
        column_index = column_index + 1
    file.write(header_line + "\n")

    # Write one line per result row.
    for row in results:
        line = ""
        column_index = 0
        while column_index < len(column_names):
            column_name = column_names[column_index]
            line = line + str(row[column_name])
            if column_index < len(column_names) - 1:
                line = line + ","
            column_index = column_index + 1
        file.write(line + "\n")

    file.close()


def print_results_table(results):
    """
    Prints the same results in a readable, lined-up form in the console.
    """
    print(
        "Category   | Instance     | Parameter Set              | "
        "Best  | Worst  | Avg     | StdDev | AvgTime(s) | TotalTime(s) | AvgConvGen"
    )
    for row in results:
        print(
            row["category"], "|",
            row["instance"], "|",
            row["parameter_set"], "|",
            row["best_makespan"], "|",
            row["worst_makespan"], "|",
            round(row["average_makespan"], 2), "|",
            round(row["std_dev_makespan"], 2), "|",
            round(row["average_execution_time"], 3), "|",
            round(row["total_execution_time"], 3), "|",
            round(row["average_convergence_generation"], 1)
        )


def get_convergence_curves_for_instance(results, instance_filename):
    """
    Pulls out the best-per-generation curve and the parameter set name for
    every row in `results` that belongs to the given instance file. Used to
    build a convergence comparison chart (one curve per parameter set) for
    ONE specific instance, from the real experiment results.

    Returns (list_of_curves, list_of_labels).
    """
    curves = []
    labels = []

    for row in results:
        if row["instance"] == instance_filename:
            curves.append(row["best_per_generation_first_run"])
            labels.append(row["parameter_set"])

    return curves, labels


def get_average_makespan_by_category(results, parameter_set_name):
    """
    For ONE parameter set (so the comparison is fair, not mixing
    different parameter settings together), computes the average makespan
    for each category ("small", "medium", "large") by averaging over the
    two instances in that category. Used to build the category comparison
    bar chart from the real experiment results.

    Returns (list_of_categories, list_of_average_makespans).
    """
    totals_by_category = {}
    counts_by_category = {}

    for row in results:
        if row["parameter_set"] == parameter_set_name:
            category = row["category"]
            if category not in totals_by_category:
                totals_by_category[category] = 0
                counts_by_category[category] = 0
            totals_by_category[category] = totals_by_category[category] + row["average_makespan"]
            counts_by_category[category] = counts_by_category[category] + 1

    categories = []
    averages = []
    for category in totals_by_category:
        categories.append(category)
        averages.append(totals_by_category[category] / counts_by_category[category])

    return categories, averages


# ---------------------------------------------------------------------------
# 4. Plotting functions — all save an image file in addition to showing it
# ---------------------------------------------------------------------------

def plot_convergence(best_per_generation_list, title, save_path=None):
    """
    Draws one best-fitness-per-generation curve.
    """
    generation_numbers = []
    generation_index = 0
    while generation_index < len(best_per_generation_list):
        generation_numbers.append(generation_index)
        generation_index = generation_index + 1

    plt.figure(figsize=(8, 5))
    plt.plot(generation_numbers, best_per_generation_list)
    plt.xlabel("Generation")
    plt.ylabel("Best makespan")
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
    plt.close()


def plot_convergence_comparison(list_of_best_per_generation_lists, labels, title, save_path=None):
    """
    Draws several best-fitness-per-generation curves on the SAME chart, one
    line per parameter setting, with a legend.
    """
    plt.figure(figsize=(8, 5))

    curve_index = 0
    while curve_index < len(list_of_best_per_generation_lists):
        curve = list_of_best_per_generation_lists[curve_index]

        generation_numbers = []
        generation_index = 0
        while generation_index < len(curve):
            generation_numbers.append(generation_index)
            generation_index = generation_index + 1

        plt.plot(generation_numbers, curve, label=labels[curve_index])
        curve_index = curve_index + 1

    plt.xlabel("Generation")
    plt.ylabel("Best makespan")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
    plt.close()


def plot_category_comparison(categories, values, ylabel, title, save_path=None):
    """
    Draws a simple bar chart, one bar per category (e.g. "small", "medium",
    "large").
    """
    x_positions = []
    index = 0
    while index < len(categories):
        x_positions.append(index)
        index = index + 1

    plt.figure(figsize=(6, 5))
    plt.bar(x_positions, values)
    plt.xticks(x_positions, categories)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# 5. Demo / manual test — run this file directly to check everything works
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    os.makedirs(RESULTS_DIR, exist_ok=True)

    """
    --- Quick pipeline check (COMMENTED OUT, this was only needed once,
    to verify decoder.py + ga_core.py work together, before running the
    full sweep below every single time). Delete the two triple-quote lines
    around this block if you want to re-run the quick sanity check.

    demo_instance_filename = INSTANCE_CATEGORIES["small"][0]
    demo_instance_path = os.path.join(DATA_DIR, demo_instance_filename)
    demo_instance = parse_instance(demo_instance_path)

    demo_number_of_runs = 3
    demo_curves = []
    demo_labels = []

    print("=== Quick pipeline check (demo) ===")

    parameter_index = 0
    while parameter_index < len(PARAMETER_SETS):
        parameter_setting = PARAMETER_SETS[parameter_index]
        progress_label = demo_instance_filename + " [demo], parameter set " + parameter_setting["name"]

        combination_result = run_one_combination(
            demo_instance, parameter_setting, demo_number_of_runs, progress_label
        )

        print(
            parameter_setting["name"],
            "- best:", combination_result["best_makespan"],
            "worst:", combination_result["worst_makespan"],
            "avg:", round(combination_result["average_makespan"], 2)
        )

        demo_curves.append(combination_result["best_per_generation_first_run"])
        demo_labels.append(parameter_setting["name"])

        parameter_index = parameter_index + 1

    demo_convergence_path = os.path.join(RESULTS_DIR, CONVERGENCE_COMPARISON_FILENAME)
    plot_convergence_comparison(
        demo_curves,
        demo_labels,
        "Convergence comparison across parameter sets (demo)",
        save_path=demo_convergence_path
    )
    print("Saved:", demo_convergence_path)

    demo_category_path = os.path.join(RESULTS_DIR, CATEGORY_COMPARISON_FILENAME)
    plot_category_comparison(
        ["small", "medium", "large"],
        [700, 1200, 2500],
        "Average makespan",
        "Average makespan by category (example numbers)",
        save_path=demo_category_path
    )
    print("Saved:", demo_category_path)
    """

    # --- The full experiment sweep -------------------------------------------
    # This runs every instance x every parameter set x NUMBER_OF_RUNS
    # independent runs, and can take a few minutes depending on the values
    # chosen in config.yaml.
    results = run_experiment(INSTANCE_CATEGORIES, PARAMETER_SETS, NUMBER_OF_RUNS)
    csv_path = os.path.join(RESULTS_DIR, EXPERIMENT_RESULTS_CSV)
    save_results_to_csv(results, csv_path)
    print_results_table(results)
    print("Saved:", csv_path)

    # --- Convergence comparison chart, from REAL results ---------------------
    # Picks one representative instance (the first large instance, since
    # that is where the three parameter sets showed the clearest difference
    # in our own test run) and plots all three parameter sets' convergence
    # curves on it.
    representative_instance = INSTANCE_CATEGORIES["large"][0]
    curves, labels = get_convergence_curves_for_instance(results, representative_instance)

    convergence_comparison_path = os.path.join(RESULTS_DIR, CONVERGENCE_COMPARISON_FILENAME)
    plot_convergence_comparison(
        curves,
        labels,
        "Convergence comparison across parameter sets (" + representative_instance + ")",
        save_path=convergence_comparison_path
    )
    print("Saved:", convergence_comparison_path)

    # --- Category comparison chart, from REAL results -------------------------
    # Uses only the baseline parameter set (Set 1) so the comparison is a
    # fair one: the only thing changing across the bars is problem size, not
    # also the GA parameters.
    baseline_parameter_set_name = PARAMETER_SETS[0]["name"]
    categories, averages = get_average_makespan_by_category(results, baseline_parameter_set_name)

    category_comparison_path = os.path.join(RESULTS_DIR, CATEGORY_COMPARISON_FILENAME)
    plot_category_comparison(
        categories,
        averages,
        "Average makespan",
        "Average makespan by category (" + baseline_parameter_set_name + ")",
        save_path=category_comparison_path
    )
    print("Saved:", category_comparison_path)