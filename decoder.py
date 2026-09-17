import os
import random
import matplotlib.pyplot as plt

from util.LoadConfig import (
    DATA_DIR,
    RESULTS_DIR,
    EXAMPLE_GANTT_FILENAME,
    EXAMPLE_INSTANCE_GANTT_FILENAME
)


# ---------------------------------------------------------------------------
# 1. Reading a JSPLib instance file
# ---------------------------------------------------------------------------

def parse_instance(filepath):
    """
    Reads a JSPLib-format job shop instance file and returns a dictionary
    describing the jobs, machines, and processing times.

    File format (see data/ folder):
        line 1:  number_of_jobs number_of_machines
        line 2+: one line per job, listing pairs of
                 (machine_number, processing_time), in the order that job
                 must visit them.

    Returns a dictionary like:
        {
            "num_jobs": 10,
            "num_machines": 5,
            "jobs": {
                0: [(machine, time), (machine, time), ...],   # job 0's operations
                1: [(machine, time), (machine, time), ...],   # job 1's operations
                ...
            }
        }
    """
    file = open(filepath, "r")
    lines = file.readlines()
    file.close()

    # The first line has two numbers: number of jobs and number of machines.
    first_line = lines[0].split()
    num_jobs = int(first_line[0])
    num_machines = int(first_line[1])

    jobs = {}

    # Each of the next num_jobs lines describes one job's operations.
    job_number = 0
    while job_number < num_jobs:
        line = lines[job_number + 1].split()

        operations = []
        position = 0
        while position < len(line):
            machine_number = int(line[position])
            processing_time = int(line[position + 1])
            operations.append((machine_number, processing_time))
            position = position + 2

        jobs[job_number] = operations
        job_number = job_number + 1

    instance = {
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "jobs": jobs
    }

    return instance


# ---------------------------------------------------------------------------
# 2. Decoding a chromosome into a schedule (Schedule Building Algorithm)
# ---------------------------------------------------------------------------

def decode(chromosome, instance):
    """
    Turns a chromosome into an actual schedule using a SEMI-ACTIVE Schedule
    Building Algorithm (SBA).

    Chromosome meaning: a list of job numbers, where each job number appears
    once for every operation that job has. Reading the chromosome left to
    right, the k-th time a job's number appears refers to that job's k-th
    operation (0-indexed), in the order listed in the instance file.

    "Semi-active" means: every operation starts as early as the two rules
    below allow, but operations are always placed one after another in the
    order given by the chromosome, never squeezed into an earlier gap on a
    machine (that would be the "active" strategy instead).

    Two rules every operation must respect:
      1. Precedence: an operation cannot start before the previous
         operation of the SAME JOB has finished.
      2. Machine capacity: an operation cannot start before the machine
         it needs is free (a machine can only process one operation at a
         time).

    Returns (schedule, makespan):
        schedule: a list of dictionaries, one per operation, each with keys
                  "job", "op_index", "machine", "start", "end"
        makespan: a single number, the time the last operation finishes
                  (this is also the fitness value the GA tries to minimize, lower makespan means a better, shorter schedule)
    """
    num_machines = instance["num_machines"]
    jobs = instance["jobs"]

    # Step (a): keep track of when each machine becomes free next.
    # All machines start free at time 0.
    machine_free_at = []
    machine_index = 0
    while machine_index < num_machines:
        machine_free_at.append(0)
        machine_index = machine_index + 1

    # Step (b): keep track of how far each job has progressed.
    # "job_finished_at" = the time the job's most recently scheduled
    # operation finished. "job_next_op" = which operation number (0, 1, 2...)
    # is next for that job.
    job_finished_at = {}
    job_next_op = {}
    for job_number in jobs:
        job_finished_at[job_number] = 0
        job_next_op[job_number] = 0

    schedule = []

    # Step (c): go through the chromosome left to right, one job number at
    # a time, and schedule that job's next operation.
    for job_number in chromosome:

        op_index = job_next_op[job_number]
        machine_number, processing_time = jobs[job_number][op_index]

        # The operation cannot start before (i) the machine is free, and
        # (ii) the previous operation of the same job is finished.
        # So its start time is the LATER of those two times.
        earliest_machine_time = machine_free_at[machine_number]
        earliest_job_time = job_finished_at[job_number]

        if earliest_machine_time > earliest_job_time:
            start_time = earliest_machine_time
        else:
            start_time = earliest_job_time

        end_time = start_time + processing_time

        # Update the machine and job trackers for the next operation.
        machine_free_at[machine_number] = end_time
        job_finished_at[job_number] = end_time
        job_next_op[job_number] = op_index + 1

        # Save this operation's details.
        operation_record = {
            "job": job_number,
            "op_index": op_index,
            "machine": machine_number,
            "start": start_time,
            "end": end_time
        }
        schedule.append(operation_record)

    # Step (d): the makespan is the largest end time across every operation.
    makespan = 0
    for operation_record in schedule:
        if operation_record["end"] > makespan:
            makespan = operation_record["end"]

    return schedule, makespan


# ---------------------------------------------------------------------------
# 3. Drawing a Gantt chart from a decoded schedule
# ---------------------------------------------------------------------------

def plot_gantt(schedule, title, save_path=None):
    """
    Draws a simple Gantt chart from a decoded schedule: one horizontal row
    per machine, one colored bar per operation, labeled with the job number.

    If save_path is given (e.g. "results/example_gantt.png"), the chart is
    also saved to that exact file. Matplotlib figures out whether to save a
    PNG, JPG, or JPEG automatically from the file extension in save_path.
    """
    # A small fixed list of colors, one per job. If there are more jobs
    # than colors, we just start reusing colors from the start of the list.
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
              "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

    figure, axis = plt.subplots(figsize=(10, 5))

    for operation_record in schedule:
        job_number = operation_record["job"]
        machine_number = operation_record["machine"]
        start_time = operation_record["start"]
        end_time = operation_record["end"]
        duration = end_time - start_time

        color_index = job_number % len(colors)
        bar_color = colors[color_index]

        axis.barh(
            machine_number,
            duration,
            left=start_time,
            color=bar_color,
            edgecolor="black"
        )

        # Label the bar with the job number, centered on the bar.
        label_x = start_time + duration / 2
        axis.text(
            label_x,
            machine_number,
            "J" + str(job_number),
            ha="center",
            va="center",
            color="white",
            fontsize=8
        )

    axis.set_xlabel("Time")
    axis.set_ylabel("Machine")
    axis.set_title(title)

    # Machine axis ticks: one tick per machine number that actually appears.
    machine_numbers = []
    for operation_record in schedule:
        if operation_record["machine"] not in machine_numbers:
            machine_numbers.append(operation_record["machine"])
    machine_numbers.sort()
    axis.set_yticks(machine_numbers)

    # Build a simple legend: one colored patch per job number that appears.
    job_numbers = []
    for operation_record in schedule:
        if operation_record["job"] not in job_numbers:
            job_numbers.append(operation_record["job"])
    job_numbers.sort()

    legend_handles = []
    for job_number in job_numbers:
        color_index = job_number % len(colors)
        patch = plt.Rectangle((0, 0), 1, 1, color=colors[color_index])
        legend_handles.append(patch)

    legend_labels = []
    for job_number in job_numbers:
        legend_labels.append("Job " + str(job_number))

    axis.legend(legend_handles, legend_labels, loc="upper right", fontsize=8)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# 4. A small helper used only by the demo below: build ONE random valid
#    chromosome for a given instance. (The real GA population/selection/
#    crossover/mutation code lives in a teammate's ga_core.py, this is
#    just enough to test decode() and plot_gantt() on their own.)
# ---------------------------------------------------------------------------

def create_random_chromosome(instance):
    """
    Builds one random valid chromosome: a list of job numbers where each
    job number appears once for every operation that job has.
    """
    chromosome = []

    for job_number in instance["jobs"]:
        num_operations = len(instance["jobs"][job_number])
        count = 0
        while count < num_operations:
            chromosome.append(job_number)
            count = count + 1

    random.shuffle(chromosome)
    return chromosome


# ---------------------------------------------------------------------------
# 5. Demo / manual test — run this file directly to check everything works
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Make sure the results folder exists before we try to save anything into it.
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Part 1: a tiny, hand-built 2-job, 2-machine example -----------------
    # This is the small illustrative example for the report: easy to check
    # by hand and easy to show as a schedule table + Gantt chart.
    #
    #   Job 0: operation 0 on machine 0 (3 time units), then
    #          operation 1 on machine 1 (2 time units)
    #   Job 1: operation 0 on machine 1 (2 time units), then
    #          operation 1 on machine 0 (2 time units)
    toy_instance = {
        "num_jobs": 2,
        "num_machines": 2,
        "jobs": {
            0: [(0, 3), (1, 2)],
            1: [(1, 2), (0, 2)]
        }
    }

    toy_chromosome = [0, 1, 0, 1]

    print("=== Tiny example ===")
    print("Chromosome:", toy_chromosome)

    toy_schedule, toy_makespan = decode(toy_chromosome, toy_instance)

    print("Job | Op | Machine | Start | End")
    for operation_record in toy_schedule:
        print(
            operation_record["job"], "  ",
            operation_record["op_index"], "  ",
            operation_record["machine"], "      ",
            operation_record["start"], "    ",
            operation_record["end"]
        )
    print("Makespan:", toy_makespan)

    example_gantt_path = os.path.join(RESULTS_DIR, EXAMPLE_GANTT_FILENAME)
    plot_gantt(toy_schedule, "Tiny example schedule", save_path=example_gantt_path)
    print("Saved:", example_gantt_path)

    # --- Part 2: a real benchmark instance (la01) ----------------------------
    # This checks that parse_instance() and decode() also work correctly on
    # an actual JSPLib file, not just the hand-built toy example above.
    data_dir = DATA_DIR
    la01_path = os.path.join(data_dir, "la01.txt")

    print()
    print("=== Real instance example (la01) ===")

    la01_instance = parse_instance(la01_path)
    print("Number of jobs:", la01_instance["num_jobs"])
    print("Number of machines:", la01_instance["num_machines"])

    la01_chromosome = create_random_chromosome(la01_instance)
    la01_schedule, la01_makespan = decode(la01_chromosome, la01_instance)

    print("Makespan for this random chromosome:", la01_makespan)

    la01_gantt_path = os.path.join(RESULTS_DIR, EXAMPLE_INSTANCE_GANTT_FILENAME)
    plot_gantt(la01_schedule, "la01 example schedule", save_path=la01_gantt_path)
    print("Saved:", la01_gantt_path)