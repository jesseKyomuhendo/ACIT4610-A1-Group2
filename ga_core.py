# Genetic Algorithm core for the Job Shop Scheduling Problem (ACIT 4610).
#
# This file only contains the GA itself. Turning a chromosome into a schedule
# (decoding) is done by a separate decode_function that is passed into run_ga.
#
# ---------------------------------------------------------------------------
# REPRESENTATION
# ---------------------------------------------------------------------------
# A chromosome is a list of job numbers. Each job number appears once for
# every operation that job has. Reading left to right, the k-th time a job's
# number appears means "this job's k-th operation".
#
# Example: [0, 1, 0, 1] means: job 0 op 1, job 1 op 1, job 0 op 2, job 1 op 2.
#
# Assumed instance format: instance is the dictionary produced by decoder.py's
# parse_instance(), e.g.:
#   {"num_jobs": 10, "num_machines": 5, "jobs": {0: [(machine, time), ...], ...}}
# so instance["jobs"][j] is the list of operations of job j, and
# len(instance["jobs"][j]) is the number of operations job j has. Only
# create_chromosome looks inside the instance.
#
# ---------------------------------------------------------------------------
# FITNESS FUNCTION (OBJECTIVE)
# ---------------------------------------------------------------------------
# The fitness of a chromosome is its MAKESPAN. decode_function(chromosome,
# instance) is expected to match decoder.py's decode(): it returns a TUPLE
# (schedule, makespan), not a single number, so run_ga always unpacks it and
# uses only the makespan (the second value) as the fitness.
#
# LOWER IS BETTER. A shorter makespan means the whole schedule finishes sooner.
#
# Therefore run_ga is MINIMIZING the fitness, not maximizing it. For exactly
# this reason tournament_selection picks the chromosome with the LOWEST
# makespan out of its random group, and the "best" chromosome everywhere in
# this file is the one with the lowest makespan.
# ---------------------------------------------------------------------------

import random

from util.LoadConfig import TOURNAMENT_SIZE


# Builds one random valid chromosome where each job number appears once per operation of that job.
def create_chromosome(instance):
    chromosome = []
    for job in instance["jobs"]:
        number_of_operations = len(instance["jobs"][job])
        for operation in range(number_of_operations):
            chromosome.append(job)
    random.shuffle(chromosome)
    return chromosome


# Creates a starting population as a plain list of random chromosomes.
def create_population(instance, population_size):
    population = []
    for i in range(population_size):
        population.append(create_chromosome(instance))
    return population


# Picks tournament_size random chromosomes and returns the one with the lowest (best) makespan.
#
# Tournament selection is an ordinal-based selection scheme:
#  it only compares which chromosome is better, so it
# works directly with a fitness that must be minimized.
def tournament_selection(population, fitness_values, tournament_size):
    if tournament_size > len(population):
        tournament_size = len(population)

    competitors = random.sample(range(len(population)), tournament_size)

    best_index = competitors[0]
    for index in competitors:
        if fitness_values[index] < fitness_values[best_index]:
            best_index = index

    # Return a copy so later changes to the child never change the population.
    return list(population[best_index])


# Combines two parents into one child with Order Crossover, keeping every job's operation count correct.
#
# METHOD: Order Crossover
#   Step 1: pick two cut points and copy the middle segment from one parent
#           into the child at the same positions.
#   Step 2: go through the other parent from left to right and cross out the
#           genes that are already used by the segment.
#   Step 3: fill the empty positions of the child, left to right, with the
#           genes that are left, in the order they appear in the other parent.
#
# Small adaptation for JSSP: in the TSP slide every city appears once, so
# "already used" means "already in the child". Here a job number appears
# several times, so we COUNT: a job number from the other parent is only
# taken while that job still has operations left to place.
#

#
# TRACE OF THIS ORDER CROSSOVER (same parents, cut points at positions 2 and 3):
#   1) copy parent1[2..3] = [2, 0]      -> child = [_, _, 2, 0, _, _]
#   2) still needed: job 0 x1, job 1 x2, job 2 x1
#   3) walk parent2 [2, 2, 1, 0, 1, 0]: take 2, skip 2, take 1, take 0, take 1, skip 0 -> [2, 1, 0, 1]
#   4) fill the gaps left to right      -> child = [2, 1, 2, 0, 0, 1] (every job twice)
def crossover(parent1, parent2):
    length = len(parent1)

    # Step 1: choose two cut points (start <= end).
    start = random.randint(0, length - 1)
    end = random.randint(0, length - 1)
    if start > end:
        temp = start
        start = end
        end = temp

    # Count how many operations each job has in total.
    remaining = {}
    for gene in parent1:
        if gene in remaining:
            remaining[gene] = remaining[gene] + 1
        else:
            remaining[gene] = 1

    # Copy the middle segment from parent1 and update the counts.
    child = [None] * length
    for i in range(start, end + 1):
        child[i] = parent1[i]
        remaining[parent1[i]] = remaining[parent1[i]] - 1

    # Step 2: take the still-needed genes from parent2, in parent2's order.
    fill_genes = []
    for gene in parent2:
        if gene in remaining and remaining[gene] > 0:
            fill_genes.append(gene)
            remaining[gene] = remaining[gene] - 1

    # Step 3: put them into the empty positions from left to right.
    fill_index = 0
    for i in range(length):
        if child[i] is None:
            child[i] = fill_genes[fill_index]
            fill_index = fill_index + 1

    return child


# Swaps the genes at two random positions and returns the mutated chromosome.
#
# TRACE:
#   chromosome        = [0, 1, 2, 0, 2, 1]
#   random positions  = 1 and 4 (genes 1 and 2)
#   swap them         -> [0, 2, 2, 0, 1, 1]
#   job counts are unchanged: every job still appears twice
def mutate(chromosome):
    mutated = list(chromosome)
    if len(mutated) < 2:
        return mutated

    positions = random.sample(range(len(mutated)), 2)
    first = positions[0]
    second = positions[1]

    temp = mutated[first]
    mutated[first] = mutated[second]
    mutated[second] = temp

    return mutated


# Runs the GA (minimizing makespan) and returns the best solution plus per-generation statistics.
#
# decode_function(chromosome, instance) must return a TUPLE (schedule, makespan)
# -- exactly what decoder.py's decode() returns. run_ga always unpacks it and
# uses only the makespan as the fitness; lower is better, so this loop
# MINIMIZES. Tournament selection picks the lowest makespan in each group.
#
# Generations are numbered from 0 (generation 0 is the starting population),
# so best_per_generation[g] belongs to generation g.
def run_ga(instance, decode_function, population_size, num_generations,
           crossover_rate, mutation_rate, patience):
    population = create_population(instance, population_size)

    best_chromosome = None
    best_fitness = None
    convergence_generation = 0
    generations_without_improvement = 0

    best_per_generation = []
    worst_per_generation = []
    average_per_generation = []

    for generation in range(num_generations):
        # Evaluate every chromosome: decode_function returns (schedule, makespan),
        # and only the makespan is used as the fitness value.
        fitness_values = []
        for chromosome in population:
            schedule, makespan = decode_function(chromosome, instance)
            fitness_values.append(makespan)

        # Statistics for this generation.
        generation_best_index = 0
        generation_worst = fitness_values[0]
        total = 0
        for i in range(len(fitness_values)):
            if fitness_values[i] < fitness_values[generation_best_index]:
                generation_best_index = i
            if fitness_values[i] > generation_worst:
                generation_worst = fitness_values[i]
            total = total + fitness_values[i]

        generation_best = fitness_values[generation_best_index]
        best_per_generation.append(generation_best)
        worst_per_generation.append(generation_worst)
        average_per_generation.append(total / len(fitness_values))

        # Update the best solution found so far (strictly lower = improvement).
        if best_fitness is None or generation_best < best_fitness:
            best_fitness = generation_best
            best_chromosome = list(population[generation_best_index])
            convergence_generation = generation
            generations_without_improvement = 0
        else:
            generations_without_improvement = generations_without_improvement + 1

        # Stop early if there has been no improvement for `patience` generations.
        if generations_without_improvement >= patience:
            break

        # No need to build a new population after the last generation.
        if generation == num_generations - 1:
            break

        # Build the next generation. Elitism: the best chromosome so far is
        # copied in unchanged, so it can never be lost.
        new_population = [list(best_chromosome)]

        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitness_values, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, fitness_values, TOURNAMENT_SIZE)

            if random.random() < crossover_rate:
                child = crossover(parent1, parent2)
            else:
                child = parent1

            if random.random() < mutation_rate:
                child = mutate(child)

            new_population.append(child)

        population = new_population

    result = {}
    result["best_chromosome"] = best_chromosome
    result["best_makespan"] = best_fitness
    result["convergence_generation"] = convergence_generation
    result["best_per_generation"] = best_per_generation
    result["worst_per_generation"] = worst_per_generation
    result["average_per_generation"] = average_per_generation
    return result


if __name__ == "__main__":
    # Tiny made-up instance, in the same dictionary shape decoder.py's
    # parse_instance() produces: instance["jobs"][job] = list of
    # (machine, processing_time) pairs.
    # Job 0 has 3 operations, job 1 has 2, job 2 has 3.
    test_instance = {
        "num_jobs": 3,
        "num_machines": 3,
        "jobs": {
            0: [(0, 3), (1, 2), (2, 2)],
            1: [(0, 2), (2, 1)],
            2: [(1, 4), (2, 3), (0, 1)],
        }
    }

    # Fake decoder: ignores the chromosome and returns a random (schedule, makespan)
    # tuple, matching decoder.py's real decode() return shape.
    def fake_decode_function(chromosome, instance):
        fake_schedule = []
        fake_makespan = random.randint(10, 50)
        return fake_schedule, fake_makespan

    result = run_ga(test_instance, fake_decode_function,
                    population_size=20, num_generations=100,
                    crossover_rate=0.9, mutation_rate=0.2, patience=15)

    print("Best chromosome:", result["best_chromosome"])
    print("Best makespan found:", result["best_makespan"])
    print("Best makespan last improved in generation:", result["convergence_generation"])
    print("Generations run:", len(result["best_per_generation"]))