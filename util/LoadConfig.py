import yaml
from pathlib import Path

# Path to this script's directory
script_dir = Path(__file__).resolve().parent

# Go up one level to project root, then find config.yaml
config_path = script_dir.parent / "config.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Test
test = config["test"]["test"]
print(test)

# --- Dataset settings -----------------------------------------------------
# Folder where the JSPLib instance files (la01.txt, la03.txt, ...) live.
DATA_DIR = config["dataset"]["data_dir"]

# Which instance files belong to which problem-size category, e.g.
# {"small": ["la01.txt", "la03.txt"], "medium": [...], "large": [...]}
INSTANCE_CATEGORIES = config["dataset"]["categories"]

# --- Decoder settings -------------------------------------------------------
# Which Schedule Building Algorithm strategy decode() uses
# (e.g. "semi_active" or "active").
DECODER_STRATEGY = config["decoder"]["strategy"]

# --- GA settings -------------------------------------------------------------
# Number of chromosomes that compete in each tournament inside
# ga_core.py's tournament_selection(). Stays fixed across all three
# parameter sets used in the experiments.
TOURNAMENT_SIZE = config["ga"]["tournament_size"]

# --- Results/output settings ------------------------------------------------
# Folder where generated files (Gantt charts, CSVs, plots) get saved.
RESULTS_DIR = config["results"]["results_dir"]

# File extension used when saving images (e.g. "png", "jpg").
IMAGE_FORMAT = config["results"]["image_format"]

# Filenames for the example Gantt charts produced by decoder.py's demo.
EXAMPLE_GANTT_FILENAME = config["results"]["example_gantt_filename"]
EXAMPLE_INSTANCE_GANTT_FILENAME = config["results"]["example_instance_gantt_filename"]