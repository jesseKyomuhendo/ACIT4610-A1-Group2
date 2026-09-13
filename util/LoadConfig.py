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


#Retrive other configurations