from pathlib import Path
import json

from setuptools import setup, find_packages

setup(
    name="rdm_wach_ai",
    version="0.1",
    packages=find_packages(where="src"),  # <-- find packages inside src
    package_dir={"": "src"},              # <-- tell Python root is src
    install_requires=[
        "pandas",
        "influxdb-client",
        "python-dotenv",
        "pyarrow",
        "python-dateutil"
    ],
)

# -------------------------
# Project Root Directory
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# -------------------------
# Folder Structure
# -------------------------
folders = [
    "src/rdm_wach_ai/extract",
    "src/rdm_wach_ai/clean",
    "src/rdm_wach_ai/preprocess",
    "src/rdm_wach_ai/models",
    "src/rdm_wach_ai/utils",
    "src/rdm_wach_ai/config",
    "data/raw",
    "data/cleaned",
    "logs/extraction",
    "logs/cleaning",
    "logs/model_training",
    "tests"
]

# -------------------------
# Files to Create
# -------------------------
files = [
    "src/rdm_wach_ai/__init__.py",
    "src/rdm_wach_ai/extract/__init__.py",
    "src/rdm_wach_ai/extract/extract.py",
    "src/rdm_wach_ai/clean/__init__.py",
    "src/rdm_wach_ai/clean/cleaning.py",
    "src/rdm_wach_ai/preprocess/__init__.py",
    "src/rdm_wach_ai/models/__init__.py",
    "src/rdm_wach_ai/models/train_model.py",
    "src/rdm_wach_ai/utils/__init__.py",
    "src/rdm_wach_ai/utils/config.py",
    "requirements.txt",
    "setup.py",
    "README.md",
    ".gitignore"
]

# -------------------------
# Create Folders
# -------------------------
for folder in folders:
    path = PROJECT_ROOT / folder
    path.mkdir(parents=True, exist_ok=True)
    print(f" Folder ensured: {folder}")

# -------------------------
# Create Empty Files
# -------------------------
for file in files:
    file_path = PROJECT_ROOT / file
    if not file_path.exists():
        file_path.touch()
        print(f" File created: {file}")
    else:
        print(f" File already exists: {file}")

# -------------------------
# Create Default fields.json
# -------------------------
fields_json_path = PROJECT_ROOT / "src/rdm_wach_ai/config/fields.json"

if not fields_json_path.exists():
    default_fields = {
        "energy_fields": [
            "current_l1",
            "current_l2",
            "current_l3",
            "power_factor_avg",
            "power_total",
            "volts_l1_n",
            "volts_l2_n",
            "volts_l3_n"
        ]
    }
    with open(fields_json_path, "w") as f:
        json.dump(default_fields, f, indent=4)
    print("✔ fields.json created")

# -------------------------
# Create Default .env
# -------------------------
env_path = PROJECT_ROOT / "src/rdm_wach_ai/config/.env"

if not env_path.exists():
    env_content = """URL=http://178.128.53.199:8086
TOKEN=YOUR_TOKEN_HERE
ORG=wach
BUCKET=wach_bucket_3
"""
    env_path.write_text(env_content)
    print(" .env created")

print("\n Project structure setup complete!")