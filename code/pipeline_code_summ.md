Feature extraction part:

# Run with default settings
python main.py

# Force recomputation of all features
python main.py --force

# Process specific cohorts
python main.py --cohorts hup mni

# Use a different config file
python main.py --config custom_config.yaml

GLOBAL ANALYSIS:
Mann-Whitney U uses two-tailed by default (which is what we want)