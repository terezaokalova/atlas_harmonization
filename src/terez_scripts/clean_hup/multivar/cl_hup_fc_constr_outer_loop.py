#!/usr/bin/env python3
# name: cl_hup_fc_constr_outer_loop.py
# this script loops over each subject and each epoch in the 'clean' directory.
# for each epoch, it calls a separate connectivity processing script via subprocess.run.
# by running each epoch in a separate process, memory is released after the process ends,
# preventing the kernel from crashing.
import os
import subprocess

import sys
sys.path.insert(0, "/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/external_dependencies/CNT_research_tools/python/CNTtools")
sys.path.insert(0, "/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization")

from config.config import BASE_PATH
from config.config import CLEAN_PATH
base_path = BASE_PATH
clean_path = CLEAN_PATH
# clean_path = '/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/Data/hup/derivatives/clean'

subject_dirs = sorted([d for d in os.listdir(clean_path)
                       if os.path.isdir(os.path.join(clean_path, d)) and d.startswith("sub-")])

# iterate over the 20 epochs
for subject in subject_dirs:
    subject_path = os.path.join(clean_path, subject)
    print(f"processing subject: {subject}")
    for epoch_idx in range(20):
        # build a command that calls the epoch processing script,
        # passing the subject folder and epoch index as arguments.
        # cmd = f"cl_hup_fc_constr_single_epoch.py {subject} {epoch_idx}"
        cmd = f"python cl_hup_fc_constr_single_epoch.py {subject} {epoch_idx}"
        print(f"  running command: {cmd}")
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"error processing {subject} epoch {epoch_idx}: {e}")
