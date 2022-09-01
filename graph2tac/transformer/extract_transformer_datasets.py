# Authors: Jelle Piepenbrock
# 2022

from pathlib import Path
from graph2tac.loader.data_server import DataServer
from graph2tac.common import uuid
import numpy as np
import os
import tqdm
import random
import sys
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--raw_dataset_location", "-r", help="Location of the dataset we want to process", type=str)
parser.add_argument("--dataset_stem", "-s", help="The location and name of the created output text files", type=str)
parser.add_argument("--tactic_form", "-t", help="Which of the three tactic text forms to take", type=int)

args = parser.parse_args()
raw_dataset_location = args.raw_dataset_location
dataset_stem = args.dataset_stem
tactic_form = int(args.tactic_form)

export_file_train = f"{dataset_stem}_train"
export_file_val = f"{dataset_stem}_val"


data_dirs = [p.expanduser() for p in [

    Path(raw_dataset_location)
            ]]

data_dir = data_dirs[0]
#d = DataServer(data_dir=data_dir, encode_all=True)
d = DataServer(data_dir=data_dir, max_subgraph_size=10)

with open(export_file_train, "w") as f1:
    all_train_samples = [k for k in enumerate(d.data_train(as_text=True))]
    random.shuffle(all_train_samples)
    for e, sam in all_train_samples:

        f1.write(sam[0].decode("utf-8") + " OUTPUT " + sam[1][tactic_form].decode("utf-8") + " <END> \n")


with open(export_file_val, "w") as f2:

    all_val_samples = [k for k in enumerate(d.data_valid(as_text=True))]
    random.shuffle(all_val_samples)
    for e, sam in all_val_samples:

        f2.write(sam[0].decode("utf-8") + " OUTPUT " + sam[1][tactic_form].decode("utf-8") + " <END> \n")

