"""Module to extract contineous data-driven descriptors for a file of SMILES."""
import os
import numpy as np
import sys
import argparse
import pandas as pd
import tensorflow as tf
from inference import InferenceModel
from preprocessing import preprocess_smiles,randomize_smile
from hyperparameters import DEFAULT_DATA_DIR

_default_model_dir = os.path.join(DEFAULT_DATA_DIR, 'default_model')
model_dir = _default_model_dir

infer_model = InferenceModel(model_dir=model_dir,
                                 use_gpu=True,
                                 batch_size=128,
                                 cpu_threads=1)


def encode(sml_list:list) -> np.array:
    """Main function that extracts the contineous data-driven descriptors for a file of SMILES."""
    descriptors = infer_model.seq_to_emb(sml_list)
    return descriptors


def decode(emb:np.array) -> list:
    sml_list = infer_model.emb_to_seq(emb)
    # print(sml)
    return sml_list

