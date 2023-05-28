#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
Loads Blood-brain-barrier-permeability data with SMILES strings and converts them to numeric embedding fingerprints with ChemBERTa-2 model loaded from huggingface.

Author: Pranjal Dhole
E-mail: dhole.pranjal@gmail.com
'''

from os.path import join
import argparse
import numpy as np
import pandas as pd
import h5py
from pandas import HDFStore

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from molecule_gym.config import DATA_DIR

import warnings
warnings.filterwarnings('ignore')

def main(modelname):
    '''
    Runs the main script to generate numeric fingerprint with ChemBERTa-2 from SMILES strings
    and save the fingerprints in HDF5 file.
    '''
    # load data
    df = pd.read_csv(join(DATA_DIR, 'BBBP.csv'))

    if modelname == 'chemberta':
        # ChemBERTa
        tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        model = AutoModelForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    elif modelname == 'chemberta2':
        # ChemBERTa-2
        tokenizer = AutoTokenizer.from_pretrained("jonghyunlee/DrugLikeMolecule_chemBERTa_2")
        model = AutoModelForMaskedLM.from_pretrained("jonghyunlee/DrugLikeMolecule_chemBERTa_2")
    else:
        raise AssertionError(f'{modelname} loading is not implemented!')

    # ### Tokenization
    # We break the SMILES up into smaller chunks in the format compatible with ChemBERTa.
    tokenized = df['smiles'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))


    # ### Padding
    # After tokenization, `tokenized` is a list of SMILES chunks -- each chunk is represented as a list of tokens.
    # We want ChemBERTa to process our examples all at once (as one batch).
    # For that reason, we need to pad all lists to the same size, so we can represent the input as one 2-d array, rather than a list of lists (of different lengths).
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

    # Attention Mask
    # We need to create another variable to tell ChemBERTa to ignore (mask) the padding added when it's processing its input.
    attention_mask = np.where(padded != 0, 1, 0)
    attention_mask.shape

    # Generate SMILES embeddings
    print('Converting SMILES strings to ChemBERTa model embeddings...')
    input_ids = torch.tensor(padded)  
    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
    print('done!')
    

    # Let's slice only the part of the output that we need.
    # The embeddings that we use as numeric features correspond to the first token of each SMILES embedding.
    # The way ChemBERTa does sentence classification, is that it adds a token called `[CLS]` (for classification) at the beginning of every SMILE string.
    # The output corresponding to that token can be thought of as an embedding for the entire SMILES string.
    features = last_hidden_states[0][:,0,:].numpy()

    # Save data to HDF5 file
    print('Saving data to HDF5 file...')
    file_name = join(DATA_DIR, f'{modelname}_blood_brain_barrier_permeability.h5')
    hdf = HDFStore(file_name, mode='a')
    hdf.put('bbbp_raw', df, format='table', data_columns=True)
    hdf.close()

    f = h5py.File(file_name, 'a')
    f.create_dataset(f"{modelname}_feature_embeddings", data=features)
    f.close()
    print('done!')

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-m', '--modelname',
                        choices=["chemberta", "chemberta2"],
                        help='Choose a model to generate embeddings from SMILES strings',
                        default="chemberta2")
    ARGS = PARSER.parse_args()

    main(ARGS.modelname)
