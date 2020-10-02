# I/O
import os
import sys
import joblib
from pathlib import Path
import argparse

# Data handling
import pandas as pd
import numpy as np
from sklearn import model_selection

# Modeling
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

# Custom
from datasets import Dataset, CTRPDataModule
from film_model import FiLMNetwork, ConcatNetwork


def prepare(exp, subset=True):
    data_path = Path("../../film-gex-data/processed/")
    input_cols = joblib.load(data_path.joinpath("gene_cols.pkl"))
    if subset:
        data = pd.read_pickle(data_path.joinpath("train_sub.pkl.gz"))
    else:
        data = pd.read_pickle(data_path.joinpath("train.pkl.gz"))
    
    if exp=='id':
        cpd_id = "broad_cpd_id"
        cond_cols = np.array([cpd_id, 'cpd_conc_umol'])
        data[cpd_id] = data[cpd_id].astype("category").cat.codes
    else:
        fp_cols = joblib.load(data_path.joinpath("fp_cols.pkl"))
        cond_cols = np.append(fp_cols, ['cpd_conc_umol'])
    return data, input_cols, cond_cols


def cv(exp, nfolds, data, input_cols, cond_cols, batch_size):
    seed_everything(2299)
    
    for fold in np.arange(0,nfolds):
        train = data[data['fold']!=fold]
        val = data[data['fold']==fold]
        # DataModule
        dm = CTRPDataModule(train,
                            val,
                            input_cols,
                            cond_cols,
                            target='cpd_avg_pv',
                            batch_size=batch_size)
        # Model
        if exp=='film':
            model = FiLMNetwork(len(input_cols), len(cond_cols))
        else:
            model = ConcatNetwork(len(input_cols), len(cond_cols))
        # Callbacks
        logger = TensorBoardLogger(save_dir=os.getcwd(),
                                   version="{}_fold_{}".format(exp, fold),
                                   name='lightning_logs')
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0.01)
        # Trainer
        trainer = Trainer(max_epochs=50, 
                          gpus=-1,
                          logger=logger,
                          early_stop_callback=early_stop,
                          distributed_backend='dp')
        trainer.fit(model, dm)
    return print("Completed CV")


def main():
    """Parse Arguments"""
    desc = "Script for training multiple methods of conditional featurization for prediction of cell viability."
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # positional
    parser.add_argument("experiment", type=str, choices=['id', 'concat', 'film'],
        help="Model type.")
    parser.add_argument("--batch-size", type=int,
        help="Size of batches for training.")
    parser.add_argument("--nfolds", type=int, choices=[1,2,3,4,5],
        help="Number of folds to run (sequential).")
    parser.add_argument("--test-run", default=False, action="store_true",
        help="Use a subset of the data for testing.")
    args = parser.parse_args()
    
    data, input_cols, cond_cols = prepare(exp=args.experiment,
                                          subset=args.test_run)
    return cv(exp=args.experiment,
              nfolds=args.nfolds,
              data=data,
              input_cols=input_cols,
              cond_cols=cond_cols,
              batch_size=args.batch_size)


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())