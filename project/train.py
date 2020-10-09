# I/O
import os
import sys
import joblib
from pathlib import Path
import argparse
from datetime import datetime

# Data handling
import pandas as pd
import numpy as np
from sklearn import model_selection
import pyarrow.dataset as ds

# Modeling
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

# Custom
from datasets import Dataset, CTRPDataModule
from models import FiLMNetwork, ConcatNetwork


def read(path, exp, fold, subset=True):
    """
    Format data for access.
    
    path :: Path to data location.
    exp :: Type of model to train/evaluate.
    subset :: Access data subset for testing.
    """
    path = Path(path)
    input_cols = joblib.load(path.joinpath("gene_cols.pkl"))
    
    if exp=='id':
        cpd_id = "master_cpd_id"
        cond_cols = np.array([cpd_id, 'cpd_conc_umol'])
    else:
        fp_cols = joblib.load(path.joinpath("fp_cols.pkl"))
        cond_cols = np.append(fp_cols, ['cpd_conc_umol'])
        
    if subset:
        train_ds = ds.dataset(path.joinpath("sub_train_fold_{}.feather").format(fold), format='feather')
        val_ds = ds.dataset(path.joinpath("sub_val_fold_{}.feather").format(fold), format='feather')
    else:
        train_ds = ds.dataset(path.joinpath("train_fold_{}.feather").format(fold), format='feather')
        val_ds = ds.dataset(path.joinpath("val_fold_{}.feather").format(fold), format='feather')

    return train_ds, val_ds, input_cols, cond_cols


def cv(name, exp, target, batch_size, path, gpus, nfolds, subset):
    seed_everything(2299)
    path = Path(path)

    for fold in range(nfolds):
        start = datetime.now()
        train_ds, val_ds, input_cols, cond_cols = read(path, exp, fold, subset)
        cols = list(np.concatenate((input_cols, cond_cols, [target])))
        train = train_ds.to_table(columns=cols).to_pandas()
        val = val_ds.to_table(columns=cols).to_pandas()
        # DataModule
        dm = CTRPDataModule(train,
                            val,
                            input_cols,
                            cond_cols,
                            target,
                            batch_size)
        print("Completed dataloading in {}".format(str(datetime.now() - start)))
        # Model
        if exp=='film':
            model = FiLMNetwork(len(input_cols), len(cond_cols), learning_rate=1e-3)
        else:
            model = ConcatNetwork(len(input_cols), len(cond_cols), learning_rate=1e-3)
        # Callbacks
        logger = TensorBoardLogger(save_dir=os.getcwd(),
                                   version="{}_{}_fold_{}".format(name, exp, fold),
                                   name='lightning_logs')
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0.001)
        # Trainer
        start = datetime.now()
        trainer = Trainer(default_root_dir=logger.log_dir, #in order to avoid lr_find_temp.ckpt conflicts
                          auto_lr_find=False,
                          auto_scale_batch_size=False,
                          max_epochs=10, 
                          gpus=[gpus],
                          logger=logger,
                          distributed_backend=False,
                          #callbacks=[early_stop],
                          flush_logs_every_n_steps=200)
        #trainer.tune(model=model, datamodule=dm) # for auto_lr_find
        trainer.fit(model, dm)
        print("Completed fold {} in {}".format(fold, str(datetime.now() - start)))
    
    return print("/done")


def main():
    """Parse Arguments"""
    desc = "Script for training multiple methods of conditional featurization for prediction of cell viability."
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # positional
    parser.add_argument("name", type=str,
        help="Prepended name of experiment.")
    parser.add_argument("experiment", type=str, choices=['id', 'concat', 'film'],
        help="Model type.")
    parser.add_argument("target", type=str, choices=['cpd_avg_pv', 'cpd_pred_pv'],
        help="Target variable.")
    parser.add_argument("bs", type=int,
        help="Training batch size.")
    parser.add_argument("path", type=str,
        help="Path to preprocessed data.")
    parser.add_argument("--gpus", type=int, choices=[0,1,2,3,4,5,6,7], default=5,
        help="Number of gpus.")
    parser.add_argument("--nfolds", type=int, choices=[1,2,3,4,5], default=1,
        help="Number of folds to run (sequential).")
    parser.add_argument("--test-run", default=False, action="store_true",
        help="Use a subset of the data for testing.")
    args = parser.parse_args()

    return cv(*args)


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())