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
from models import ConditionalNetwork, StandardNetwork


def read(path, exp, fold, subset=True):
    """
    Format data for access.
    
    path :: Path to data location.
    exp :: Type of model to train/evaluate.
    subset :: Access data subset for testing.
    """
    path = Path(path)
    input_cols = joblib.load(path.joinpath("gene_cols.pkl"))
    
    if exp == 'id':
        cpd_id = "master_cpd_id"
        cond_cols = np.array([cpd_id, 'cpd_conc_umol'])
    else:
        fp_cols = joblib.load(path.joinpath("fp_cols.pkl"))
        cond_cols = np.append(fp_cols, ['cpd_conc_umol'])
    
    if exp == 'vanilla':
        input_cols = np.append(input_cols, cond_cols)
        cond_cols = np.array([])
        
    if subset:
        train_ds = ds.dataset(path.joinpath(f"sub_train_fold_{fold}.feather"), format='feather')
        val_ds = ds.dataset(path.joinpath(f"sub_val_fold_{fold}.feather"), format='feather')
    else:
        train_ds = ds.dataset(path.joinpath(f"train_fold_{fold}.feather"), format='feather')
        val_ds = ds.dataset(path.joinpath(f"val_fold_{fold}.feather"), format='feather')

    return train_ds, val_ds, input_cols, cond_cols


def cv(name, exp, target, batch_size, learning_rate, epochs, path, logs, nfolds, gpus, subset):
    seed_everything(2299)
    # Paths
    path = Path(path)
    logs = Path(logs)
    logs.mkdir(parents=True, exist_ok=True)

    for fold in nfolds:
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
        # Remove data from CPU
        del train, val
        print("Completed dataloading in {}".format(str(datetime.now() - start)))
        # Model
        if exp == 'vanilla':
            model = StandardNetwork(exp, len(input_cols), learning_rate=learning_rate, batch_size=batch_size)
        else:
            model = ConditionalNetwork(exp, len(input_cols), len(cond_cols), learning_rate=learning_rate, batch_size=batch_size)
        # Callbacks
        logger = TensorBoardLogger(save_dir=logs,
                                   version=f"{name}_{exp}_fold_{fold}",
                                   name='lightning_logs')
        early_stop = EarlyStopping(monitor='val_r2',
                                   min_delta=0.001,
                                   patience=5,
                                   verbose=False,
                                   mode='max')
        # Trainer
        start = datetime.now()
        trainer = Trainer(default_root_dir=logger.log_dir, #in order to avoid lr_find_temp.ckpt conflicts
                          auto_lr_find=False,
                          auto_scale_batch_size=False,
                          max_epochs=epochs, 
                          gpus=[gpus],
                          logger=logger,
                          distributed_backend=None,
                          callbacks=[early_stop,],
                          flush_logs_every_n_steps=200,
                          profiler=True)
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
    parser.add_argument("exp", type=str, choices=['vanilla', 'id', 'shift', 'scale', 'film'],
        help="Model type.")
    parser.add_argument("target", type=str, choices=['cpd_avg_pv', 'cpd_pred_pv'],
        help="Target variable.")
    parser.add_argument("--batch_size", type=int, default=32768,
        help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
        help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=15,
        help="Max number of epochs.")
    parser.add_argument("--path", type=str,
        help="Path to preprocessed data.")
    parser.add_argument("--logs", type=str,
        help="Path to model logs and checkpoints.")
    parser.add_argument("--nfolds", type=int, nargs="+", choices=[0,1,2,3,4], default=0, required=True,
        help="List of folds to run.")
    parser.add_argument("--gpus", type=int, choices=[0,1,2,3,4,5,6,7], default=5, required=True,
        help="Selected GPU.")
    parser.add_argument("--subset", default=False, action="store_true",
        help="Use a subset of the data for testing.")
    args = parser.parse_args()

    return cv(**vars(args))


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())