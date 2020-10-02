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
import pyarrow.dataset as ds

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
    
    if exp=='id':
        cpd_id = "master_cpd_id"
        cond_cols = np.array([cpd_id, 'cpd_conc_umol'])
    else:
        fp_cols = joblib.load(data_path.joinpath("fp_cols.pkl"))
        cond_cols = np.append(fp_cols, ['cpd_conc_umol'])
        
    if subset:
        dataset = ds.dataset(data_path.joinpath("train_sub.feather"), format='feather')
    else:
        dataset = ds.dataset(data_path.joinpath("train.feather"), format='feather')
        
    return dataset, input_cols, cond_cols


def cv(name, exp, gpus, nfolds, dataset, input_cols, cond_cols):
    seed_everything(2299)
    cols = list(np.concatenate((input_cols, cond_cols, ['cpd_avg_pv'])))
    
    for fold in np.arange(0,nfolds):
        train = dataset.to_table(columns=cols, filter=ds.field('fold') != fold).to_pandas()
        val = dataset.to_table(columns=cols, filter=ds.field('fold') == fold).to_pandas()
        # DataModule
        dm = CTRPDataModule(train,
                            val,
                            input_cols,
                            cond_cols,
                            target='cpd_avg_pv')
        # Model
        if exp=='film':
            model = FiLMNetwork(len(input_cols), len(cond_cols))
        else:
            model = ConcatNetwork(len(input_cols), len(cond_cols))
        # Callbacks
        logger = TensorBoardLogger(save_dir=os.getcwd(),
                                   version="{}_{}_fold_{}".format(name, exp, fold),
                                   name='lightning_logs')
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0.01)
        # Trainer
        trainer = Trainer(auto_lr_finder=True,
                          auto_scale_batch_size='power',
                          max_epochs=25, 
                          gpus=gpus,
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
    parser.add_argument("name", type=str,
        help="Prepended name of experiment.")
    parser.add_argument("experiment", type=str, choices=['id', 'concat', 'film'],
        help="Model type.")
    parser.add_argument("--gpus", type=int, choices=[0,1,2,3,4,5,6,7],
        help="Number of gpus.")
    parser.add_argument("--nfolds", type=int, choices=[1,2,3,4,5],
        help="Number of folds to run (sequential).")
    parser.add_argument("--test-run", default=False, action="store_true",
        help="Use a subset of the data for testing.")
    args = parser.parse_args()
    
    dataset, input_cols, cond_cols = prepare(exp=args.experiment,
                                             subset=args.test_run)
    return cv(name=args.name,
              exp=args.experiment,
              gpus=args.gpus,
              nfolds=args.nfolds,
              dataset=dataset,
              input_cols=input_cols,
              cond_cols=cond_cols)


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())