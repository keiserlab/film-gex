# I/O
import sys
import joblib
from pathlib import Path

# Data handling
import pandas as pd
import numpy as np
from sklearn import model_selection

# Modeling
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping

# Custom
from datasets import Dataset, CTRPDataModule
from film_model import FiLMNetwork

# Read data
data_path = Path("../../film-gex-data/processed/")
input_cols = joblib.load(data_path.joinpath("input_cols.pkl"))
cond_cols = joblib.load(data_path.joinpath("cond_cols.pkl"))
data = pd.read_pickle(data_path.joinpath("train.pkl.gz"))

# Cross Validation
def cv(n_splits=5, target="cpd_avg_pv", group="stripped_cell_line_name"):
    seed_everything(2299)
    gkf = model_selection.GroupKFold(n_splits=n_splits)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X=data, y=data[target].to_numpy(), groups=data[group].to_numpy())):
        model_path = Path("l")
        train = data.iloc[train_idx]
        val = data.iloc[val_idx]

        dm = CTRPDataModule(train,
                            val,
                            input_cols,
                            cond_cols,
                            target=target,
                            batch_size=1280)
        model = FiLMNetwork(len(input_cols), len(cond_cols))
        trainer = Trainer(max_epochs=100, 
                          gpus=-1,
                          early_stop_callback=EarlyStopping(monitor='val_loss'),
                          distributed_backend='dp')
        trainer.fit(model, dm)
        joblib.dump(val_idx, "lightning_logs/version_{}/val_idx.fold_{}".format(fold, fold))
    
if __name__ == '__main__':  # pragma: no cover
    sys.exit(cv())