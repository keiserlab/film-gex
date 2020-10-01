#!/srv/home/wconnell/anaconda3/envs/fastai

"""
Author: Will Connell
Date Initialized: 2020-09-27
Email: connell@keiserlab.org

Script to preprocess data.
""" 


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       # 
#                                                                IMPORT MODULES                                                            
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #     
###########################################################################################################################################


# I/O
import os
from pathlib import Path
import joblib

# Data handling
import numpy as np
import pandas as pd
from sklearn import model_selection

# Chem
from rdkit import DataStructs
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import AllChem


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       # 
#                                                              PRIMARY FUNCTIONS                                                           
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #     
###########################################################################################################################################


def smiles_to_bits(smiles, nBits):
    mols = [MolFromSmiles(s) for s in smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=nBits) for m in mols]
    np_fps = []
    for fp in fps:
        arr = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        np_fps.append(arr)
    df = pd.DataFrame(np_fps)
    return df


###########################################################################################################################################
#        #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       # 
#                                                                 PROCESSING                                                           
#    #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #     
###########################################################################################################################################


if __name__ == "__main__":
    np.random.seed(2299)
    ## Read data
    # Paths
    ds_path = Path("../../film-gex-data/drug_screens/")
    cm_path = Path("../../film-gex-data/cellular_models/")
    out_path = Path("../../film-gex-data/processed/")
    # CCLE
    meta_ccle = pd.read_csv(cm_path.joinpath("sample_info.csv"))
    ccle = pd.read_csv(cm_path.joinpath("CCLE_expression.csv"), index_col=0)
    # L1000 genes
    genes = pd.read_csv(cm_path.joinpath("GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz"), sep="\t", index_col=0)
    # CTRP
    cp_ctrp = pd.read_csv(ds_path.joinpath("CTRP/v20.meta.per_compound.txt"), sep="\t", index_col=0)
    cl_ctrp = pd.read_csv(ds_path.joinpath("CTRP/v20.meta.per_cell_line.txt"), sep="\t", index_col=0)
    exp_ctrp = pd.read_csv(ds_path.joinpath("CTRP/v20.meta.per_experiment.txt"), sep="\t", index_col=0)
    ctrp = pd.read_csv(ds_path.joinpath("CTRP/v20.data.per_cpd_post_qc.txt") ,sep='\t', index_col=0)

    ## Merge data
    data = ctrp.join(exp_ctrp['master_ccl_id']).drop_duplicates()
    data = data.merge(cl_ctrp['ccl_name'], left_on='master_ccl_id', right_index=True)
    data = data.merge(cp_ctrp[['broad_cpd_id', 'cpd_smiles']], left_on='master_cpd_id', right_index=True)
    data = data.merge(meta_ccle[['stripped_cell_line_name', 'DepMap_ID']], left_on='ccl_name', right_on='stripped_cell_line_name')

    ## Generate Fingerprints
    nBits=512
    cpd_data = data[['broad_cpd_id', 'cpd_smiles']].drop_duplicates()
    bits = smiles_to_bits(cpd_data['cpd_smiles'], nBits=nBits)
    bits.index = cpd_data['broad_cpd_id']
    bits.columns =["FP_{}".format(i) for i in range(len(bits.columns))]

    ## Subset L1000 gene features by Entrez ID
    genes_lm = genes[genes['pr_is_lm']==1]
    ccle_cols = np.array([[gene[0], gene[1].strip("()")] for gene in ccle.columns.str.split(" ")])
    ccle.columns = ccle_cols[:,1]
    ccle = ccle[genes_lm.index.astype(str)]

    ## Merge bits and GEX
    data = data.merge(bits, left_on="broad_cpd_id", right_index=True)
    data = data.merge(ccle, left_on="DepMap_ID", right_index=True)
    print("{} unique compounds and {} unique cell lines comprising {} data points".format(data['broad_cpd_id'].nunique(),
                                                                                          data['stripped_cell_line_name'].nunique(),
                                                                                          data.shape[0]))
    
    ## Generate folds
    target = "cpd_avg_pv"
    group = "stripped_cell_line_name"
    n_splits = 5
    gene_cols = ccle.columns.to_numpy()
    fp_cols = np.append(bits.columns)
    
    data["fold"] = -1
    data = data.sample(frac=1).reset_index(drop=True)
    gkf = model_selection.GroupKFold(n_splits=n_splits)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X=data, y=data[target].to_numpy(), groups=data[group].to_numpy())):
        print(len(train_idx), len(val_idx))
        data.loc[val_idx, 'fold'] = fold
    
    ## Write out
    out_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(gene_cols, out_path.joinpath("gene_cols.pkl"))
    joblib.dump(fp_cols, out_path.joinpath("fp_cols.pkl"))
    data.to_pickle(out_path.joinpath("train.pkl.gz"))
    