---

<div align="center">    
 
# Conditional Modulation of Gene Expression for Prediction of Cellular Sensitivity    

[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)



<!--  
Conference   
-->   
</div>
 
## Description   
Evaluation of methods conditioning gene expression by small-molecule structure and dosage for prediction of cellular percent viability. Scripts for data downloads, preprocessing, training, and evaluation are provided in this repository.

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/wconnell/film-gex

# install project   
cd film-gex
conda env create -f environment.yml
 ```   
 Next, navigate to any file and run it.   
 ```bash
# data download (this will automatically create a few directories)
bash download.sh

# preprocessing and fold creation
python project/preprocess.py /some/output/dir/.

# training cross validation
python project/train.py --h
```

## Imports
This project is setup as a package which means you can now easily import any file.
Analyses can be found in `project/notebooks/`.

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
