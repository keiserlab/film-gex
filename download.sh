#!/bin/bash/sh

mkdir -p ../film-gex-data/{cellular_models,drug_screens/{CTRP,GDSC,PRISM}}

## Cell models data
# CCLE
wget https://ndownloader.figshare.com/files/24613394 -O ../film-gex-data/cellular_models/sample_info.csv
wget https://ndownloader.figshare.com/files/24613325 -O ../film-gex-data/cellular_models/CCLE_expression.csv

## Drug sensitivity data
# CTRP
wget ftp://caftpd.nci.nih.gov/pub/OCG-DCC/CTD2/Broad/CTRPv2.0_2015_ctd2_ExpandedDataset/CTRPv2.0_2015_ctd2_ExpandedDataset.zip -O ../film-gex-data/drug_screens/CTRP/CTRPv2.0_2015_ctd2_ExpandedDataset.zip
# GDSC
wget https://depmap.org/portal/download/api/download?file_name=processed_portal_downloads%2Fgdsc-drug-set-export-658c.5%2Fsanger-viability.csv&bucket=depmap-external-downloads -O ../film-gex-data/drug_screens/GDSC/sanger-viability.csv
# PRISM
wget https://ndownloader.figshare.com/files/20237709 -O ../film-gex-data/drug_screens/PRISM/primary-screen-replicate-collapsed-logfold-change.csv
