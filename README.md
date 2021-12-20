echo "# HPE-GCN" >> README.md 
git init 
git add README.md 
git commit -m "first commit" 
git branch -M main 
git remote add origin https://github.com/JJun0718/HPE-GCN。 git
 git push -u origin main



# HPE-GCN

This repository is the source code of paper: "HPE-GCN: predicting efficacy of tonic formulae via graph convolutional networks integrating traditionally defined herbal properties"

## Folders and files:

`/data` contains the raw data file used in this paper. The two excel files 'buyi_exband.xlsx' and 'zhongyaodatabase.xlsx' in the folder '/data/origin' are part of the raw data used in the paper. The other two excel files 'tonifying formulae.xlsx' and 'TCM-HPs.xlsx' are the English versions of the above two files.

`/Data_Process` contains the data partitioning and FHHG construction.

`/models` contains the two-layer GCN structure.

`/Result` contains the output results

`/utils` contains the extraction of FHHG information (such as adjacency matrix, node characteristics)


## Require

Python 3.6

Pytorch >= 1.6.0


## Input data

`/data/buyi.txt` indicates formula names, training/test split, formula labels. Each line is for a formula.

`/data/corpus/buyi.txt` contains raw herbs of each formula, each line is for the corresponding line in `/data/buyi.txt`

`/Data_Process/standardfile.py` is an example for preparing your own data.


## Train and evaluate

1. (i) cd Data_Process (ii) Run `python build_FHHG.py`

2. (i) cd .. (ii) Run `python train.py`