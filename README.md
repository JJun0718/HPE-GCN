echo "# HPE-GCN" >> README.md 
git init 
git add README.md 
git commit -m "first commit" 
git branch -M main 
git remote add origin https://github.com/JJun0718/HPE-GCN。 git
 git push -u origin main



# HPE-GCN

This repository is the source code of paper: "HPE-GCN: predicting efficacy of tonic formu-lae via graph convolutional networks integrat-ing traditionally defined herbal properties"

## Folders and files:

`/data` contains the raw data file used in this paper. The two excel files 'buyi_exband.xlsx' and 'zhongyaodatabase.xlsx' in the folder '/data/origin' are part of the raw data used in the paper. The other two excel files 'tonifying formulae.xlsx' and 'TCM-HPs.xlsx' are the English versions of the above two files .

`/Data_Process` contains the data partitioning and FHHG construction.

`/models` contains the two-layer GCN structure.

`/Result` contains the extraction of FHHG information (such as adjacency matrix, node characteristics)

`/utils` contains the output results


## Require

Python 3.6

Pytorch >= 1.6.0


## Reproducing results

1. (i) cd Data_Process (ii) Run `python standardfile.py`

2. Run `python build_FHHG.py`

3. (i) cd .. (ii) Run `python train.py`
