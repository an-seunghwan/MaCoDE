# Masked Language Modeling Becomes Conditional Density Estimation for Tabular Data Synthesis

This repository is the official implementation of 'Masked Language Modeling Becomes Conditional Density Estimation for Tabular Data Synthesis' (`MaCoDE`) with pytorch. 

> **_NOTE:_** This repository supports [WandB](https://wandb.ai/site) MLOps platform!

## Training & Evaluation 

### 1. Training 

- `--bins`: the number of bins used for discretization (default: `50`)
- `--dataset`: dataset options (`covtype`, `loan`, `kings`, `banknote`, `concrete`, `redwine`, `whitewine`, `breast`, `letter`, `abalone`)
- `--missing_type`: how to generate missing (`None`(complete data), `MCAR`, `MAR`, `MNARL`, `MNARQ`)
- `--missing_rate`: missingness rate (default: `0.3`)

#### Q1:
```
python main.py --imputation False --dataset <dataset> --missing_type None 
```   

#### Q2:
```
python main.py --imputation False --dataset <dataset> --missing_type <missing_type> --missing_rate <missing_rate>
```   

#### Q3:
```
python main.py --imputation True --dataset <dataset> --missing_type <missing_type> --missing_rate <missing_rate>
```   

### 2. Evaluation 

#### Q1:
```
python inference.py --imputation False --dataset <dataset> --missing_type None 
```   

#### Q2:
```
python inference.py --imputation False --dataset <dataset> --missing_type <missing_type> --missing_rate <missing_rate>
```   
- for privacy control experiment: `--tau`: user defined temperature for privacy controlling (default: `1.0`)
```
python inference.py --imputation False --dataset <dataset> --missing_type <missing_type> --missing_rate <missing_rate> --tau <tau>
```

#### Q3:
```
python imputer.py --imputation True --dataset <dataset> --missing_type <missing_type> --missing_rate <missing_rate>
```   

## Directory and codes

```
.
+-- assets 
+-- datasets
|       +-- imputation.py
|       +-- preprocess.py
|       +-- raw_data.py
+-- modules 
|       +-- evaluation_imputation.py
|       +-- evaluation.py
|       +-- metric_MLu.py
|       +-- metric_privacy.py
|       +-- metric_stat.py
|       +-- missing.py
|       +-- model.py
|       +-- train_missing.py
|       +-- train.py
|       +-- utility.py

+-- main.py
+-- inference.py
+-- imputer.py
+-- LICENSE
+-- README.md
```

## Citation
```

```
