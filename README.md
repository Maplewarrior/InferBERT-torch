# Inferbert pytorch implementation
![Static Badge](https://img.shields.io/badge/project-status_done-green)


This repository was made for the final academic project in [02456 Deep Learning](https://kurser.dtu.dk/course/02456) at the Technical University of Denmark.

The project was created by:
- [s204125 Andreas Fiehn](https://github.com/AndreasLF)
- [s204138 Michael Harborg](https://github.com/Maplewarrior)
- [s204139 August Tollerup](https://github.com/4ug-aug)
- [s200925 Oliver Elmgreen](https://github.com/FenrisWulven)

Under the supervision of [Jes Frellsen](https://orbit.dtu.dk/en/persons/jes-frellsen).

## Content
<!-- TOC start  -->
- [Introduction](#introduction)
- [Data](#data)
   * [Preprocessing](#preprocessing)
- [Reproducing Results](#reproducing-results)
   * [Traning](#training)
   * [Causal Analysis](#causal-analysis)
   * [Robustness Evaluation](#robustness-evaluation)
<!-- TOC end -->


## Introduction

Administering medication to patients is a critical element in healthcare and medical treatments. 
However, few drugs come without side effects or adverse events (AE), which are unwanted or harmful effects caused by medication \cite{2010AdverseEffects}. To track and understand the unwanted effects of drugs and medical treatments, the field of pharmacovigilance is essential. Pharmacovigilance, as defined by the European Commission, involves overseeing the safety of medicines with the goals of minimizing risk and enhancing the benefits ([EUPhamacovigilance](https://www.ema.europa.eu/en/human-regulatory-overview/pharmacovigilance-overview)).
This is done by collecting data on drug use and trying to detect potential unwanted effects.

Current methods to find statistical associations between drugs and adverse events are mostly based on two-by-two contingency tables and include methods such as proportional reporting ratio (PRR), reporting odds ratio (ROR), and empirical Bayes geometric mean (EBGM). 
These methods are inherently limited, however, since they cannot account for interactions between features in their predictions. 
The InferBERT model, proposed in [Wang2021InferBERT:Pharmacovigilance](https://www.frontiersin.org/articles/10.3389/frai.2021.659622/full), addresses this shortcoming by pairing a transformer encoder with a post-hoc analysis to assess feature importance tested on two datasets extracted from the FDA Adverse Event Reporting System containing cases of analgesics-induced liver failure and Tramadol fatalities. 
In this project, we wish to reimplement the methods used in the InferBERT paper using Pytorch and reproduce the results obtained [Wang2021InferBERT:Pharmacovigilance](https://www.frontiersin.org/articles/10.3389/frai.2021.659622/full).

## Data
The data used for the Inferbert paper was pulled from FAERS and can be accessed here: [Analgesics-induced Acute Liver Failure](https://drive.google.com/file/d/1VGGs7uxC4UiOIWFZ2LQ6N2cLweMxOSqi/view?usp=sharing) & [Tramadol-related mortalities](https://drive.google.com/file/d/1VIg5vpQhk2FbAwDBwTzyJ18LyxGZ6VII/view?usp=sharing).

> [!IMPORTANT]
> The datasets pulled should be placed in `InferBERT_data/Analgesics-induced_acute_liver_failure/dataset/` and `InferBERT_data/Analgesics-induced_acute_liver_failure/dataset/` for the Analgesics and Tramadol respectively.

The directory structure should look like the following after adding the data:

```bash
├── .venv
├── configs
│   ├── liverfailure_config_ALF.yaml
│   ├── liverfailure_config.yaml
│   ├── stats_config.yaml
│   ├── tramadol_config_ALF.yaml
│   └── tramadol_config.yaml
├── InferBERT_data
│   └── LiverFailure
│       └── processed
│       ├── test.csv
│       ├── all.csv
│       ├── dev.csv
│       ├── feature.csv
│       └── train.csv
│   └── TramadolMortalities
│       └── processed
│       ├── test.csv
│       ├── all.csv
│       ├── dev.csv
│       ├── feature.csv
│       └── train.csv
[...]
```

### Preprocessing
To preprocess the data, run the scripts in `utils/preprocessing/` for Analgesics run `preprocess_rep_liver.py` and for Tramadol run `preprocess_tramadol_corrected.py`.
These scripts will generate preprocessed data located in the directory specified in `configs/liverfailure_config.yaml` and `configs/tramadol_config.yaml`.

## Reproducing Results
> [!NOTE]
> All results in this paper was produced on a High Performance Compute cluster using A100 GPU's using [Python 3.10](https://www.python.org/downloads/release/python-31013/). It is therefore adviced, if attempting to reproduce, to run in a similar framework.

Firstly, clone the repository:
```bash
git clone https://github.com/Maplewarrior/InferBERT-torch.git
```
**(Optional)**: In the root directory, create a virtual python environment using your preferred method. Our project used `venv`:
```bash
python -m venv venv
```
Remember to activate your environment.

**(Required)**: Install the requirements from `requirements.txt`
```bash
python -m pip install -r requirements.txt
```

### Training
The model is trained using the `utils/trainer.py` script based on configs provided in `configs/liverfailure_config.yaml` and `configs/tramadol_config.yaml`. Since training can be heavy, we have provided the weights for different pretrained models in [this google drive](https://drive.google.com/drive/folders/1yePG7mih9w296gjyex6T2O-XkYqVvYmd?usp=drive_link). 

These, can be used by changing the following configs:

```yml
[...]
model:
  model_version: albert-base-v2
  pretrained_ckpt: 'experiments/reproduction/outputs/liverfailure/model_weights.pt' # <------- leave blank if training is from scratch
  hidden_size: 768 # hidden dimension, default=768 for base-v2
  intermediate_size: 3072 # dimension in FFN of encoder block, default=3072 for base-v2
  n_attention_heads: 12
  n_memory_blocks: 0
  n_classes: 1 # positive/negative but dim 1 is needed for the loss function
  fc_dropout_prob: 0.2 # dropout at classification head
  attention_dropout_prob: 0.0 # dropout at MHA, default=0.0
  hidden_dropout_prob: 0.0 # dropout at FFN in encoder block, default=0.0
```

The pretrained models provided are *no_dropout*, *fc_dropout_02*, *all_dropout_02* for both Analgesics and Tramadol models.

> [!WARNING]
> Even with a pretrained model, it is no adviced to run on a personal computer as these models tend to eat RAM.

To conduct experiments on different model specifications one can modify the config as mentioned above.

Finally, to train the model run:
```bash
python utils/trainer.py
```

### Causal Analysis
To generate the causal analysis results run the `utils/causal_inference.py` script:
```bash
python utils/causal_inference.py
```
This will generate two csv files, one for the root effects and one for sub population effects. These are saved to `'experiments/reproduction/outputs/liverfailure/causality_output'` or otherwise stated paths in the `configs/liverfailure_config.yaml` or `configs/tramadol_config.yaml`.

### Robustness Evaluation


