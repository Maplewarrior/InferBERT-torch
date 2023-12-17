# Inferbert pytorch implementation

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
The InferBERT model, proposed in (Wang2021InferBERT:Pharmacovigilance), addresses this shortcoming by pairing a transformer encoder with a post-hoc analysis to assess feature importance tested on two datasets extracted from the FDA Adverse Event Reporting System containing cases of analgesics-induced liver failure and Tramadol fatalities. 
In this project, we wish to reimplement the methods used in the InferBERT paper using Pytorch and reproduce the results obtained (Wang2021InferBERT:Pharmacovigilance).

## Data
The data used for the Inferbert paper was pulled from FAERS and can be accessed here: [Analgesics-induced Acute Liver Failure](https://drive.google.com/file/d/1VGGs7uxC4UiOIWFZ2LQ6N2cLweMxOSqi/view?usp=sharing) & [Tramadol-related mortalities](https://drive.google.com/file/d/1VIg5vpQhk2FbAwDBwTzyJ18LyxGZ6VII/view?usp=sharing).

> [!IMPORTANT]
> The datasets pulled should be placed in `InferBERT_data/Analgesics-induced_acute_liver_failure/dataset/` and `InferBERT_data/Analgesics-induced_acute_liver_failure/dataset/` for the Analgesics and Tramadol respectively.

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


### Causal Analysis

### Robustness Evaluation











Link to drive with model weights:
https://drive.google.com/drive/folders/1yePG7mih9w296gjyex6T2O-XkYqVvYmd?usp=drive_link
