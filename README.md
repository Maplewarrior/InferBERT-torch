# Inferbert pytorch implementation

## Introduction

Administering medication to patients is a critical element in healthcare and medical treatments. 
However, few drugs come without side effects or adverse events (AE), which are unwanted or harmful effects caused by medication \cite{2010AdverseEffects}. To track and understand the unwanted effects of drugs and medical treatments, the field of pharmacovigilance is essential. Pharmacovigilance, as defined by the European Commission, involves overseeing the safety of medicines with the goals of minimizing risk and enhancing the benefits ([EUPhamacovigilance](https://www.ema.europa.eu/en/human-regulatory-overview/pharmacovigilance-overview)).
This is done by collecting data on drug use and trying to detect potential unwanted effects.

Current methods to find statistical associations between drugs and adverse events are mostly based on two-by-two contingency tables and include methods such as proportional reporting ratio (PRR), reporting odds ratio (ROR), and empirical Bayes geometric mean (EBGM). 
These methods are inherently limited, however, since they cannot account for interactions between features in their predictions. 
The InferBERT model, proposed in (Wang2021InferBERT:Pharmacovigilance), addresses this shortcoming by pairing a transformer encoder with a post-hoc analysis to assess feature importance tested on two datasets extracted from the FDA Adverse Event Reporting System containing cases of analgesics-induced liver failure and Tramadol fatalities. 
In this project, we wish to reimplement the methods used in the InferBERT paper using Pytorch and reproduce the results obtained (Wang2021InferBERT:Pharmacovigilance).



Link to drive with model weights:
https://drive.google.com/drive/folders/1yePG7mih9w296gjyex6T2O-XkYqVvYmd?usp=drive_link
