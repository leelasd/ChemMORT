# ChemMRAT

The ChemMRAT (Molecular Represent & Translate) consists of three modules, including **SMILES Encoder**, **Embedding Decoder** and **Molecular Optimizer**.

## Introduction

### SMILES Encoder

The ChemMRAT SMILES Encoder allows the user to easily embed a SMILES string to a 512-dimensional vector, which can be used for building a QSAR model. Especially for DNN, the encoding descriptors satisfy the fundamental idea of representation learning: DNNs should learn a suitable representation of the data from a simple but complete featurization rather than relying on sophisticated human-engineered representations. Besides, DNNs often require massive amounts of data for training, but the available QSAR data is often small. Through enumerating the SMILES of a molecule, the data is extended to several times of the original repository. Users can input a chemical to be evaluated in the following three ways: drawing it in an included chemical sketcher window, entering a structure text file, or imputing the SMILES of the chemical structures.

### Embedding Decoder

The ChemMRAT Embedding Decoder was implemented to translate the embedding descriptors, retrieved from ChemMRAT SMILES Encoder, to a SMILES string. The Decoder assists the molecular property optimization where the user could adjust the embedding descriptors to hit an aimed property, and then use decoder to obtain the SMILES of each molecule. Users can input a .csv file, and a .smi file can be returned after a few seconds to minutes.

### Molecular Optimizer

The ChemMRAT Molecular Optimizer,  merged the Encoder, Decoder and Particle Swarm Optimization (PSO) method, was designed to optimize molecules with respect to a single objective, under constraints with chemical substructures and a multi-objective value function. Not only does our proposed method exhibit competitive or better performance in finding optimal solutions compared to baseline method, is also achieves significant reduction in computational time. After users input the SMILES of a chemical structure and select property to be optimized, several best solutions can be obtained in the results.

#### Endpoint of Optimizer

| Endpoint | Description | Performance | Type | Method | Dataset |
| :------: | :--- | :--- | :--- | :--- | ---- |
| logD<sub>7.4</sub>  | Log of the octanol/water distribution coefficient at pH<sub>7.4</sub>.<br>\* Optimal: 1~3 | **Test Set**<br>RMSE: 0.647<br>MAE: 0.491<br>R<sup>2</sup>: 0.778<br>**5-Fold CV**<br>RMSE: 0.648<br>MAE: 0.494<br>R<sup>2</sup>: 0.780 | Basic  property | XGBoost |      |
| AMES   | The probability to be positive in Ames test. <br>\* The smaller AMES score, the less likely to be AMES positive. | **Test Set**<br>ACC: 0.809<br>SEN: 0.836<br>SPE: 0.777<br>AUC: 0.886<br>**5-Fold CV**<br>ACC: 0.808<br>SEN: 0.833<br>SPE: 0.778<br>AUC: 0.888 | Toxicity | XGBoost |      |
| Caco-2 | Papp (Caco-2 Permeability)<br> Optimal: higher than -5.15 Log unit or -4.70 or -4.80 | **Test Set**<br>RMSE: 0.307<br>MAE: 0.236<br>R<sup>2</sup>: 0.777<br>**5-Fold CV**<br>RMSE: 0.314<br>MAE: 0.234<br>R<sup>2</sup>: 0.750 | Absorption | DNN |      |
| MDCK | Papp (MDCK Permeability)<br> | **Test Set**<br>RMSE: 0.261<br>MAE: 0.195<br>R<sup>2</sup>: 0.748<br>**5-Fold CV**<br>RMSE: 0.313<br>MAE: 0.220<br>R<sup>2</sup>: 0.663 | Absorption | DNN | |
| PPB | Plasma Protein Binding<br>\* Significant with drugs that are highly protein-bound and have a low therapeutic index. | **Test Set**<br>RMSE: 0.138<br>MAE: 0.098<br>R<sup>2</sup>: 0.725<br>**5-Fold CV**<br>RMSE: 0.146<br>MAE: 0.102<br>R<sup>2</sup>: 0.707 | Distribution | DNN | |
| QED | quantitative estimate of drug-likeness | n/a | Drug-likeness score | Molecular Function | |
| SlogP | Log of the octanol/water partition coefficient, based on an atomic contribution model [[Crippen 1999](https://doi.org/10.1021/ci990307l)].<br>\* Optimal: 0< LogP <3<br>\* logP <0: poor lipid bilayer permeability.<br>\* logP >3: poor aqueous solubility. | Fitted on an extensive training set of 9920 molecules, with R<sup>2</sup> = 0.918 and σ = 0.677 | Basic  property | Molecular Function | |



## Dev Environment

```&#39;
tensorflow=='1.14.0'
scikit-learn=='0.23.2'
rdkit=='2019.03.1'
```

## Base

[cddd](https://github.com/jrwnter/cddd)<br>[mso](https://github.com/jrwnter/mso)

