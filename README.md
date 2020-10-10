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
| logD<sub>7.4</sub>  | Log of the octanol/water distribution coefficient at pH<sub>7.4</sub>.<br>\* Optimal: 1~3 | **Test Set**<br>RMSE: 0.555±0.010<br>MAE: 0.426±0.007<br>R<sup>2</sup>: 0.840±0.004<br>**5-Fold CV**<br>RMSE: 0.562±0.009<br>MAE: 0.428±0.13<br>R<sup>2</sup>: 0.834±0.005 | Basic  property | XGBoost |      |
| AMES   | The probability to be positive in Ames test. <br>\* The smaller AMES score, the less likely to be AMES positive. | **Test Set**<br>ACC: 0.813±0.007<br>SEN: 0.835±0.013<br>SPE: 0.787±0.013<br>AUC: 0.888±0.004<br>**5-Fold CV**<br>ACC: 0.810±0.016<br/>SEN: 0.838±0.014<br/>SPE: 0.777±0.031<br/>AUC: 0.889±0.013 | Toxicity | XGBoost |      |
| Caco-2 | Papp (Caco-2 Permeability)<br> Optimal: higher than -5.15 Log unit or -4.70 or -4.80 | **Test Set**<br>RMSE: 0.332±0.007<br>MAE: 0.244±0.004<br>R<sup>2</sup>: 0.718±0.019<br>**5-Fold CV**<br>RMSE: 0.328±0.004<br/>MAE: 0.245±0.005<br/>R<sup>2</sup>: 0.728±0.011 | Absorption | XGBoost& Data Augment |      |
| MDCK | Papp (MDCK Permeability)<br> | **Test Set**<br>RMSE: 0.323±0.022<br>MAE: 0.232±0.011<br>R<sup>2</sup>: 0.650±0.041<br>**5-Fold CV**<br>RMSE: 0.322±0.034<br>MAE: 0.235±0.021<br>R<sup>2</sup>: 0.644±0.057 | Absorption | XGBoost& Data Augment | |
| PPB | Plasma Protein Binding<br>\* Significant with drugs that are highly protein-bound and have a low therapeutic index. | **Test Set**<br>RMSE: 0.152±0.003<br>MAE: 0.104±0.002<br>R<sup>2</sup>: 0.691±0.016<br>**5-Fold CV**<br>RMSE: 0.154±0.010<br>MAE: 0.106±0.007<br>R<sup>2</sup>: 0.691±0.025 | Distribution | DNN | |
| QED | quantitative estimate of drug-likeness | n/a | Drug-likeness score | Molecular Function | |
| SlogP | Log of the octanol/water partition coefficient, based on an atomic contribution model [[Crippen 1999](https://doi.org/10.1021/ci990307l)].<br>\* Optimal: 0< LogP <3<br>\* logP <0: poor lipid bilayer permeability.<br>\* logP >3: poor aqueous solubility. | Fitted on an extensive training set of 9920 molecules, with R<sup>2</sup> = 0.918 and σ = 0.677 | Basic  property | Molecular Function | |
| logS | Log of Solubility<br>\* Optimal: higher than -4 log mol/L<br>\* <10 μg/mL: Low solubility.<br>\* 10–60 μg/mL: Moderate solubility.<br>\* >60 μg/mL: High solubility | **Test Set**<br>RMSE: 0.854<br>MAE: 0.606<br>R<sup>2</sup>: 0.847<br>**5-Fold CV**<br>RMSE: 0.890<br>MAE: 0.632<br>R<sup>2</sup>: 0.820 | Basic  property | XGBoost | |
| hERG | The probability to be hERG Blocker<br>\* The higher hERG score, the more likely to be hERG Blocker. | **Test Set**<br/>ACC: 0.814±0.026<br/>SEN: 0.841±0.042<br/>SPE: 0.760±0.065<br/>AUC: 0.854±0.032<br/>**5-Fold CV**<br/>ACC: 0.800±0.036<br/>SEN: 0.820±0.068<br/>SPE: 0.754±0.147<br/>AUC: 0.857±0.053 | Toxicity | XGBoost | |
| Hepatoxicity | The probability of owning liver toxicity<br>\* The smaller hepatoxicity score, the less likely to be liver toxic. | **Test Set**<br/>ACC: 0.731<br/>SEN: 0.743<br/>SPE: 0.715<br/>AUC: 0.793<br/>**5-Fold CV**<br/>ACC: 0.700<br/>SEN: 0.698<br/>SPE: 0.698<br/>AUC: 0.759 | Toxicity | XGBoost | |
| LD50 | LD50 of acute toxicity<br>\* High-toxicity: 1\~50 mg/kg.<br>\* Toxicity: 51\~500 mg/kg.<br>\* low-toxicity: 501~5000 mg/kg. | **Test Set**<br/>ACC: 0.786<br/>SEN: 0.738<br/>SPE: 0.815<br/>AUC: 0.793<br/>**5-Fold CV**<br/>ACC: 0.784<br/>SEN: 0.737<br/>SPE: 0.814<br/>AUC: 0.858 | Toxicity | XGBoost | |


## Dev Environment

```&#39;
tensorflow=='1.14.0'
scikit-learn=='0.23.2'
rdkit=='2019.03.1'
```

## Base

[cddd](https://github.com/jrwnter/cddd)<br>[mso](https://github.com/jrwnter/mso)

