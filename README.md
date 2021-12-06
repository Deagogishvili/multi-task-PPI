# multi-task-PPI

Protein protein interactions (PPI) are crucial for protein functioning, nevertheless predicting residues in PPI interfaces from the protein sequence remains a challenging problem. In addition, structure-based functional annotations, such as the PPI interface annotations, are scarce: only for about one-third of the protein structures dataset residue-based PPI interface annotations are available. If we want to use a deep learning strategy, we have to overcome the problem of limited data availability. Here we use a multi-task learning strategy, that can handle missing data. We start with the multi-task model architecture and structural annotated training data of OPUS-TASS, while carefully handling missing data in the cost function. As related learning tasks we include secondary structure, solvent accessibility and buried residues. Our results show that the multi-task learning strategy significantly outperforms single task approaches. Moreover, only the multi-task strategy is able to effectively learn over a dataset extended with structural feature data, while still missing PPI annotations. The multi-task setup becomes even more important, if the fraction of PPI annotations becomes very small: the multi-task learner trained on only one-eighth of the PPI annotations -- with data extension -- reaches the same performances as the single-task learner on all PPI annotations. Thus, we show that the multi-task learning strategy can be beneficial for a small training dataset where the protein's functional properties of interest are only partially annotated.

### Data
The .txt files contain the PDB IDs of the protein structures belonging to the training, validation and test set.

### Code
The code is based on the OPUS-TASS model of Xu et al. (2020)

The "Train_" scripts are used to train the model including the prediction tasks indicated by the following abbreviations:
IF: Protein-protein interaction interface 
BU: Buried residues
SA: Absolute solvent accessibility
S3: Secondary structure in 3 classes
S8: Secondary structure in 8 classes

The model contains of a CNN, RNN and a transformer. 

Training models on smaller numbers of training data (see Figure 4) is performed using a part of the total dataset ("utils_mylabels_prob_partdata" and using a part of the PPI annotations "utils_mylabels_prob_partPPI".

## Reference
...
