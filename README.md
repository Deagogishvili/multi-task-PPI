# multi-task-PPI

Protein protein interactions (PPI) are crucial for protein functioning, nevertheless predicting residues in PPI interfaces from the protein sequence remains a challenging problem. In addition, structure-based functional annotations, such as the PPI interface annotations, are scarce: only for about one-third of the protein structures dataset residue-based PPI interface annotations are available. If we want to use a deep learning strategy, we have to overcome the problem of limited data availability. Here we use a multi-task learning strategy, that can handle missing data. We start with the multi-task model architecture and structural annotated training data of OPUS-TASS, while carefully handling missing data in the cost function. As related learning tasks we include secondary structure, solvent accessibility and buried residues. Our results show that the multi-task learning strategy significantly outperforms single task approaches. Moreover, only the multi-task strategy is able to effectively learn over a dataset extended with structural feature data, while still missing PPI annotations. The multi-task setup becomes even more important, if the fraction of PPI annotations becomes very small: the multi-task learner trained on only one-eighth of the PPI annotations -- with data extension -- reaches the same performances as the single-task learner on all PPI annotations. Thus, we show that the multi-task learning strategy can be beneficial for a small training dataset where the protein's functional properties of interest are only partially annotated.

### Data
The .txt files contain the PDB IDs of the protein structures belonging to the training, validation and test set.
The zip files containing the labels ("my_lables"), fasta files ("fastas"), input ("inputs") of the training and validation ("trainval") and test set ("test"), which are needed to train and evaluate the model, can be found at https://ibi.vu.nl/downloads/multi-task-PPI/. The PPI annotations per PDB ID are also available. 

### Code
The code is based on the OPUS-TASS model of Xu et al. (2020)

The "train_*.py" scripts are used to train the model including the prediction tasks indicated by the following abbreviations:
IF: Protein-protein interaction interface 
BU: Buried residues
SA: Absolute solvent accessibility
S3: Secondary structure in 3 classes
S8: Secondary structure in 8 classes

All training scripts use the model, which is stored in "my_model_mylabels_prob.py", and contain a transformer ("my_transformer.py"), CNN ("my_cnn.py") and a RNN ("my_rnn_prob.py"). The classses defined in the RNN file determine which prediction tasks the output contain. 
 
The code needed for training and evaluation the model on a part of the total dataset is stored in "utils_mylabels_prob_partdata.py". The code for training and evaluating the model on only a part of the PPI annotations is stored in "utils_mylabels_prob_partPPI.py". We provided the code for training and evaluating the IFBUS3SA model on only a part of the PPI annoations ("train_IFBUS3SA_partPPI.py)".

## Reference
...
