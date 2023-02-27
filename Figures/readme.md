The figures created in the process of this project are saved within these folder. The following text shows what which file represents.

## arcplotter
seq_example_no_pp.png <br>
predicted base pairs of example sequence without postprocessing using web tool arcplotter, model was trained on inverse data for 5 epochs

seq_example_w_pp.png <br>
predicted base pairs of example sequence with postprocessing using web tool arcplotter, model was trained on inverse data for 5 epochs

## basepairs_metrics
bp_length.png <br>
number of base pairs vs sequence length of UFold model with and without postprocessing and of RNADeep, model used: UFold model "models/ufold_train.pt"

mcc_w_pp.png <br>
Matthews correlation coefficient of unbiased data vs. training set size of different sequence lengths with postprocessing
 
mcc_no_pp.png <br>
Matthews correlation coefficient of unbiased data vs. training set size of different sequence lengths without postprocessing

## known_structures
bp_distance_tRNA_inverse_10000.png <br>
testing of known structure, reference structure = tRNA "tdbD00000793.ct" of RNAStrAlign/tRNA_database, tested on 10000 sequences, model trained on inverse data for 5 epochs

bp_distance_sRNA_b_inverse_10000.png <br>
testing of known structure, reference structure = 5sRNA "B00695.ct" of RNAStrAlign/5S_rRNA_database/Bacteria, tested on 10000 sequences, model trained on inverse data for 5 epochs

bp_distance_sRNA_e_inverse_10000.png <br>
testing of known structure, reference structure = 5sRNA "E02308.ct" of RNAStrAlign/5S_rRNA_database/Eukaryota, tested on 10000 sequences, model trained on inverse data for 5 epochs

bp_distance_tRNA_ufold_10000.png <br>
testing of known structure, reference structure = tRNA "tdbD00000793.ct" of RNAStrAlign/tRNA_database, tested on 10000 sequences, UFold model used "models/ufold_train.pt"

bp_distance_sRNA_e_ufold_10000.png <br>
testing of known structure, reference structure = 5sRNA "E02308.ct" of RNAStrAlign/5S_rRNA_database/Eukaryota, tested on 10000 sequences, UFold model used "models/ufold_train.pt"
