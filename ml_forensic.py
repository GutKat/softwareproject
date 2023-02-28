'''
Author: Katrin Gutenbrunner
Based on the ml_fornesic.py of RNADeep (https://github.com/ViennaRNA/RNAdeep)
Expanded for the Softwareproject 2022/23
Contains functions to analysis secondary structure predictions
'''

import multiprocessing

import ufold_predict
import RNA
from matplotlib import pyplot as plt
from ufold.utils import *
import time
from ufold.postprocess import postprocess_new as postprocess
from tqdm import tqdm

nts = ["A", "C", "G", "U"]
can_pair = {'A': 'U', 'C': 'G', 'G': 'C', 'U': 'A'}

random.seed(42)


def remove_conflicts(a, seq=None):
    """removes conflicts within a contact matrix

    copied from https://github.com/ViennaRNA/rnadeep/blob/main/examples/mlforensics.py
    """
    # extract make symmteric upper triangle
    a = np.triu((a + a.T) / 2, 1)
    # remove unpaired elements
    a = np.where((a < 0.5), 0, a)

    nbp = np.count_nonzero(a)

    # Get indices of the largest element in a
    (i, j) = np.unravel_index(a.argmax(), a.shape)
    while (i, j) != (0, 0):
        if not seq or canon_bp(seq[i], seq[j]):
            a[i,], a[:, j] = 0, 0  # looks inefficient
            a[j,], a[:, i] = 0, 0  # looks inefficient
            a[i, j] = -1
        else:
            a[i, j] = 0
        (i, j) = np.unravel_index(a.argmax(), a.shape)
    return np.where((a == -1), 1, a), nbp


def canon_bp(i, j):
    """checks if pair is canonical, yields 1 if pair is canonical, 0 otherwise

    copied from https://github.com/ViennaRNA/rnadeep/blob/main/examples/mlforensics.py
    """
    can_pair = {'A': {'A': 0, 'C': 0, 'G': 0, 'U': 1},
                'C': {'A': 0, 'C': 0, 'G': 1, 'U': 0},
                'G': {'A': 0, 'C': 1, 'G': 0, 'U': 1},
                'U': {'A': 1, 'C': 0, 'G': 1, 'U': 0}}
    return can_pair[i][j]


def julia_prediction(seqs, data):
    """creates structures from input and counts occuring pseudo-knots

    copied from https://github.com/ViennaRNA/rnadeep/blob/main/examples/mlforensics.py
    """
    nn_structs = []
    collisions = 0
    tot_nbp = 0
    pseudoknots = 0
    for (seq, nnd) in zip(seqs, data):
        ## remove one nesting
        a = np.reshape(nnd, (len(seq), len(seq)))

        # version 1: allow all base-pairs
        a, nbp = remove_conflicts(a)

        # Make a pair table by looping over the upper triangular matrix ...
        pt = np.zeros(len(seq) + 1, dtype=int)
        pt[0] = len(seq)
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                if a[i][j] == 1:
                    if pt[i + 1] == pt[j + 1] == 0:
                        pt[i + 1], pt[j + 1] = j + 1, i + 1
                    else:
                        collisions += 1

        # remove pseudoknots & convert to dot-bracket
        ptable = tuple(map(int, pt))
        processed = RNA.pt_pk_remove(ptable)
        if ptable != processed:
            pseudoknots += 1
        nns = RNA.db_from_ptable(processed)
        nn_structs.append(nns)
    tot_nbp /= len(seqs)

    return nn_structs, pseudoknots


def julia_looptypes(seqs, structs):
    """structural element (loop-types) analysis of input

    copied from https://github.com/ViennaRNA/rnadeep/blob/main/examples/mlforensics.py
    """
    stats = {'S': 0,  # stack (actually: paired)
             'E': 0,  # exterior
             'B': 0,  # bulge
             'H': 0,  # hairpin
             'I': 0,  # interior
             'M': 0}  # multi
    counts = {'#S': 0,  # stack (actually: paired)
              '#E': 0,  # exterior
              '#B': 0,  # bulge
              '#H': 0,  # hairpin
              '#I': 0,  # interior
              '#M': 0}  # multi
    for (seq, ss) in zip(seqs, structs):
        assert len(seq) == len(ss)
        # Counting paired vs unpaired is easy ...
        S = len([n for n in ss if n != '.'])
        L = len([n for n in ss if n == '.'])
        # ... but which loop are the unpaired ones in?
        tree = RNA.db_to_tree_string(ss, 5)  # type 5
        print(f'\r', end='')  # Unfortunately, the C function above prints a string!!!
        tdata = [x for x in tree.replace(")", "(").split("(") if x and x != 'R']
        scheck, lcheck = 0, 0
        for x in tdata:
            if x[0] == 'S':
                stats[x[0]] += 2 * int(x[1:]) / len(seq)
                counts[f'#{x[0]}'] += 1  # hmmm.... 1 or 2?
                scheck += 2 * int(x[1:])
            else:
                stats[x[0]] += int(x[1:]) / len(seq)
                counts[f'#{x[0]}'] += 1
                lcheck += int(x[1:])
        assert scheck == S and lcheck == L
    stats = {t: c / len(seqs) for t, c in stats.items()}
    counts = {t: c / len(seqs) for t, c in counts.items()}
    assert np.isclose(sum(stats.values()), 1.)
    return stats, counts


def get_bp_counts(seqs, structs):
    '''counts base pairs of given sequences and structures

    copied from https://github.com/ViennaRNA/rnadeep/blob/main/examples/mlforensics.py
    '''
    counts = {}
    for (seq, ss) in zip(seqs, structs):
        pt = RNA.ptable(ss)
        for i, j in enumerate(pt[1:], 1):
            if j == 0 or i > j:
                continue
            bp = (seq[i - 1], seq[j - 1])
            counts[bp] = counts.get(bp, 0) + 1
    return counts


def length_analysis(folder, n_lengths, stem_file_name = "", save = ""):
    '''analysis of relation between number of predicted base pairs and length of sequences

    The function analyzes the relation between number of bp and length of sequences and creates a plot of this realtion.
    It also prints the amount of occuring self-loops in the structures. This function is optimized for UFold and requires
    certain files within the given folder, which are:
        - {folder}/{stem_file_name}_{n}_sequence.npy
        - {folder}/{stem_file_name}_{n}_structure.npy
        - {folder}/{stem_file_name}_{n}_matrix_nopp.npy
        - {folder}/{stem_file_name}_{n}_matrix_pp.npy
    where n corresponding to the elements in the given list n_lengths. the stem of the file names are optional, if no stem is given (stem_file_name)
    the files should look like this: "{folder}/{n}_sequence.npy". For creating the files "/{stem_file_name}_{n}_matrix_nopp.npy and {stem_file_name}_{n}_matrix_pp.npy"
    the python script "predict.py" can be used.

    Args:
        folder (str): path to the folder where the required files are stored
            (files needed: sequence.npy, structure.npy, matrix_nopp.npy, matrix_pp.npy)
        n_lengths (list): list with the length of sequences (n), which should be evaluated
        stem_file_name (str, optional): stem name of the prediction file, e.g. for 1000 sequences stem_file_name = "N1000", default = ""
        save (str, optional): if given the plot is saved under this path, default = ""

    Returns:
        None
            plots a figure of number of basepairs vs sequence length
    '''
    #length we want to test
    bp_count_truth = []
    bp_count_no_pp = []
    bp_count_pp = []
    ids_nopp = []
    ids_pp = []
    if not folder.endswith("/"):
        folder += "/"
    for n in tqdm(n_lengths):
        prediction_file = f"{folder}{stem_file_name}_n{n}"

        sequence_file = prediction_file + "_sequence.npy"
        structure_file = prediction_file + "_structure.npy"
        matrix_nopp_file = prediction_file + "_matrix_nopp.npy"
        matrix_pp_file = prediction_file + "_matrix_pp.npy"

        # load the files
        seqs = np.load(sequence_file)
        vrna = np.load(structure_file)
        data_no_pp = np.load(matrix_nopp_file, allow_pickle=True)
        data_pp = np.load(matrix_pp_file, allow_pickle=True)

        #get the values of Vienna RNAFold
        bp_vrna = get_bp_counts(seqs, vrna)
        bp_count_truth.append(sum(bp_vrna.values()) / len(seqs))

        #check for unprocessed data
        bp_counts = 0   #bp counter
        ids = 0         #identity check counter (are there bp with itself)

        for a in data_no_pp:
            a = (a + a.T) / 2                   #make matrix symmetric
            id = a * np.identity(a.shape[0])    #check for 1s in diagonal (bp with itself)
            if torch.sum(id) != 0:
                ids += 1
            bp_counts += torch.sum(a)/2
        bp_count_no_pp.append(bp_counts/len(seqs))
        ids_nopp.append(ids)

        # check for postprocessed data
        ids = 0         #identity check counter (are there bp with itself)
        bp_counts = 0   #bp counter

        for a in data_pp:
            a = (a + a.T) / 2                   #make matrix symmetric
            bp_counts += torch.sum(a)/2
            id = a * np.identity(a.shape[0])    #check for 1s in diagonal (bp with itself)
            if torch.sum(id) != 0:
                ids += 1

        ids_pp.append(ids)
        bp_count_pp.append(bp_counts/len(seqs))     #get average # of bp pairs for length n postprocessed


    print(f"before postprocessing there are {sum(ids_nopp)} self-loops in the {len(seqs)*len(n_lengths)} sequences")
    print(f"after postprocessing there are {sum(ids_pp)} self-loops in  the {len(seqs)*len(n_lengths)} sequences")

    x = np.array(n_lengths)
    RNADeep = (x**2) * 0.002532 - x*0.021662 - 2.797929
    plt.plot(n_lengths, RNADeep)

    z = np.polyfit(x, bp_count_no_pp, 2)
    p = np.poly1d(z)
    plt.plot(x, p(x))

    z = np.polyfit(x, bp_count_pp, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x))

    plt.ylim(0,250)

    # plt.plot(n_lengths, bp_count_no_pp)
    # plt.plot(n_lengths, bp_count_pp)

    plt.legend(["RNA Deep", "UFold without postprocessing", "UFold with postprocessing"])
    plt.xlabel("sequence length")
    plt.ylabel("number of bp")
    plt.title("Number of base pairs vs. length")
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    if save:
        plt.savefig(save)
        print(f"Plot was saved as {save}.")
    plt.show()


def mcc(y_true, y_pred):
    '''calculates MCC of prediction and true structure

    copied from https://github.com/ViennaRNA/rnadeep/blob/main/rnadeep/metrics.py
    '''
    y_pred = y_pred.clone().detach()
    y_true = y_true.clone().detach()
    y_pred_pos = torch.round(torch.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = torch.round(torch.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = torch.sum(y_pos * y_pred_pos)
    tn = torch.sum(y_neg * y_pred_neg)
    fp = torch.sum(y_neg * y_pred_pos)
    fn = torch.sum(y_pos * y_pred_neg)
    numerator = (tp * tn - fp * fn)
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    eps = 1e-10
    return numerator / (denominator + eps)


def model_eval(contact_net, test_generator):
    '''evaluates different metrics of given UFold model with and without postprocessing

    function to evaluate the given UFold model without postprocessing and with postprocessing, on the given dataset
    UFold model must be given as an object of the class U_Net from Network.py
    The dataset must be given as an object of the class RNASSDataGenerator from data_generator.py
    Function evaluates: MCC, F1, precision, recall

    Args:
        contact_net (Network.U_Net): UFold model, which should be tested
        test_generator (RNASSDataGenerator): generator with the dataset, which should be tested

    Returns:
        results (tuple): contains the results for the unprocessed and processed data
            unprocessed data (tuple): contains average MCC, F1, precision and recall of the data set without postprocessing
            processed data (tuple): contains average MCC, F1, precision and recall of the data set with postprocessing
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contact_net.train()
    result_no_pp_train = list()
    result_pp_train = list()
    mcc_no_pp = 0
    mcc_pp = 0
    batch_n = 0
    run_time = []

    #iterate over our dataset
    for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name, nc_map, l_len in tqdm(test_generator):
        batch_n += 1
        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        #start run time measurement
        tik = time.time()
        with torch.no_grad():
            pred_contacts = contact_net(seq_embedding_batch)

        #get matrix with only 0 and 1 (threshold = 0.5)
        map_no_pp = (pred_contacts > 0.5).float()

        # post-processing
        pred_postprocessed = postprocess(pred_contacts, seq_ori, 0.01, 0.1, 100, 1.6, True, 1.5)
        # get matrix with only 0 and 1 (threshold = 0.5)
        map_pp = (pred_postprocessed > 0.5).float()

        #get run time of batch
        tok = time.time()
        t0 = tok - tik
        run_time.append(t0)


        #calculate mcc
        mcc_no_pp += mcc(contacts_batch, map_no_pp)
        mcc_pp += mcc(contacts_batch, map_pp)


        #calculate f1, precision and recall
        result_no_pp_tmp = list(map(lambda i: evaluate_exact_new(map_no_pp.cpu()[i], contacts_batch.cpu()[i]),range(contacts_batch.shape[0])))
        result_pp_tmp = list(map(lambda i: evaluate_exact_new(map_pp.cpu()[i], contacts_batch.cpu()[i]),range(contacts_batch.shape[0])))
        result_no_pp_train += result_no_pp_tmp
        result_pp_train += result_pp_tmp

    #divide mcc by size of dataset
    mcc_no_pp /= batch_n
    mcc_pp /= batch_n

    #unzip results (precision, recall and f1)
    p_no_pp, r_no_pp, f1_no_pp = zip(*result_no_pp_train)
    p_pp, r_pp, f1_pp = zip(*result_pp_train)
    return (mcc_no_pp, np.average(f1_no_pp), np.average(p_no_pp), np.average(r_no_pp)), (mcc_pp, np.average(f1_pp), np.average(p_pp), np.average(r_pp))


def structural_elements(file_path):
    '''analyses structural features of truth and prediction with and without postprocessing

    function to analysis structural elements of the true data and given prediction without postprocessing and with postprocessing
    evaluates:
        - structural features (External Loop (EL), Bulge Loop (BL), Hairpin Loop (HL), Internal Loop (IL), Multi Loop (ML))
        - Average number of these structural elements
        - relative frequencies of bp types

    Args:
        file_path (str): path to the required files, files needed (file_path_sequence.npy, file_path_structure.npy) - can be created with metrics.random_ml_forensic or metrics.fa2npy

    Returns:
        None
            prints the results of analysis
    '''
    header = (f'Model paired exterior bulge hairpin interior multi '
              f'#helices #exterior #bulge #hairpin #interior #multi '
              f'base-pairs   %GC   %CG   %AU   %UA   %GU   %UG          %NC')
    #required files
    sequence_file = file_path + "_sequence.npy"
    structure_file = file_path + "_structure.npy"
    matrix_nopp_file = file_path + "_matrix_nopp.npy"
    matrix_pp_file = file_path + "_matrix_pp.npy"

    #load the sequences and structure of "Truth" (vienna RNA)
    seqs = np.load(sequence_file)
    vrna = np.load(structure_file)
    data_no_pp = np.load(matrix_nopp_file, allow_pickle=True)
    data_pp = np.load(matrix_pp_file, allow_pickle=True)


    l = max(len(s) for s in seqs)
    if min(len(s) for s in seqs) != l:
        l = 0

    # Show loop types from the MFE structures
    lt_vrna, lt_counts = julia_looptypes(seqs, vrna)
    bp_vrna = get_bp_counts(seqs, vrna)
    bp_tot_vrna = sum(bp_vrna.values())
    bp_vrna = {bp: cnt / bp_tot_vrna for bp, cnt in bp_vrna.items()}

    nc_vrna = sum([val for (i, j), val in bp_vrna.items() if not canon_bp(i, j)])
    # assert nc_vrna == 0
    print(header)
    print((f"{'vrna':5s}"
           f"{lt_vrna['S']:>7.3f} "
           f"{lt_vrna['E']:>8.3f} "
           f"{lt_vrna['B']:>5.3f} "
           f"{lt_vrna['H']:>7.3f} "
           f"{lt_vrna['I']:>8.3f} "
           f"{lt_vrna['M']:>5.3f} "
           f"{lt_counts['#S']:>8.3f} "
           f"{lt_counts['#E']:>9.3f} "
           f"{lt_counts['#B']:>6.3f} "
           f"{lt_counts['#H']:>8.3f} "
           f"{lt_counts['#I']:>9.3f} "
           f"{lt_counts['#M']:>6.3f} "
           f"{bp_tot_vrna:>10d} "
           f"{bp_vrna[('G', 'C')]:>5.3f} "
           f"{bp_vrna[('C', 'G')]:>5.3f} "
           f"{bp_vrna[('A', 'U')]:>5.3f} "
           f"{bp_vrna[('U', 'A')]:>5.3f} "
           f"{bp_vrna[('G', 'U')]:>5.3f} "
           f"{bp_vrna[('U', 'G')]:>5.3f} "
           f"{nc_vrna:>12.10f} "))

    #no postprocessing
    nnss, pseudoknots_no_pp = julia_prediction(seqs, data_no_pp)
    # Show loop types from the neural network structures
    lt_nnss, lt_counts = julia_looptypes(seqs, nnss)
    bp_nnss = get_bp_counts(seqs, nnss)
    bp_tot_nnss = sum(bp_nnss.values())
    bp_nnss = {bp: cnt / bp_tot_nnss for bp, cnt in bp_nnss.items()}
    nc_nnss = sum([val for (i, j), val in bp_nnss.items() if not canon_bp(i, j)])
    print((f"{'no pp':5s}"
           f"{lt_nnss['S']:>7.3f} "
           f"{lt_nnss['E']:>8.3f} "
           f"{lt_nnss['B']:>5.3f} "
           f"{lt_nnss['H']:>7.3f} "
           f"{lt_nnss['I']:>8.3f} "
           f"{lt_nnss['M']:>5.3f} "
           f"{lt_counts['#S']:>8.3f} "
           f"{lt_counts['#E']:>9.3f} "
           f"{lt_counts['#B']:>6.3f} "
           f"{lt_counts['#H']:>8.3f} "
           f"{lt_counts['#I']:>9.3f} "
           f"{lt_counts['#M']:>6.3f} "
           f"{bp_tot_nnss:>10d} "
           f"{bp_nnss[('G', 'C')]:>5.3f} "
           f"{bp_nnss[('C', 'G')]:>5.3f} "
           f"{bp_nnss[('A', 'U')]:>5.3f} "
           f"{bp_nnss[('U', 'A')]:>5.3f} "
           f"{bp_nnss[('G', 'U')]:>5.3f} "
           f"{bp_nnss[('U', 'G')]:>5.3f} "
           f"{nc_nnss:>12.10f} "))

    #with postprocessing
    nnss, pseudoknots_pp = julia_prediction(seqs, data_pp)
    # Show loop types from the neural network structures
    lt_nnss, lt_counts = julia_looptypes(seqs, nnss)
    bp_nnss = get_bp_counts(seqs, nnss)
    bp_tot_nnss = sum(bp_nnss.values())
    bp_nnss = {bp: cnt / bp_tot_nnss for bp, cnt in bp_nnss.items()}
    nc_nnss = sum([val for (i, j), val in bp_nnss.items() if not canon_bp(i, j)])
    print((f"{'pp':5s}"
           f"{lt_nnss['S']:>7.3f} "
           f"{lt_nnss['E']:>8.3f} "
           f"{lt_nnss['B']:>5.3f} "
           f"{lt_nnss['H']:>7.3f} "
           f"{lt_nnss['I']:>8.3f} "
           f"{lt_nnss['M']:>5.3f} "
           f"{lt_counts['#S']:>8.3f} "
           f"{lt_counts['#E']:>9.3f} "
           f"{lt_counts['#B']:>6.3f} "
           f"{lt_counts['#H']:>8.3f} "
           f"{lt_counts['#I']:>9.3f} "
           f"{lt_counts['#M']:>6.3f} "
           f"{bp_tot_nnss:>10d} "
           f"{bp_nnss[('G', 'C')]:>5.3f} "
           f"{bp_nnss[('C', 'G')]:>5.3f} "
           f"{bp_nnss[('A', 'U')]:>5.3f} "
           f"{bp_nnss[('U', 'A')]:>5.3f} "
           f"{bp_nnss[('G', 'U')]:>5.3f} "
           f"{bp_nnss[('U', 'G')]:>5.3f} "
           f"{nc_nnss:>12.10f} "))

    print(f"Pseudo knots: \nUFold without postprocessing\t{pseudoknots_no_pp} of {len(seqs)}\nUFold with postprocessing\t\t{pseudoknots_pp} of {len(seqs)}")


def simple_seq_design(structure):
    '''creates sequences from given structure

    function which uses a simple sequence design approach to create a new sequence from a given structure.
    All unpaired position of the structure are filled randomly with A,C,G and U. Paired position are filled randomly
    with canonical pairs.

    Args:
        structure (str): secondary structure of RNA in form of dot-bracktes

    Returns:
        sequence (str): new created sequences, which could fold into given structure
    '''
    pairs = ct2struct(structure)
    seq = list(random.choices(nts, k=len(structure)))
    for pair in pairs:
        nt1 = random.choice(nts)
        seq[pair[0]] = nt1
        seq[pair[1]] = can_pair[nt1]
    return "".join(seq)


def known_structure_test(itr, structure, save = False, show = False, model = r"ufold_training/06_01_2023/12_11_2.pt", unbias_model = False):
    '''testing how likely model folds into over-represented structure

    Function to test how often the prediction of the given model and ViennaRNA folds into the given structure.
    It calculates the bp distance between predictions of given UFold model/ViennaRNA and the reference structure and plots it as a histogramm.

    Args:
        itr (int): number of sequences which should be created and used for testing
        structure (str): optional, structure, which should be used as a reference and for the sequence design
        save (bool, str): optional, if False, no figure is save. if str plot is saved under this path, default = False
        show (bool): optional,  whether to show the plot, if False, plot is not shown, if True, plot is shown
        model (str): optional, path to the model, which should be used for predicting, default = "ufold_training/06_01_2023/12_11_2.pt"
        unbias_model (bool, str): optional, path to unbias model (trained on unbias data), if False it is not included in histogramm, default = False

    Returns:
        None
    '''
    length = len(structure) + (5-len(structure)%5)
    diffs_vRNA = []
    vRNA_same = 0

    with open("data/input.txt", "w") as i:
        for _ in tqdm(range(itr)):
            seq = simple_seq_design(structure)
            i.write(f">{_}\n")
            i.write(f"{seq}\n")

            vRNA_ss = RNA.fold(seq)[0]
            bp_dis = RNA.bp_distance(structure, vRNA_ss)
            diffs_vRNA.append(bp_dis)
            if bp_dis == 0:
                vRNA_same += 1

    child = multiprocessing.Process(target=ufold_predict.main(model))
    #multiprocessing.freeze_support()
    child.start()
    # wait for the child process to terminate
    child.join()

    UFold_same_bias = 0
    diffs_UFold_bias = []
    #biased
    with open("results/input_dot_ct_file.txt", "r") as file:
        lines = file.readlines()
    predictions = lines[2::4]
    for pred in predictions:
        struct = pred.replace("\n", "")
        bp_dis = RNA.bp_distance(structure, struct)
        diffs_UFold_bias.append(bp_dis)
        if bp_dis == 0:
            UFold_same_bias += 1

    if unbias_model:

        child = multiprocessing.Process(target=ufold_predict.main(unbias_model))
        #multiprocessing.freeze_support()
        child.start()
        # wait for the child process to terminate
        child.join()

        UFold_same_unbias = 0
        diffs_UFold_unbias = []

        #biased
        with open("results/input_dot_ct_file.txt", "r") as file:
            lines = file.readlines()
        predictions = lines[2::4]
        for pred in predictions:
            struct = pred.replace("\n", "")
            bp_dis = RNA.bp_distance(structure, struct)
            diffs_UFold_unbias.append(bp_dis)
            if bp_dis == 0:
                UFold_same_unbias += 1

    print(f"UFold biased: {UFold_same_bias} of the {itr} predictions have an identical structure to the original structure")
    if unbias_model:
        print(f"UFold unbiased: {UFold_same_unbias} of the {itr} predictions have an identical structure to the original structure")
    print(f"ViennaRNA: {vRNA_same} of the {itr} predictions have an identical structure to the original structure")

    # bins = np.linspace(0, 75, 31)
    bins = np.linspace(0, length, int(length/2.5 + 1))

    plt.hist(diffs_UFold_bias, bins, alpha=0.5, color="red", label='UFold bias model', weights=np.ones_like(diffs_UFold_bias) / len(diffs_UFold_bias))
    if unbias_model:
        plt.hist(diffs_UFold_unbias, bins, alpha=0.5, color="blue", label='UFold unbias model', weights=np.ones_like(diffs_UFold_unbias) / len(diffs_UFold_unbias))
    plt.hist(diffs_vRNA, bins, alpha=0.5, color="green", label='vRNA', weights=np.ones_like(diffs_vRNA) / len(diffs_vRNA))
    plt.title("base pair distance of true structure and predictions")
    plt.xlabel("base pair distance")
    plt.ylabel("relative frequency")
    plt.legend(loc='upper right')
    if save:
        plt.savefig(save)
        print(f"Figure was saved under {save}")
    if show:
        plt.show()
    return None


if __name__ == '__main__':
    tRNA_ss_42 = "(((((((..((((........)))).(((((.......))))).....(((((.......))))))))))))."
    sRNA_e_ss_42 = "(((((((((....(((((((......((((((............))))..)).....))))).)).((((((.....(((((..((....)).)))))....)))))).)))))))))...."
    sRNA_b_ss_42 = "(.((((((.....((((((((.....((((((.............))))..))....)))))).)).((.((....((((((((....))))))))....)).))...)))))).)."
    # tRNA_ss_42 = seed 42,   dbD00000793.ct
    # sRNA_e_ss_42 = seed 42, Eukaryota/E02308.ct
    # sRNA_b_ss_42 = seed 42, Bacteria/B00695.ct
    structure = sRNA_e_ss_42
    bias_model = "ufold_training/06_01_2023/12_11_2.pt"
    unbias_model_120 = "ufold_training/sorted/n120/N125000_e0.pt"
    unbias_model_70 = "ufold_training/sorted/n70/N125000_e0.pt"
    known_structure_test(2000, structure, save=False, show=True, model = bias_model, unbias_model=unbias_model_120) #save="results/bp_distance_tRNA_inverse_10000.png" save="results/bp_distance_tRNA_inverse_10000.png"

    # folder = "data/analysis/length/length_test_ufold_model/"
    # folder = "data/analysis/type_analysis/N2000_n70_n100/"
    # length_analysis(folder)
    # prediction_file = "data/analysis/type_analysis/N2000_n70_n100/N2000_n100"
    # prediction_file = "data/analysis/type_analysis/ufold_model/N2000_n100"
    # structural_elements(prediction_file)
    #n_range = list(range(30,251,10))
    #length_analysis(folder, n_range, "N1000")

