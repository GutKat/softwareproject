'''
Author Katrin Gutenbrunner
Script for creating new sequences from a fasta file. The Script was created as part of the softwareproject 2022/23.
The code is based on the tutorial "Hua-Ting Yao, Yann Ponty, Sebastian Will. Developing complex RNA design applications in the Infrared framework. RNA Folding - Methods and Protocols, 2022. hal-03711828v2."
It takes in an input fasta file with the argument -i / --input and stores the newly created sequences in the output file, which can be specified with the arguemnt -o / --output
The sequence design approach can be chosen using the argument -d / --design, where 3 possibiilites are available (1,2,3)
1 = simple sequence design
2 = frequency based sequence design
3 = constraint generation sequence design

The script could not be tested (Infrared and ViennaRNA was not compatible with computer) 
'''

import RNA
import infrared as ir
import infrared.rna as rna
import random
import math
from collections import Counter
import time
import argparse

parser = argparse.ArgumentParser(description='A program for creating new sequences from a fasta file using Infrared.')
parser.add_argument('-i', '--input', type=str, help='the input fasta file')
parser.add_argument('-o', '--output', type=str, help='the output file for the created sequences')
parser.add_argument('-d', '--design', choices=["1","2","3"], help=f'the sequence design approach, which should be used, possibilites are: 1, 2, 3', default = 1)
args = parser.parse_args()


if args.input and args.output:
    input_file = args.input
    output_file = args.output

    structures = []
    sequences = []
    try:
        with open(input_file, "r") as f:
            for i, line in enumerate(f):
                if i%3==2:
                    structures.append(line[:-1])
                elif i%3==1:
                    sequences.append(line[:-1])
    except:
        print("Fasta file could not be read.")
        print("Fasta file must be in the format:")
        print("\t - Name")
        print("\t - sequence")
        print("\t - structure")
        exit()

    new_sequences = []

    bases = ["A", "C", "G", "U"]
    if int(args.design) == 1:
        for target, seq in zip(structures, sequences):
            n = len(target)
            model = ir.Model(n, 4)
            model.add_constraints(rna.BPComp(i,j) for (i,j) in rna.parse(target))
            model.add_functions([rna.GCCont(i) for i in range(n)], 'gc')

            base_count = Counter(seq)
            GC = (base_count["G"]+base_count["C"]) / len(seq)
            model.set_feature_weight(GC, 'gc')

            sampler = ir.Sampler(model)
            sample = sampler.sample()
            new_seq = rna.ass_to_seq(sample)
            new_sequences.append(new_seq)
    
    
    elif int(args.design) == 2:
        def single_target_design_model(target):
            n, bps = len(target), rna.parse(target)
            model = ir.Model(n, 4)
            model.add_constraints(rna.BPComp(i, j) for (i, j) in bps)
            model.add_functions([rna.GCCont(i) for i in range(n)], 'gc')
            model.add_functions([rna.BPEnergy(i, j, (i-1, j+1) not in bps) for (i,j) in bps], 'energy')
            #model.set_feature_weight(-1.5, 'energy')
            return model


        def target_frequency(sequence, target):
            fc = RNA.fold_compound(sequence)
            fc.pf()
            return fc.pr_structure(target)

        
        for j, target in enumerate(structures):
            n = len(target)
            sampler = ir.Sampler(single_target_design_model(target))

            best = 0
            best_seq = None
            for i in range(100):
                new_seq = rna.ass_to_seq(sampler.targeted_sample())
                freq = target_frequency(new_seq,target)
                if freq > best:
                    best = freq
                    best_seq = new_seq
            if best_seq:
                new_sequences.append(best_seq)
            else:
                new_sequences.append(new_seq)
            if j%10 == 0 and j != 0:
                print(f"{j} of {len(structures)} structures done")
        
    elif int(args.design) == 3:
        for target, seq in zip(structures, sequences):
            n = len(target)
            bps = rna.parse(target)
            def cg_design_iteration():
                model = single_target_design_model(target)
                model.add_constraints(rna.NotBPComp(i, j) for (i, j) in dbps)
                sampler = ir.Sampler(model, lazy=True)
                if sampler.treewidth() > 10 or not sampler.is_consistent():
                    return "Not found"
                ctr = Counter()
                found, sol = False, None
                for i in range(50):
                    seq = rna.ass_to_seq(sampler.targeted_sample())
                    fc = RNA.fold_compound(seq)
                    mfe, mfe_e = fc.mfe()
                    if target == mfe:
                        sol = seq
                        break 
                    ctr.update(rna.parse(mfe))
                ndbps = [x[0] for x in ctr.most_common() if x[0] not in bps]
                dbps.extend(ndbps[:2])
                return sol
            dbps, seq = [], None
            while seq is None:
                seq = cg_design_iteration()
            if seq:
                new_sequences.append(seq)
            if j%10 == 0 and j != 0:
                print(f"{j} of {len(structures)} structures done")


else:
    print("Input and output file must be given")
    print("Input file must be fasta file and can be defined with the argument --input or -i")
    print("Ouput file can be defined with the argument --output or -o")
    print("Sequence Designing approach can be defined with the argument --design or -d")