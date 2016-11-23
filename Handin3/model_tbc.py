from Handin3.read_fasta import read_fasta_file

# Getting training data:
genomes = {}
annotations = {}
for i in range(1, 6):
    filename = 'Data/genome' + str(i) + '.fa'
    genomes.update(read_fasta_file(filename))
for i in range(1, 6):
    filename = 'Data/annotation' + str(i) + '.fa'
    annotations.update(read_fasta_file(filename))


# This defines the model and initial values in the model by trainingByCounting:
nbf_states = 43

# State 0 is N or R the rest is C
states = [x for x in range(nbf_states)]

# Observables: A C G T
obs = {'A': 0, 'C': 1, 'G': 2, 'T': 3 }

# Training by counting on initial observations calculating pi:
pi = [0 for x in range(nbf_states)]
NR_seen = 0
C_seen = 0
for anno in annotations:
    if annotations[anno][0] == 'C':
        C_seen += 1
    else:
        NR_seen += 1
#TODO: udregn pi
print(NR_seen)
#A
#phi