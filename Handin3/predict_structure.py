from Handin3.initialize_model import get_model
from Handin3.file_handler import read_fasta_file, write_hmm_file
from Handin3.training_by_counting import train_by_counting

# Initializing the model:
obs, pi, A, phi = get_model()

write_hmm_file(obs, pi, A, phi, 'init_hmm')

# Getting training data:
genomes = {}
annotations = {}
reversed_genomes = {}
reversed_annotations = {}

for i in range(1, 6):
    filename_g = 'Data/genome' + str(i) + '.fa'
    genomes.update(read_fasta_file(filename_g))

    filename_a = 'Data/annotation' + str(i) + '.fa'
    annotations.update(read_fasta_file(filename_a))

# Also get the reversed genomes and annotations
for anno in annotations:
    nb = int(anno[len('annotation'):])

    # reverse annotation:
    string_anno = annotations[anno][::-1]
    rev_anno = ''
    for i in range(len(string_anno)):
        if string_anno[i] == 'R':
            rev_anno += 'C'
        elif string_anno[i] == 'C':
            rev_anno += 'R'
        elif string_anno[i] == 'N':
            rev_anno += 'N'

    reversed_annotations['annotation' + str(len(annotations) + nb)] = rev_anno

    # reverse genome:
    rev_geno = genomes['genome' + str(nb)][::-1]
    reversed_genomes['genome' + str(len(annotations) + nb)] = rev_geno

annotations.update(reversed_annotations)
genomes.update(reversed_genomes)

# training by counting:
pi, A, phi = train_by_counting(obs, pi, A, phi, annotations, genomes)

write_hmm_file(obs, pi, A, phi, 'tbc_hmm')

