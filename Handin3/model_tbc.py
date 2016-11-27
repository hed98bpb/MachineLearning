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

# Calculating pi
pi[0] = NR_seen/len(annotations)
pi[1] = C_seen/len(annotations)

# Initializing A [from][to]
A = [[0 for x in range(nbf_states)] for x in range(nbf_states)]
A[1][2] = A[2][3] = A[3][31] = 1
A[4][5] = A[5][6] = A[6][31] = 1
A[7][8] = A[8][9] = A[9][31] = 1
A[10][11] = A[11][12] = A[12][31] = 1
A[13][14] = A[14][15] = A[15][31] = 1
A[16][17] = A[17][18] = A[18][31] = 1
A[19][20] = A[20][21] = A[21][31] = 1
A[22][23] = A[23][24] = A[24][31] = 1
A[25][26] = A[26][27] = A[27][31] = 1
A[28][29] = A[29][30] = A[30][31] = 1

A[31][32] = A[32][33] = 1

A[34][35] = A[35][36] = A[36][0] = 1
A[37][38] = A[38][39] = A[39][0] = 1
A[40][41] = A[41][42] = A[42][0] = 1




# Training by counting on transitions
N_to_N = 0
N_to_1 = N_to_4 = N_to_7 = N_to_10 = N_to_13 = N_to_16 = N_to_19 = N_to_22 = N_to_25 = N_to_28 = 0
C_to_C = 0
C33_to_31 = C33_to_34 = C33_to_37 = C33_to_40 = 0
C_to_N = 0


for anno in annotations:
    previous_seen = annotations[anno][0]
    C_counter = 0
    for i in range(1,len(annotations[anno])):
        if annotations[anno][i] == ('N' or 'R'):
            if previous_seen == ('N' or 'R'):
                N_to_N += 1
            else:
                C_to_N += 1
        else:
            codon = genomes[('genome' + anno[len(anno) - 1])][i:i + 3]
            if C_counter % 3 == 0:
                C_counter += 1
                if previous_seen == ('N' or 'R'):
                    if codon == 'ATG':
                        N_to_1 += 1
                    elif codon == 'ATC':
                        N_to_4 += 1
                    elif codon == 'ATA':
                        N_to_7 += 1
                    elif codon == 'GTG':
                        N_to_10 += 1
                    elif codon == 'ATT':
                        N_to_13 += 1
                    elif codon == 'CTG':
                        N_to_16 += 1
                    elif codon == 'GTT':
                        N_to_19 += 1
                    elif codon == 'CTC':
                        N_to_22 += 1
                    elif codon == 'TTA':
                        N_to_25 += 1
                    elif codon == 'TTG':
                        N_to_28 += 1

                else:
                    if codon == 'TAA':
                        C33_to_34 += 1
                    elif codon == 'TAG':
                        C33_to_37 += 1
                    elif codon == 'TGA':
                        C33_to_40 += 1
                    else:
                        C33_to_31 += 1

            else:
                C_to_C += 1

        previous_seen = annotations[anno][i]
print(A)

# Calculate A
all_N_to = N_to_N + N_to_1 + N_to_4 + N_to_7 + N_to_10 + N_to_13 + N_to_16 + N_to_19 + N_to_22 + N_to_25 + N_to_28
A[0][1] = N_to_1/all_N_to
A[0][4] = N_to_4/all_N_to
A[0][7] = N_to_7/all_N_to
A[0][10] = N_to_10/all_N_to
A[0][13] = N_to_13/all_N_to
A[0][16] = N_to_16/all_N_to
A[0][19] = N_to_19/all_N_to
A[0][22] = N_to_22/all_N_to
A[0][25] = N_to_25/all_N_to
A[0][28] = N_to_28/all_N_to

all_C_to = C_to_C + C33_to_34 + C33_to_37 + C33_to_40 + C33_to_31
A[33][31] = C33_to_31/all_C_to
A[33][34] = C33_to_34/all_C_to
A[33][37] = C33_to_37/all_C_to
A[33][40] = C33_to_40/all_C_to


# Initializing phi [observable][state]
phi = [[0 for i in range(nbf_states)] for i in range(len(obs))]

# Lists over states where we emit the letters
Alist = [1, 4, 7, 13, 9, 27, 35, 36, 38]
Clist = [16, 22, 6, 24]
Glist = [3, 10, 12, 18, 19, 30, 39, 41]
Tlist = [2, 5, 8, 11, 14, 15, 17, 20, 21, 23, 25, 26, 28, 29, 34, 37, 40]

for i in Alist:
    phi[obs['A']][i] = 1
for i in Clist:
    phi[obs['C']][i] = 1
for i in Glist:
    phi[obs['G']][i] = 1
for i in Tlist:
    phi[obs['T']][i] = 1

emissions_from_N = [0 for i in range(len(obs))]
emissions_from_31 = [0 for i in range(len(obs))]
emissions_from_32 = [0 for i in range(len(obs))]
emissions_from_33 = [0 for i in range(len(obs))]
emissions_rest = [0 for i in range(len(obs))]


# Training by counting on emissions
for anno in annotations:
    C_counter = 0
    for i in range(len(annotations[anno])):
        letter = genomes[('genome' + anno[len(anno) - 1])][i]
        if annotations[anno][i] == ('N' or 'R'):
            emissions_from_N[obs[letter]] += 1
        else:
            pass


#TODO calculate emissions
#TODO check A


print(NR_seen)

#A
#phi