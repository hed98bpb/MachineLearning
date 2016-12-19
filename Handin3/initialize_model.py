

"""
The model:
State 0 = N/R
state 1-30 = start-codon states:
        1-3 = ATG
        4-6 = ATC
        7-9 = ATA
        10-12 = GTG
        13-15 = ATT
        16-18 = CTG
        19-21 = GTT
        22-24 = CTC
        25-27 = TTA
        28-30 = TTG
state 31-33 = middle-codon states
state 34-42 = end-codon states
        34-36 = TAA
        37-39 = TAG
        40-42 = TGA

"""

nbf_states = 43

def get_model():

    # Observables: A C G T
    obs = {'A': 0, 'C': 1, 'G': 2, 'T': 3 }

    # Initializing pi
    pi = [0 for x in range(nbf_states)]

    # Initializing A[from][to]
    A = [[0 for x in range(nbf_states)] for x in range(nbf_states)]
    # Filling out all the transissions that should have the value 1
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

    # Initializing phi[state][obsevable] shape=(43,4)
    phi = [[0 for i in range(len(obs))] for i in range(nbf_states)]

    # Lists over states where we emit a specific letters in start- and end-codons
    Alist = [1, 4, 7, 13, 9, 27, 35, 36, 38, 42]
    Clist = [16, 22, 6, 24]
    Glist = [3, 10, 12, 18, 19, 30, 39, 41]
    Tlist = [2, 5, 8, 11, 14, 15, 17, 20, 21, 23, 25, 26, 28, 29, 34, 37, 40]

    for i in Alist:
        phi[i][obs['A']] = 1
    for i in Clist:
        phi[i][obs['C']] = 1
    for i in Glist:
        phi[i][obs['G']] = 1
    for i in Tlist:
        phi[i][obs['T']] = 1

    return obs, pi, A, phi