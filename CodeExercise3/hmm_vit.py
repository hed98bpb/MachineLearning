import numpy as np
from string import whitespace
import re

hmm_file = open('hmm.txt')
hmm_lines = hmm_file.readlines()

X = list('GTTTCCCAGTGTATATCGAGGGATACTACGTGCATAGTAACATCGGCCAA')
states = dict()
inv_states = dict()
obs = dict()
pi = None
A = []
phi = []

# getting all the information:
for i, line in enumerate(hmm_lines):
    if 'states' in line:
        s = list(hmm_lines[i + 2].translate(dict.fromkeys(map(ord, whitespace)))) # removing whitespace
        for j, state in enumerate(s):
            states[state] = j
        assert len(states) == int(hmm_lines[i + 1])
        inv_states = {v: k for k, v in states.items()}
    if 'observables' in line:
        l = hmm_lines[i + 2].translate(dict.fromkeys(map(ord, whitespace)))
        for j, observation in enumerate(list(l)):
            obs[observation] = j
        assert len(obs) == int(hmm_lines[i + 1])
    if 'initProbs' in line:
        pi = re.findall("\d+\.\d+\d+", hmm_lines[i + 1])
        pi = [float(i) for i in pi]
    if 'transProbs' in line:
        for trans_line_nb in range(i+1, i + len(states)+1):
            nbs = re.findall("\d+\.\d+\d+", hmm_lines[trans_line_nb])
            A.append([float(i) for i in nbs])
    if 'emProbs' in line:
        for em_line_nb in range(i+1, i+len(states)+1):
            nbs = re.findall("\d+\.\d+\d+", hmm_lines[em_line_nb])
            phi.append([float(i) for i in nbs])

hmm_file.close()

# finding Z* (viterbi decoding) Z* is the overall most likely explanation of X:

omega = [[0 for col in range(len(X))] for row in range(len(states))]

# Base case omega[z_1]:
for i in range(len(states)):
    omega[i][0] = pi[i]

# Computing omega[z_n] for n>1:
for n in range(1, len(X)):
    for k in range(len(states)):
        if phi[k][obs[X[n]]] != 0:
            for j in range(len(obs)):
                if A[k][j] != 0:
                    omega[k][n] = max(omega[k][n], phi[k][obs[X[n]]] * omega[j][n-1] * A[k][j])

# Backtracking - finding Z*:
omega = np.array(omega) # TODO: we should do the computations above with np.arrays instead
max_z_N_index = np.argmax(omega[:, len(X)-1])

Z_star = inv_states[max_z_N_index] # The last state in the string Z_star.

for n in reversed(range(len(X)-1)):
    column_n = omega[:, n]
    for k in range(len(states)):
        column_n[k] = phi[k][obs[X[n + 1]]] * omega[k, n] * A[int(states[(Z_star[0])])][k]
    z_n_index = np.argmax(column_n)

    Z_star = inv_states[z_n_index] + Z_star

print(Z_star)




