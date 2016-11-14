import numpy as np
from string import whitespace
import re

hmm_file = open('hmm.txt')
hmm_lines = hmm_file.readlines()

X = list('GTTTCCCAGTGTATATCGAGGGATACTACGTGCATAGTAACATCGGCCAA')
states = dict()
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
print(obs)
print(states)

# finding Z* (viterbi decoding) Z* is the overall most likely explanation of X:

omega = [[0 for col in range(len(X)-1)] for row in range(len(states))]

# Base case:
for i in range(len(states)):
    omega[i][0] = pi[i]

for n in range(1, len(X)-1):
    for k in range(len(states)):
        if phi[k][obs[X[n]]] != 0:
            for j in range(len(obs)):
                if A[k][j] != 0:
                    omega[k][n] = max(omega[k][n], phi[k][obs[X[n]]] * omega[j][n-1] * A[k][j])

for l in omega:
    print(l)
