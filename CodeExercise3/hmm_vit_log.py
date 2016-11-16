import numpy as np
from CodeExercise3.util import get_hmm
import math as math


# getting the hmm and X:
states, inv_states, obs, pi, A, phi = get_hmm('hmm.txt')
X = list('GTTTCCCAGTGTATATCGAGGGATACTACGTGCATAGTAACATCGGCCAA')
N = len(X)
K = len(states)

for i, p in enumerate(pi):
    if p != 0:
        pi[i] = np.log(p)
    else:
        pi[i] = -np.inf

for i in range(K):
    for j in range(K):
        if A[i][j] != 0:
            A[i][j] = np.log(A[i][j])
        else:
            A[i][j] = -np.inf

for i in range(K):
    for j in range(len(obs)):
        if phi[i][j] != 0:
            phi[i][j] = np.log(phi[i][j])
        else:
            phi[i][j] = -np.inf


"""
Finding Z* in log-space (viterbi decoding) Z* is the overall most likely explanation of X:
"""

omega = [[-np.inf for col in range(N)] for row in range(K)]

# Base case omega[z_1]:
for i in range(K):
    omega[i][0] = pi[i] + phi[i][obs[X[0]]]

# Computing omega[z_n] for n>0:
for n in range(1, N):
    for k in range(K):
        max_n = -np.inf
        for j in range(K):
            possible_max = omega[j][n-1] + A[j][k]
            if possible_max > max_n:
                max_n = possible_max
        omega[k][n] = phi[k][obs[X[n]]] + max_n



# Backtracking - finding Z*:

max_z_N_index = np.argmax([col[N-1] for col in omega])
print('log P(X,Z) (loglikelihood): ', omega[max_z_N_index][N-1])

Z_star = inv_states[max_z_N_index] # The last state in the string Z_star.

for n in reversed(range(N-1)):
    column_n = [0 for col in range(K)]
    for k in range(K):
        column_n[k] = phi[states[Z_star[0]]][obs[X[n + 1]]] + omega[k][n] + A[k][states[Z_star[0]]]

    z_n_index = np.argmax(column_n)
    Z_star = inv_states[z_n_index] + Z_star

# printing the overall most likely explanation of X:
print(Z_star)
