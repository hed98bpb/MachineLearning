import numpy as np
from CodeExercise3.util import get_hmm


# getting the hmm and X:
states, inv_states, obs, pi, A, phi = get_hmm('hmm.txt')
X = list('GTTTCCCAGTGTATATCGAGGGATACTACGTGCATAGTAACATCGGCCAA')
N = len(X)
K = len(states)


"""
Finding Z* (viterbi decoding) Z* is the overall most likely explanation of X:
"""
omega = [[0 for col in range(N)] for row in range(K)]

# Base case omega[z_1]:
for i in range(K):
    omega[i][0] = pi[i] * phi[i][obs[X[0]]]

# Computing omega[z_n] for n>1:
for n in range(1, N):
    for k in range(K):
        if phi[k][obs[X[n]]] != 0:
            for j in range(len(obs)):
                if A[k][j] != 0:
                    omega[k][n] = max(omega[k][n], phi[k][obs[X[n]]] * omega[j][n-1] * A[k][j])

# Backtracking - finding Z*:
omega = np.array(omega) # TODO: we should do the computations above with np.arrays instead
max_z_N_index = np.argmax(omega[:, N-1])

Z_star = inv_states[max_z_N_index] # The last state in the string Z_star.

for n in reversed(range(N-1)):
    column_n = omega[:, n]
    for k in range(K):
        column_n[k] = phi[k][obs[X[n + 1]]] * omega[k, n] * A[int(states[(Z_star[0])])][k]
    z_n_index = np.argmax(column_n)

    Z_star = inv_states[z_n_index] + Z_star

# printing the overall most likely explanation of X:
print(Z_star)


