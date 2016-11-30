import numpy as np
import math
def run_viterbi(obs, pi, A, phi, X):
    # getting the hmm and X:
    N = len(X)
    K = len(pi)

    """
    Finding Z* in log-space (viterbi decoding) Z* is the overall most likely explanation of X:
    """

    omega = [[-np.inf for col in range(N)] for row in range(K)]

    # Base case omega[z_1]:
    for i in range(K):
        if pi[i] != 0 and phi[i][obs[X[0]]] != 0:
            omega[i][0] = math.log(pi[i]) + math.log(phi[i][obs[X[0]]])

    # Computing omega[z_n] for n>0:
    for n in range(1, N):
        for k in range(K):
            if phi[k][obs[X[n]]] != 0:
                for j in range(K):
                    if A[j][k] != 0:
                        omega[k][n] = max(omega[k][n], math.log(phi[k][obs[X[n]]]) + omega[j][n-1]+ math.log(A[j][k]))


    # Backtracking - finding Z*:
    for line in omega:
        print(line)


    max_z_N_index = np.argmax([col[N - 1] for col in omega])
    print('log P(X,Z) (loglikelihood): ', omega[max_z_N_index][N - 1])

    Z_star = [max_z_N_index]  # The last state in the string Z_star.

    for n in reversed(range(N - 1)):
        column_n = [0 for col in range(K)]
        for k in range(K):
            column_n[k] = phi[Z_star[len(Z_star)-1]][obs[X[n + 1]]] + omega[k][n] + A[k][Z_star[len(Z_star)-1]]

        z_n_index = np.argmax(column_n)
        Z_star.insert(0, z_n_index)

    print(Z_star)

    translated_z = ''
    for i in Z_star:
        if i == 0:
            translated_z = translated_z + 'N'
        else:
            translated_z = translated_z + 'C'

    return translated_z