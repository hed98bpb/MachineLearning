import numpy as np

def train_by_counting(obs, pi, A, phi, annotations, genomes):

    # Counting pi:
    NR_seen = 1
    C_seen = 1
    for anno in annotations:
        if annotations[anno][0] == 'C':
            C_seen += 1
        else:
            NR_seen += 1

    # Calculating pi
    pi[0] = NR_seen / len(annotations)
    pi[1] = C_seen / len(annotations)


    # Counters for transitions:
    N_to_N = 1
    N_to_1 = N_to_4 = N_to_7 = N_to_10 = N_to_13 = N_to_16 = N_to_19 = N_to_22 = N_to_25 = N_to_28 = 1
    C_to_C = 1
    C33_to_31 = C33_to_34 = C33_to_37 = C33_to_40 = 1
    C_to_N = 1

    # Counters for emissions
    emissions_from_N = [1 for i in range(len(obs))]
    emissions_from_31 = [1 for i in range(len(obs))]
    emissions_from_32 = [1 for i in range(len(obs))]
    emissions_from_33 = [1 for i in range(len(obs))]

    # The actual counting:
    for anno in annotations:
        anno_number = anno[len('annotation'):]
        previous_seen = annotations[anno][0]
        emissions_from_N[obs[genomes[('genome' + anno_number)][0]]] += 1 # counting the first obeservable in each genome

        C_counter = 0
        for i in range(1, len(annotations[anno])):
            letter_emitted = genomes[('genome' + anno_number)][i]
            # annotation is N or R
            if annotations[anno][i] == ('N' or 'R'):
                emissions_from_N[obs[letter_emitted]] += 1
                if previous_seen == ('N' or 'R'):
                    N_to_N += 1
                else:
                    C_to_N += 1

            # annotation is C
            else:
                if C_counter % 3 == 0: #start of a codon with 3 observables in

                    codon = genomes[('genome' + anno_number)][i:i + 3]  # get codon

                    if previous_seen == ('N' or 'R'): # then start-codon
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

                    else: #privious seen is C
                        if codon == 'TAA':
                            C33_to_34 += 1
                        elif codon == 'TAG':
                            C33_to_37 += 1
                        elif codon == 'TGA':
                            C33_to_40 += 1
                        else:
                            C33_to_31 += 1

                            emissions_from_31[obs[codon[0]]] += 1
                            emissions_from_32[obs[codon[1]]] += 1
                            emissions_from_33[obs[codon[2]]] += 1



                else: # in the middle of a codon
                    C_to_C += 1

                C_counter += 1
            previous_seen = annotations[anno][i]

    # Calculating A:
    all_N_to = N_to_N + N_to_1 + N_to_4 + N_to_7 + N_to_10 + N_to_13 + N_to_16 + N_to_19 + N_to_22 + N_to_25 + N_to_28
    assert(all_N_to != 0)
    A[0][0] = N_to_N / all_N_to
    A[0][1] = N_to_1 / all_N_to
    A[0][4] = N_to_4 / all_N_to
    A[0][7] = N_to_7 / all_N_to
    A[0][10] = N_to_10 / all_N_to
    A[0][13] = N_to_13 / all_N_to
    A[0][16] = N_to_16 / all_N_to
    A[0][19] = N_to_19 / all_N_to
    A[0][22] = N_to_22 / all_N_to
    A[0][25] = N_to_25 / all_N_to
    A[0][28] = N_to_28 / all_N_to

    all_C_to = C33_to_34 + C33_to_37 + C33_to_40 + C33_to_31
    assert (all_C_to != 0)

    A[33][31] = C33_to_31 / all_C_to
    A[33][34] = C33_to_34 / all_C_to
    A[33][37] = C33_to_37 / all_C_to
    A[33][40] = C33_to_40 / all_C_to


    # Calculating phi[state][observable]:
    for i in range(len(obs)):
        phi[0][i] = emissions_from_N[i] / np.sum(emissions_from_N)
        phi[31][i] = emissions_from_31[i] / np.sum(emissions_from_31)
        phi[32][i] = emissions_from_32[i] / np.sum(emissions_from_32)
        phi[33][i] = emissions_from_33[i] / np.sum(emissions_from_33)

    return pi, A, phi

