

def read_fasta_file(filename):
    """
    Reads the given FASTA file f and returns a dictionary of sequences.

    Lines starting with ';' in the FASTA file are ignored.
    """
    sequences_lines = {}
    current_sequence_lines = None
    with open(filename) as fp:
        for line in fp:
            line = line.strip()
            if line.startswith(';') or not line:
                continue
            if line.startswith('>'):
                sequence_name = line.lstrip('>')
                current_sequence_lines = []
                sequences_lines[sequence_name] = current_sequence_lines
            else:
                if current_sequence_lines is not None:
                    current_sequence_lines.append(line)
    sequences = {}
    for name, lines in sequences_lines.items():
        sequences[name] = ''.join(lines)
    return sequences

def write_fasta_file(seq_name, seq, filename):
    output_file = filename + '.fa'
    file = open(output_file, 'w')
    file.write('>' + seq_name + '\n')
    file.write(seq)
    file.close()


def write_hmm_file(obs, pi, A, phi, output_filename):
    output_file = 'Hmm_files/' + output_filename + '.txt'
    file = open(output_file, 'w')

    nb_of_states = len(pi)
    states = [str(i) for i in range(nb_of_states)]
    file.write('states\n' + str(nb_of_states) + '\n')
    for state in states:
        file.write(state + ' ')

    file.write('\nobsevables\n')
    file.write(str(len(obs)) + '\n')
    for i in range(len(obs)):
            for o in obs:
                if obs[o] == i:
                    file.write(o + ' ')

    file.write('\ninitProbs\n')
    for p in pi:
        file.write(str(p) + ' ')

    file.write('\ntransProbs\n')
    for line in A:
        for l in line:
            file.write(str(l) + ' ')
        file.write('\n')

    file.write('emProbs\n')
    for line in phi:
        for l in line:
            file.write(str(l) + ' ')
        file.write('\n')

    file.close()
