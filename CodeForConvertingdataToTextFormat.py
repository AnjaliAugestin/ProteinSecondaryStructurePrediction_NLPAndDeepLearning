import os
import numpy as np
import gzip
import pickle

# Define paths for dataset
TRAIN_PATH = 'pssp-data/cullpdb+profile_6133_filtered.npy.gz'
TEST_PATH = 'pssp-data/cb513+profile_split1.npy.gz'

import numpy as np

def load_npy(file_path):
    """Load numpy file directly."""
    return np.load(file_path, allow_pickle=True)

def make_dataset(path):
    """Create dataset from the given path."""
    data = load_npy(path)  # Load the .npy file
    data = data.reshape(-1, 700, 57)

    idx = np.append(np.arange(21), np.arange(35, 56))
    X = data[:, :, idx].transpose(0, 2, 1).astype('float32')

    y = data[:, :, 22:30]
    y = np.array([np.dot(yi, np.arange(8)) for yi in y]).astype('float32')

    # Calculate the sequence length from the mask
    mask = data[:, :, 30] * -1 + 1
    seq_len = mask.sum(axis=1).astype(int)

    return X, y, seq_len  # Now returning seq_len properly


def get_amino_acid_array(X_amino, seq_len):
    """Get readable amino acid sequences from one-hot encoded data."""
    amino_acid = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M',
                  'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X']
    amino_acid_array = []
    for X, l in zip(X_amino, seq_len):
        acid = {}
        for i, aa in enumerate(amino_acid):
            keys = np.where(X[i] == 1)[0]
            values = [aa] * len(keys)
            acid.update(zip(keys, values))
        aa_str = ' '.join([acid[i] for i in range(l)])
        amino_acid_array.append(aa_str)
    return amino_acid_array

def get_pss_array(label, seq_len):
    """Get readable secondary structure from label data."""
    pss_icon = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T']
    pss_array = []
    for target, l in zip(label, seq_len):
        pss = np.array(['Nofill'] * l)
        target = target[:l]
        for i, p in enumerate(pss_icon):
            idx = np.where(target == i)[0]
            pss[idx] = p

        pss_str = ' '.join([pss[i] for i in range(l)])
        pss_array.append(pss_str)

    return pss_array

def main():
    # Create the directory for saving readable datasets
    if not os.path.exists('./readable_datasets'):
        os.makedirs('./readable_datasets')

    # Load the dataset (assuming the datasets are already downloaded)
    X_train, y_train, seq_len_train = make_dataset(TRAIN_PATH)
    X_test, y_test, seq_len_test = make_dataset(TEST_PATH)

    # Convert to readable format
    amino_acid_train = get_amino_acid_array(X_train, seq_len_train)
    pss_train = get_pss_array(y_train, seq_len_train)
    
    amino_acid_test = get_amino_acid_array(X_test, seq_len_test)
    pss_test = get_pss_array(y_test, seq_len_test)

    # Save the readable datasets
    with open('./readable_datasets/amino_acids_train.txt', 'w') as f:
        for sequence in amino_acid_train:
            f.write(sequence + '\n')
    
    with open('./readable_datasets/pss_train.txt', 'w') as f:
        for structure in pss_train:
            f.write(structure + '\n')

    with open('./readable_datasets/amino_acids_test.txt', 'w') as f:
        for sequence in amino_acid_test:
            f.write(sequence + '\n')
    
    with open('./readable_datasets/pss_test.txt', 'w') as f:
        for structure in pss_test:
            f.write(structure + '\n')

    print("Readable datasets have been saved!")

if __name__ == '__main__':
    main()