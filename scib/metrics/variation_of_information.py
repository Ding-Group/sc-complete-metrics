import numpy as np
from sklearn.metrics import mutual_info_score

def variation_of_information(y1, y2):
    # y1,y2 to numpy
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    if y1.shape[0] != y2.shape[0]:
        raise ValueError("y1 and y2 must have the same length")

    # H(C) 和 H(C')
    def entropy(labels):
        # Count how many times each unique label appears
        _, counts = np.unique(labels, return_counts=True)
        # Each class probability = count / total number of samples
        probs = counts / counts.sum()
        # np.log uses natural logarithm (base e)
        return -(probs * np.log(probs)).sum()

    H1 = entropy(y1)
    H2 = entropy(y2)

    # I(C, C')
    I = mutual_info_score(y1, y2)

    return H1 + H2 - 2 * I
