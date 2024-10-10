import numpy as np



def localization_error(predicted_idx, true_idx):

    return abs(predicted_idx - true_idx)



def precision_recall(predictions, labels):

    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return precision, recall