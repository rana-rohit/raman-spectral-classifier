import numpy as np

def clinical_subset_eval(y_true, y_pred):
    valid_classes = [0, 2, 3, 5, 6]
    label_map = {0:0, 2:1, 3:2, 5:3, 6:4}

    mask = np.isin(y_true, valid_classes)

    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]

    y_true_m = np.array([label_map[y] for y in y_true_f])
    y_pred_m = np.array([label_map[y] for y in y_pred_f])

    return y_true_m, y_pred_m