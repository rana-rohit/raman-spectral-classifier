import numpy as np

def clinical_subset_eval(y_true, y_pred):
    valid_classes = [0, 2, 3, 5, 6]
    label_map = {0:0, 2:1, 3:2, 5:3, 6:4}

    # Step 1: filter based on true labels
    mask = np.isin(y_true, valid_classes)

    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]

    # Step 2: map ONLY valid predictions, mark others as -1
    y_pred_m = np.array([
        label_map[y] if y in label_map else -1
        for y in y_pred_f
    ])

    y_true_m = np.array([label_map[y] for y in y_true_f])

    # Step 3: remove invalid predictions (-1)
    valid_mask = y_pred_m != -1

    y_true_m = y_true_m[valid_mask]
    y_pred_m = y_pred_m[valid_mask]

    return y_true_m, y_pred_m