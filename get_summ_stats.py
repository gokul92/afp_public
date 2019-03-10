import numpy as np
import pandas as pd

def get_stats(y, y_pred, n, n_classes):
        argmax_prediction = np.argmax(y_pred, 1)
        argmax_y = np.argmax(y, 1)
        TP = np.count_nonzero(argmax_prediction * argmax_y)
        TN = np.count_nonzero((argmax_prediction - 1) * (argmax_y - 1))
        FP = np.count_nonzero(argmax_prediction * (argmax_y - 1))
        FN = np.count_nonzero((argmax_prediction - 1) * argmax_y)

        return [TP, TN, FP, FN]
