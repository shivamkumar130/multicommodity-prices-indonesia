\"\"\"evaluate.py - evaluation utilities (MAE, plots)\"\"\"
def mae(y_true, y_pred):
    return abs(y_true - y_pred).mean()
