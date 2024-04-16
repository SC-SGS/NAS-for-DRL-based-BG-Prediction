import numpy as np
import pandas
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--csv_path", dest="csv_path", default=".")
args = argparser.parse_args()

# ----------------------------------------- Global parameters (config) -------------------------------------------------
csv_path = args.csv_path

# -------------------------------------------- Function definitions ----------------------------------------------------

def load_csv_data(path):
    df = pandas.read_csv(path)
    return df


def data_to_float(values):
    float_values = []
    for v in values:
        row_values = []
        for d in v:
            if d != "":
                row_values.append(float(d))
        float_values.append(row_values)
    return float_values


def calculate_std_pred_vs_gt(data_frame):
    total_rmse_vals, total_mae_vals = [], []
    prediction = [x.strip('[ ]').split(',') for x in data_frame['prediction'].values]
    ground_truth = [x.strip('[ ]').split(',') for x in data_frame['ground_truth'].values]
    for pred, gt in zip(prediction, ground_truth):
        pred_values = [x.strip('[ ]').split(',') for x in pred]
        pred_values = data_to_float([z[0].split(" ") for z in pred_values])
        gt_values = [x.strip('[ ]').split(',') for x in gt]
        gt_values = data_to_float([z[0].split(" ") for z in gt_values])
        for g, p in zip(gt_values, pred_values):
            rmse = np.sqrt(np.mean((np.array(p) - np.array(g)) ** 2))
            mae = np.mean(np.abs(np.array(p) - np.array(g)))
            total_rmse_vals.append(rmse)
            total_mae_vals.append(mae)

    rmse_std = np.std(total_rmse_vals)
    rmse_mae = np.std(total_mae_vals)
    print("Standard deviation RMSE: {}".format(rmse_std))
    print("Standard deviation MAE: {}".format(rmse_mae))

# ------------------------------------------------- Main loop ----------------------------------------------------------

if __name__ == '__main__':
    data = load_csv_data(csv_path)
    calculate_std_pred_vs_gt(data)
