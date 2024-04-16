import numpy as np
import pandas
import argparse
import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser()
argparser.add_argument("--csv_path", dest="csv_path", default=".")
argparser.add_argument("--patient", dest="patient", default="")
argparser.add_argument("--ph", dest="ph", default=30)
argparser.add_argument("--time_h", dest="time_h", default=30)
argparser.add_argument("--save_path", dest="save_path", default=".")
args = argparser.parse_args()

# ----------------------------------------- Global parameters (config) -------------------------------------------------
csv_path = args.csv_path
patient_id = args.patient
vis_time = args.time_h
prediction_horizon = args.ph
save_path = args.save_path

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


def plot_ts_prediction_vs_ground_truth(data_frame, subject_id):
    fig, ax = plt.subplots()
    prediction = [x.strip('[ ]').split(',') for x in data_frame['prediction'].values]
    ground_truth = [x.strip('[ ]').split(',') for x in data_frame['ground_truth'].values]
    all_pred_values, all_gt_values = [], []
    for pred, gt in zip(prediction, ground_truth):
        pred_values = [x.strip('[ ]').split(',') for x in pred]
        pred_values = data_to_float([z[0].split(" ") for z in pred_values])
        gt_values = [x.strip('[ ]').split(',') for x in gt]
        gt_values = data_to_float([z[0].split(" ") for z in gt_values])
        all_pred_values.append(pred_values)
        all_gt_values.append(gt_values)

    pred_values = np.array(all_pred_values).flatten()
    print(pred_values)
    gt_values = np.array(all_gt_values).flatten()
    x_values = [((5 * x)/60) for x in range(len(pred_values))]
    pred_values = pred_values[:vis_time * 12]
    gt_values = gt_values[:vis_time * 12]
    x_values = x_values[:vis_time * 12]
    plt.plot(x_values, pred_values, color="blue", label="Prediction")
    plt.plot(x_values, gt_values, color="green", label="Ground Truth")
    ax.set_xlabel("Measurement time in hours")
    ax.set_ylabel("Blood glucose values")
    plt.legend(loc='upper right')
    plt.savefig(
        "{}/pred_vs_gt_{}min_{}.pdf".format(save_path, prediction_horizon, subject_id), dpi=600, bbox_inches='tight')




# ------------------------------------------------- Main loop ----------------------------------------------------------

if __name__ == '__main__':
    data = load_csv_data(csv_path)
    plot_ts_prediction_vs_ground_truth(data, patient_id)
