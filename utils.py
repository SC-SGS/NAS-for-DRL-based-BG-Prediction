import json
import numpy as np
import tensorflow as tf
def extract_hp_dicts_from_hpo_log(path_to_log):
    # read log file as str
    with open(path_to_log, 'r') as f:
        log_str = f.read()
    # split log file into individual hpo runs (dicts)
    hpo_dicts = log_str.split("parameters:")[1:]
    hpo_dicts = [(x.split(". Best")[0]) for x in hpo_dicts]
    # use jason to convert str to dict
    hpo_dicts = [json.loads(x.strip().replace("'", "\"")) for x in hpo_dicts]

    return hpo_dicts

def calculate_network_complexity(rl_network):
    total_complexity = 0
    input_enc_complexity, output_enc_complexity, rnn_complexity, final_layer_complexity = 0, 0, 0, 0
    for tv in rl_network.trainable_variables:
        if len(tv.shape) > 0:
            # calculate number of parameters for current layer from shape
            current_complexity = np.prod(list(tv.shape))
        else:
            current_complexity = 1

        total_complexity += current_complexity

        if "EncodingNetwork" in tv.name:
            input_enc_complexity += current_complexity
        if "dynamic_unroll" in tv.name or "gru_cell" in tv.name:
            rnn_complexity += current_complexity
        if "ActorDistributionRnnNetwork/dense_" in tv.name:
            output_enc_complexity += current_complexity
        if "TanhNormalProjectionNetwork" in tv.name:
            final_layer_complexity += current_complexity

    network_complexities = {
        "input_enc_complexity": input_enc_complexity,
        "output_enc_complexity": output_enc_complexity,
        "rnn_complexity": rnn_complexity,
        "final_layer_complexity": final_layer_complexity,
        "total_complexity": total_complexity
    }

    return network_complexities

def convert_dict_to_hp_format(hp_dict):
    hp = {'actor_net': {}, 'critic_net': {}}
    conditional_hp = []
    for key, value in hp_dict.items():
        if key in conditional_hp:
            continue

        if "actor_" in key:
            current_net = 'actor_net'
        elif "critic_" in key:
            current_net = 'critic_net'
        else:
            hp[key] = value
            continue

        if "_neurons" in key or "_layers" in key:
            if "_neurons" in key:
                dict_key = key.replace(current_net.split("_")[0] + "_", "").replace("_neurons", "")
                layers_key = key.replace("_neurons", "_layers")
                conditional_hp.append(layers_key)
                if key == "actor_input_fc_layer_params_neurons" or key == "actor_output_fc_layer_params_neurons":
                    num_layers = hp_dict[layers_key]
                    if num_layers == 2:# and key == "actor_output_fc_layer_params_neurons":
                        hp_val = tuple(
                            [x * value for x in range(1, num_layers + 1)]
                        )
                        # hp_val = tuple([value for _ in range(num_layers)])
                    # elif num_layers == 3 and key == "actor_output_fc_layer_params_neurons":
                    #         hp_val = tuple([value for _ in range(num_layers)])
                    else:
                        hp_val = tuple(
                            [
                                (x % num_layers if x % num_layers != 0 else 1)
                                * value for x in range(1, num_layers + 1)
                            ]
                        )
                else:
                    hp_val = tuple([value for _ in range(hp_dict[layers_key])])
            elif "_layers" in key:
                dict_key = key.replace(current_net.split("_")[0] + "_", "").replace("_layers", "")
                neurons_key = key.replace("_layers", "_neurons")
                conditional_hp.append(neurons_key)
                if key == "actor_input_fc_layer_params_layers" or key == "actor_output_fc_layer_params_layers":
                    if value == 2:# and key == "actor_output_fc_layer_params_layers":
                        hp_val = tuple(
                            #[(x % value if x % value != 0 else 1) * hp_dict[neurons_key] for x in range(value)]
                            [x * hp_dict[neurons_key] for x in range(1, value + 1)]
                        )
                        # hp_val = tuple(
                        #     [hp_dict[neurons_key] for _ in range(value)]
                        # )
                    # elif value == 3 and key == "actor_output_fc_layer_params_layers":
                    #     hp_val = tuple(
                    #         [hp_dict[neurons_key] for _ in range(value)]
                    #     )
                    else:
                        hp_val = tuple(
                            [(x % value if x % value != 0 else 1)
                             * hp_dict[neurons_key]
                             for x in range(1, value + 1)]
                        )
                else:
                    hp_val = tuple([hp_dict[neurons_key] for _ in range(value)])
            else:
                raise ValueError("Unknown key.")
        elif "activation_fn" in key:
            dict_key = key.replace(current_net.split("_")[0] + "_", "")
            if value == "relu":
                hp_val = tf.keras.activations.relu
            elif value == "tanh":
                hp_val = tf.keras.activations.tanh
            elif value == "sigmoid":
                hp_val = tf.keras.activations.sigmoid
            else:
                raise ValueError("Unknown activation function.")
        elif "cell_type" in key:
            dict_key = key.replace(current_net.split("_")[0] + "_", "")
            hp_val = value
        else:
            dict_key = key.replace(current_net.split("_")[0] + "_", "")
            hp_val = (value,)

        hp[current_net][dict_key] = hp_val

    return hp


'''
CLARKE ERROR GRID ANALYSIS      ClarkeErrorGrid.py

Need Matplotlib Pyplot


The Clarke Error Grid shows the differences between a blood glucose predictive measurement and a reference measurement,
and it shows the clinical significance of the differences between these values.
The x-axis corresponds to the reference value and the y-axis corresponds to the prediction.
The diagonal line shows the prediction value is the exact same as the reference value.
This grid is split into five zones. Zone A is defined as clinical accuracy while
zones C, D, and E are considered clinical error.

Zone A: Clinically Accurate
    This zone holds the values that differ from the reference values no more than 20 percent
    or the values in the hypoglycemic range (<70 mg/dL).
    According to the literature, values in zone A are considered clinically accurate.
    These values would lead to clinically correct treatment decisions.

Zone B: Clinically Acceptable
    This zone holds values that differe more than 20 percent but would lead to
    benign or no treatment based on assumptions.

Zone C: Overcorrecting
    This zone leads to overcorrecting acceptable BG levels.

Zone D: Failure to Detect
    This zone leads to failure to detect and treat errors in BG levels.
    The actual BG levels are outside of the acceptable levels while the predictions
    lie within the acceptable range

Zone E: Erroneous treatment
    This zone leads to erroneous treatment because prediction values are opposite to
    actual BG levels, and treatment would be opposite to what is recommended.


SYNTAX:
        plot, zone = clarke_error_grid(ref_values, pred_values, title_string)

INPUT:
        ref_values          List of n reference values.
        pred_values         List of n prediciton values.
        title_string        String of the title.

OUTPUT:
        plot                The Clarke Error Grid Plot returned by the function.
                            Use this with plot.show()
        zone                List of values in each zone.
                            0=A, 1=B, 2=C, 3=D, 4=E

EXAMPLE:
        plot, zone = clarke_error_grid(ref_values, pred_values, "00897741 Linear Regression")
        plot.show()

References:
[1]     Clarke, WL. (2005). "The Original Clarke Error Grid Analysis (EGA)."
        Diabetes Technology and Therapeutics 7(5), pp. 776-779.
[2]     Maran, A. et al. (2002). "Continuous Subcutaneous Glucose Monitoring in Diabetic
        Patients" Diabetes Care, 25(2).
[3]     Kovatchev, B.P. et al. (2004). "Evaluating the Accuracy of Continuous Glucose-
        Monitoring Sensors" Diabetes Care, 27(8).
[4]     Guevara, E. and Gonzalez, F. J. (2008). Prediction of Glucose Concentration by
        Impedance Phase Measurements, in MEDICAL PHYSICS: Tenth Mexican
        Symposium on Medical Physics, Mexico City, Mexico, vol. 1032, pp.
        259261.
[5]     Guevara, E. and Gonzalez, F. J. (2010). Joint optical-electrical technique for
        noninvasive glucose monitoring, REVISTA MEXICANA DE FISICA, vol. 56,
        no. 5, pp. 430434.


Made by:
Trevor Tsue
7/18/17

Based on the Matlab Clarke Error Grid Analysis File Version 1.2 by:
Edgar Guevara Codina
codina@REMOVETHIScactus.iico.uaslp.mx
March 29 2013
'''



import matplotlib.pyplot as plt


#This function takes in the reference values and the prediction values as lists and returns a list with each index corresponding to the total number
#of points within that zone (0=A, 1=B, 2=C, 3=D, 4=E) and the plot
def clarke_error_grid(ref_values, pred_values, title_string):

    #Checking to see if the lengths of the reference and prediction arrays are the same
    assert (len(ref_values) == len(pred_values)), "Unequal number of values (reference : {}) (prediction : {}).".format(len(ref_values), len(pred_values))

    #Checks to see if the values are within the normal physiological range, otherwise it gives a warning
    if max(ref_values) > 500 or max(pred_values) > 500:
        print(
            "Input Warning: the maximum reference value {} or the maximum prediction value {} exceeds the normal "
            "physiological range of glucose (<500 mg/dL).".format(max(ref_values), max(pred_values)))
    if min(ref_values) < 0 or min(pred_values) < 0:
        print(
            "Input Warning: the minimum reference value {} or the minimum prediction value {} "
            "is less than 0 mg/dL.".format(min(ref_values),  min(pred_values)))

    #Clear plot
    plt.clf()

    #Set up plot
    plt.scatter(ref_values, pred_values, marker='o', color='black', s=1)
    # plt.title(title_string + " Clarke Error Grid")
    plt.xlabel("Measured Blood Glucose Values (mg/dL)")
    plt.ylabel("Predicted Blood Glucose Values (mg/dL)")
    plt.xticks([0, 100, 200, 300, 400, 500, 600])
    plt.yticks([0, 100, 200, 300, 400, 500, 600])
    plt.gca().set_facecolor('white')

    #Set axes lengths
    plt.gca().set_xlim([0, 600])
    plt.gca().set_ylim([0, 600])
    plt.gca().set_aspect((600)/(600))

    # Plot zone lines
    plt.plot([0, 600], [0, 600], ':', c='black')  # Theoretical 45 regression line
    plt.plot([0, 175 / 3], [70, 70], '-', c='black')
    # plt.plot([175/3, 320], [70, 400], '-', c='black')
    # Note: line follows the equation y=1.2x (offset c is 0)
    plt.plot([175 / 3, 600 / 1.2], [70, 600], '-', c='black')  # Replace 320 with 400/1.2 because 100*(400 - 400/1.2)/(400/1.2) =  20% error
    plt.plot([70, 70], [84, 600], '-', c='black')
    plt.plot([0, 70], [180, 180], '-', c='black')
    # Note: line follows the equation y=1x + 110 (offset c is 110)
    plt.plot([70, 490], [180, 600], '-', c='black')
    # plt.plot([70, 70], [0, 175/3], '-', c='black')
    plt.plot([70, 70], [0, 56], '-', c='black')  # Replace 175.3 with 56 because 100*abs(56-70)/70) = 20% error
    # plt.plot([70, 400],[175/3, 320],'-', c='black')
    # Note: line follows the equation y=0.8333x + 40 (offset c is 40)
    plt.plot([70, 600], [56, (0.8333*600) + 40], '-', c='black')
    plt.plot([180, 180], [0, 70], '-', c='black')
    plt.plot([180, 600], [70, 70], '-', c='black')
    plt.plot([240, 240], [70, 180], '-', c='black')
    plt.plot([240, 600], [180, 180], '-', c='black')
    plt.plot([130, 180], [0, 70], '-', c='black')

    # Add zone titles
    plt.text(30, 10, "A", fontsize=15)
    plt.text(400, 260, "B", fontsize=15)
    plt.text(275, 360, "B", fontsize=15)
    plt.text(155, 400, "C", fontsize=15)
    plt.text(155, 15, "C", fontsize=15)
    plt.text(30, 120, "D", fontsize=15)
    plt.text(400, 120, "D", fontsize=15)
    plt.text(30, 400, "E", fontsize=15)
    plt.text(400, 15, "E", fontsize=15)

    #Statistics from the data
    zone = [0] * 5
    for i in range(len(ref_values)):
        if (ref_values[i] <= 70 and pred_values[i] <= 70) or (pred_values[i] <= 1.2*ref_values[i] and pred_values[i] >= 0.8*ref_values[i]):
            zone[0] += 1    #Zone A

        elif (ref_values[i] >= 180 and pred_values[i] <= 70) or (ref_values[i] <= 70 and pred_values[i] >= 180):
            zone[4] += 1    #Zone E

        elif ((ref_values[i] >= 70 and ref_values[i] <= 290) and pred_values[i] >= ref_values[i] + 110) or ((ref_values[i] >= 130 and ref_values[i] <= 180) and (pred_values[i] <= (7/5)*ref_values[i] - 182)):
            zone[2] += 1    #Zone C
        elif (ref_values[i] >= 240 and (pred_values[i] >= 70 and pred_values[i] <= 180)) or (ref_values[i] <= 175/3 and pred_values[i] <= 180 and pred_values[i] >= 70) or ((ref_values[i] >= 175/3 and ref_values[i] <= 70) and pred_values[i] >= (6/5)*ref_values[i]):
            zone[3] += 1    #Zone D
        else:
            zone[1] += 1    #Zone B

    return plt, zone
