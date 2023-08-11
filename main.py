import os
import gin
import optuna
import datetime
import tensorflow as tf
from data import dataset
from absl import app, logging
from rl import environment, rl_agent, training

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        logging.info(e)


def main(args):
    # parse config file
    gin.parse_config_file("config.gin")
    run()


@gin.configurable
def run(path_to_train_data="", path_to_eval_data="", normalization=False, normalization_type="min_max",
        setup="single_step", rl_algorithm="ddpg", env_implementation="tf", use_hpo=False, use_gpu=False,
        multi_task=False):
    # logging
    log_dir = "./logs/" + "log" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_writer = tf.summary.create_file_writer(log_dir)
    logging.get_absl_handler().use_absl_log_file(program_name="log", log_dir=log_dir)
    # load data set
    if multi_task:
        dataset_files = sorted(os.listdir(path_to_train_data), key=lambda index: int(index.split("-")[0]))
        patients_train_datasets = []
        patient_train_total_times = []
        for f in dataset_files:
            ts_train_data_sp, total_train_time_h = dataset.load_csv_dataset(os.path.join(path_to_train_data, f))
            patients_train_datasets.append(ts_train_data_sp)
            patient_train_total_times.append(total_train_time_h)
        ts_train_data = patients_train_datasets
        total_train_time_h = int((len(ts_train_data) * 5) / 60) + 1
    else:
        ts_train_data, total_train_time_h = dataset.load_csv_dataset(path_to_train_data)
    if path_to_eval_data != "":
        if multi_task:
            dataset_files = sorted(os.listdir(path_to_eval_data), key=lambda index: int(index.split("-")[0]))
            patients_eval_datasets = []
            patient_eval_total_times = []
            for f in dataset_files:
                ts_eval_data_sp, total_eval_time_h = dataset.load_csv_dataset(os.path.join(path_to_eval_data, f))
                patients_eval_datasets.append(ts_eval_data_sp)
                patient_eval_total_times.append(total_eval_time_h)
            ts_eval_data = patients_eval_datasets
            total_eval_time_h = int((len(ts_eval_data) * 5) / 60) + 1
        else:
            ts_eval_data, total_eval_time_h = dataset.load_csv_dataset(path_to_eval_data)
    else:
        ts_eval_data = ts_train_data
        total_eval_time_h = total_train_time_h
    if normalization:
        if multi_task:
            ts_train_data, ts_eval_data, data_summary = dataset.data_normalization_multi_patient(
                ts_train_data, ts_eval_data, normalization_type=normalization_type)
        else:
            ts_train_data, ts_eval_data, data_summary = dataset.data_normalization(
                ts_train_data, ts_eval_data, normalization_type=normalization_type)
    else:
        data_summary = {}
    # create environment
    if setup == "single_step":
        if env_implementation == "tf":
            train_env = environment.TsForecastingSingleStepTFEnv(ts_train_data, rl_algorithm, data_summary)
            train_env_eval = environment.TsForecastingSingleStepTFEnv(
                ts_train_data, rl_algorithm, data_summary, evaluation=True, max_window_count=-1)
        else:
            train_env = environment.TsForecastingSingleStepEnv(ts_train_data, rl_algorithm=rl_algorithm)
            train_env_eval = environment.TsForecastingSingleStepEnv(
                ts_train_data, evaluation=True, max_window_count=-1, rl_algorithm=rl_algorithm)
        if normalization:
            # max_attribute_val = train_env.max_attribute_val * data_summary["max"],
            max_attribute_val = dataset.undo_data_normalization_sample_wise(train_env.max_attribute_val, data_summary)
        else:
            max_attribute_val = train_env.max_attribute_val
        if path_to_eval_data != "":
            if env_implementation == "tf":
                eval_env = environment.TsForecastingSingleStepTFEnv(
                    ts_eval_data, rl_algorithm, data_summary, evaluation=True, max_window_count=-1)
                eval_env_train = environment.TsForecastingSingleStepTFEnv(ts_eval_data, rl_algorithm, data_summary)
            else:
                eval_env = environment.TsForecastingSingleStepEnv(
                    ts_eval_data, evaluation=True, rl_algorithm=rl_algorithm, max_window_count=-1)
                eval_env_train = environment.TsForecastingSingleStepEnv(ts_eval_data, rl_algorithm=rl_algorithm)
        else:
            if env_implementation == "tf":
                eval_env = environment.TsForecastingSingleStepTFEnv(
                    ts_train_data, rl_algorithm, data_summary, evaluation=True, max_window_count=-1)
                eval_env_train = environment.TsForecastingSingleStepTFEnv(ts_train_data, rl_algorithm, data_summary)
            else:
                eval_env = environment.TsForecastingSingleStepEnv(
                    ts_train_data, evaluation=True, rl_algorithm=rl_algorithm, max_window_count=-1)
                eval_env_train = environment.TsForecastingSingleStepEnv(ts_train_data, rl_algorithm)
        forecasting_steps = 1
        num_iter = train_env.max_window_count
    elif setup == "multi_step":
        if env_implementation == "tf":
            train_env = environment.TsForecastingMultiStepTFEnv(
                ts_train_data, rl_algorithm, data_summary, multi_task=multi_task)
            train_env_eval = environment.TsForecastingMultiStepTFEnv(
                ts_train_data, rl_algorithm, data_summary, multi_task=multi_task,
                evaluation=True, max_window_count=-1)
        else:
            train_env = environment.TsForecastingMultiStepEnv(ts_train_data, rl_algorithm)
            train_env_eval = environment.TsForecastingMultiStepEnv(
                ts_train_data, rl_algorithm, evaluation=True, max_window_count=-1)
        if normalization:
            max_attribute_val = dataset.undo_data_normalization_sample_wise(train_env.max_attribute_val, data_summary)
        else:
            max_attribute_val = train_env.max_attribute_val
        if path_to_eval_data != "":
            if env_implementation == "tf":
                eval_env = environment.TsForecastingMultiStepTFEnv(
                    ts_eval_data, rl_algorithm, data_summary, evaluation=True, max_window_count=-1,
                    multi_task=multi_task)
                eval_env_train = environment.TsForecastingMultiStepTFEnv(
                    ts_eval_data, rl_algorithm, data_summary, multi_task=multi_task)
            else:
                eval_env = environment.TsForecastingMultiStepEnv(ts_eval_data, rl_algorithm,
                                                                 evaluation=True, max_window_count=-1)
                eval_env_train = environment.TsForecastingMultiStepEnv(ts_eval_data, rl_algorithm)
        else:
            if env_implementation == "tf":
                eval_env = environment.TsForecastingMultiStepTFEnv(
                    ts_train_data, rl_algorithm, data_summary, evaluation=True, max_window_count=-1,
                    multi_task=multi_task)
                eval_env_train = environment.TsForecastingMultiStepTFEnv(
                    ts_train_data, rl_algorithm, data_summary, multi_task=multi_task)
            else:
                eval_env = environment.TsForecastingMultiStepEnv(ts_train_data, rl_algorithm, evaluation=True,
                                                                 max_window_count=-1)
                eval_env_train = environment.TsForecastingMultiStepEnv(ts_train_data, rl_algorithm)
        forecasting_steps = train_env.pred_horizon
        num_iter = train_env.max_window_count
    else:
        raise ValueError("Invalid setup: " + setup)

    if env_implementation != "tf":
        # get TF environment
        tf_train_env = environment.get_tf_environment(train_env)
        tf_train_env_eval = environment.get_tf_environment(train_env_eval)
        tf_eval_env = environment.get_tf_environment(eval_env)
        tf_eval_env_train = environment.get_tf_environment(eval_env_train)
    else:
        tf_train_env = train_env
        tf_train_env_eval = train_env_eval
        tf_eval_env = eval_env
        tf_eval_env_train = eval_env_train

    # set up RL agent
    if use_hpo:
        def optuna_objective(trial):
            current_hp = {
                'critic_net': {
                    'cell_type': trial.suggest_categorical('critic_cell_type', ['lstm', 'gru']),
                    'cell_size': (trial.suggest_categorical('critic_cell_size', [4, 8, 16, 32, 64, 128, 256]),),
                    'observation_fc_layer_params_dict': {
                        'neurons': trial.suggest_categorical(
                            'critic_observation_fc_layer_params_neurons', [4, 8, 16, 32, 64, 128, 256, 512]
                        ),
                        'layers': trial.suggest_categorical(
                            'critic_observation_fc_layer_params_layers',
                            [1, 2, 3]
                        ),
                    },
                    'action_fc_layer_params': (
                        trial.suggest_categorical('critic_action_fc_layer_params', [4, 8, 16, 32, 64, 128, 256]),
                    ),
                    'joint_fc_layer_params_dict': {
                        'neurons': trial.suggest_categorical(
                            'critic_joint_fc_layer_params_neurons', [4, 8, 16, 32, 64, 128, 256, 512]
                        ),
                        'layers': trial.suggest_categorical('critic_joint_fc_layer_params_layers', [1, 2, 3]),
                    },
                    'output_fc_layer_params_dict': {
                        'neurons': trial.suggest_categorical(
                            'critic_output_fc_layer_params_neurons', [4, 8, 16, 32, 64, 128, 256, 512]
                        ),
                        'layers': trial.suggest_categorical('critic_output_fc_layer_params_layers', [1, 2, 3]),
                    },
                    'activation_fn': trial.suggest_categorical(
                        'critic_activation_fn', ["relu", "tanh", "sigmoid"]
                    ),
                },
                'actor_net': {
                    'cell_type': trial.suggest_categorical('actor_cell_type', ['lstm', 'gru']),
                    'cell_size': (trial.suggest_categorical('actor_cell_size', [4, 8, 16, 32, 64, 128, 256]),),
                    'input_fc_layer_params_dict': {
                        'neurons': trial.suggest_categorical(
                            'actor_input_fc_layer_params_neurons', [4, 8, 16, 32, 64, 128, 256, 512]
                        ),
                        'layers': trial.suggest_categorical('actor_input_fc_layer_params_layers', [1, 2, 3]),
                    },
                    'output_fc_layer_params_dict': {
                        'neurons': trial.suggest_categorical(
                            'actor_output_fc_layer_params_neurons', [4, 8, 16, 32, 64, 128, 256, 512]
                        ),
                        'layers': trial.suggest_categorical('actor_output_fc_layer_params_layers', [1, 2, 3]),
                    },
                    'activation_fn': trial.suggest_categorical(
                        'actor_activation_fn', ["relu", "tanh", "sigmoid"]
                    ),
                },
                'target_update_period': trial.suggest_int('target_update_period', 1, 100),
                'target_update_tau': trial.suggest_float('target_update_tau', 0.001, 1.0),
            }
            # convert activation function strings to functions
            current_hp['critic_net']['activation_fn'] = getattr(
                tf.keras.activations, current_hp['critic_net']['activation_fn']
            )
            current_hp['actor_net']['activation_fn'] = getattr(
                tf.keras.activations, current_hp['actor_net']['activation_fn']
            )
            # convert dict entries of current hyperparameter to tuples
            current_hp['critic_net']['observation_fc_layer_params'] = tuple(
                [current_hp['critic_net']['observation_fc_layer_params_dict']['neurons'] for _ in range(
                    current_hp['critic_net']['observation_fc_layer_params_dict']['layers'])]
            )
            del current_hp['critic_net']['observation_fc_layer_params_dict']
            current_hp['critic_net']['joint_fc_layer_params'] = tuple(
                [current_hp['critic_net']['joint_fc_layer_params_dict']['neurons'] for _ in range(
                    current_hp['critic_net']['joint_fc_layer_params_dict']['layers'])]
            )
            del current_hp['critic_net']['joint_fc_layer_params_dict']
            current_hp['critic_net']['output_fc_layer_params'] = tuple(
                [current_hp['critic_net']['output_fc_layer_params_dict']['neurons'] for _ in range(
                    current_hp['critic_net']['output_fc_layer_params_dict']['layers'])]
            )
            del current_hp['critic_net']['output_fc_layer_params_dict']
            actor_input_layers = current_hp['actor_net']['input_fc_layer_params_dict']['layers']
            current_hp['actor_net']['input_fc_layer_params'] = tuple(
                [(x % actor_input_layers if x % actor_input_layers != 0 else 1)
                 * current_hp['actor_net']['input_fc_layer_params_dict']['neurons']
                 for x in range(1, actor_input_layers + 1)]
            )
            del current_hp['actor_net']['input_fc_layer_params_dict']
            actor_output_layers = current_hp['actor_net']['output_fc_layer_params_dict']['layers']
            current_hp['actor_net']['output_fc_layer_params'] = tuple(
                [(x % actor_input_layers if x % actor_input_layers != 0 else 1)
                 * current_hp['actor_net']['output_fc_layer_params_dict']['neurons']
                 for x in range(1, actor_output_layers + 1)]
            )
            del current_hp['actor_net']['output_fc_layer_params_dict']
            # set state size in RL environments
            current_state_size_factor = 2 if current_hp['actor_net']['cell_type'] == 'lstm' else 1
            current_tf_train_env = tf_train_env.get_environment_with_state_size(
                current_state_size_factor * current_hp['actor_net']['cell_size'][0]
            )
            current_tf_train_env_eval = tf_train_env_eval.get_environment_with_state_size(
                current_state_size_factor * current_hp['actor_net']['cell_size'][0]
            )
            current_tf_eval_env = tf_eval_env.get_environment_with_state_size(
                current_state_size_factor * current_hp['actor_net']['cell_size'][0]
            )
            current_tf_eval_env_train = tf_eval_env_train.get_environment_with_state_size(
                current_state_size_factor * current_hp['actor_net']['cell_size'][0]
            )
            current_agent = rl_agent.get_rl_agent(current_tf_train_env, rl_algorithm, use_gpu, hp=current_hp)
            current_train_steps = trial.suggest_int('train_steps', int(1e4), int(5e4))
            objective_metric = training.rl_training_loop(
                log_dir, current_tf_train_env, current_tf_train_env_eval, current_tf_eval_env,
                current_tf_eval_env_train, current_agent, ts_train_data, ts_eval_data, file_writer, setup,
                forecasting_steps, rl_algorithm, total_train_time_h, total_eval_time_h, max_attribute_val, num_iter,
                data_summary, env_implementation, multi_task, eval_interval=None, max_train_steps=current_train_steps,
                visualize=False, use_tb_logging=False, save_model=False, save_results=False
            )
            current_agent_model_complexity = 0
            for tv in current_agent.trainable_variables:
                if len(tv.shape) > 0:
                    current_agent_model_complexity += tv.shape[-1]
                else:
                    current_agent_model_complexity += 1
            # normalize complexity (default architecture has 32801 parameters)
            current_agent_model_complexity /= 32801
            # normalize objective metric
            objective_metric /= 20.0

            return objective_metric + current_agent_model_complexity

        study = optuna.create_study(direction='minimize')
        study.optimize(optuna_objective, n_trials=100, n_jobs=4, gc_after_trial=True)
        model_hyperparameters = study.best_params
        # reformat best hyperparameters
        best_hp = {'actor_net': {}, 'critic_net': {}}
        conditional_hp = []
        for key, value in model_hyperparameters.items():
            if key in conditional_hp:
                continue

            if "actor_" in key:
                current_net = 'actor_net'
            elif "critic_" in key:
                current_net = 'critic_net'
            else:
                best_hp[key] = value
                continue

            if "_neurons" in key or "_layers" in key:
                if "_neurons" in key:
                    dict_key = key.replace(current_net.split("_")[0] + "_", "").replace("_neurons", "")
                    layers_key = key.replace("_neurons", "_layers")
                    conditional_hp.append(layers_key)
                    if key == "actor_input_fc_layer_params_neurons" or key == "actor_output_fc_layer_params_neurons":
                        num_layers = model_hyperparameters[layers_key]
                        best_val = tuple(
                            [(x % num_layers if x % num_layers != 0 else 1) * value for x in range(num_layers)]
                        )
                    else:
                        best_val = tuple([value for _ in range(model_hyperparameters[layers_key])])
                elif "_layers" in key:
                    dict_key = key.replace(current_net.split("_")[0] + "_", "").replace("_layers", "")
                    neurons_key = key.replace("_layers", "_neurons")
                    conditional_hp.append(neurons_key)
                    if key == "actor_input_fc_layer_params_layers" or key == "actor_output_fc_layer_params_layers":
                        best_val = tuple(
                            [(x % value if x % value != 0 else 1) *
                             model_hyperparameters[neurons_key] for x in range(value)]
                        )
                    else:
                        best_val = tuple([model_hyperparameters[neurons_key] for _ in range(value)])
                else:
                    raise ValueError("Unknown key.")
            elif "activation_fn" in key:
                dict_key = key.replace(current_net.split("_")[0] + "_", "")
                if value == "relu":
                    best_val = tf.keras.activations.relu
                elif value == "tanh":
                    best_val = tf.keras.activations.tanh
                elif value == "sigmoid":
                    best_val = tf.keras.activations.sigmoid
                else:
                    raise ValueError("Unknown activation function.")
            elif "cell_type" in key:
                dict_key = key.replace(current_net.split("_")[0] + "_", "")
                best_val = value
            else:
                dict_key = key.replace(current_net.split("_")[0] + "_", "")
                best_val = (value,)

            best_hp[current_net][dict_key] = best_val

        model_hyperparameters = best_hp
        state_size_factor = 2 if model_hyperparameters['actor_net']['cell_type'] == 'lstm' else 1
        tf_train_env = tf_train_env.get_environment_with_state_size(
            state_size_factor * model_hyperparameters['actor_net']['cell_size'][0]
        )
        tf_train_env_eval = tf_train_env_eval.get_environment_with_state_size(
            state_size_factor * model_hyperparameters['actor_net']['cell_size'][0]
        )
        tf_eval_env = tf_eval_env.get_environment_with_state_size(
            state_size_factor * model_hyperparameters['actor_net']['cell_size'][0]
        )
        tf_eval_env_train = tf_eval_env_train.get_environment_with_state_size(
            state_size_factor * model_hyperparameters['actor_net']['cell_size'][0]
        )
    else:
        model_hyperparameters = None

    agent = rl_agent.get_rl_agent(tf_train_env, rl_algorithm, use_gpu, hp=model_hyperparameters)
    # save gin's operative config to a file before training
    config_txt_file = open(log_dir + "/gin_config.txt", "w+")
    config_txt_file.write("Configuration options available before training \n")
    config_txt_file.write("\n")
    config_txt_file.write(gin.operative_config_str())
    config_txt_file.close()
    # train agent on environment
    if model_hyperparameters is not None and 'train_steps' in model_hyperparameters:
        training.rl_training_loop(
            log_dir, tf_train_env, tf_train_env_eval, tf_eval_env, tf_eval_env_train, agent,ts_train_data, ts_eval_data,
            file_writer, setup, forecasting_steps, rl_algorithm, total_train_time_h, total_eval_time_h,
            max_attribute_val, num_iter, data_summary, env_implementation, multi_task,
            max_train_steps=model_hyperparameters['train_steps']
        )
    else:
        training.rl_training_loop(
            log_dir, tf_train_env, tf_train_env_eval, tf_eval_env, tf_eval_env_train, agent, ts_train_data,
            ts_eval_data, file_writer, setup, forecasting_steps, rl_algorithm, total_train_time_h, total_eval_time_h,
            max_attribute_val, num_iter, data_summary, env_implementation, multi_task
        )
    # save gin's operative config to a file after training
    config_txt_file = open(log_dir + "/gin_config.txt", "a")
    config_txt_file.write("\n")
    config_txt_file.write("Configuration options available after training \n")
    config_txt_file.write("\n")
    config_txt_file.write(gin.operative_config_str())
    config_txt_file.close()


if __name__ == '__main__':
    app.run(main)
