import numpy as np
import tensorflow as tf
from evaluation.evaluation import compute_metrics_multi_step
from rl.training import rl_training_loop


class RLActorDNNPruner:
    def __init__(self, agent, environments, train_data, test_data, fine_tuning_settings, log_dir, info=None):
        if info is None:
            info = {
                "pruning_method": "prune_low_magnitude",
                "pruning_rate": 0.5,
                "pruning_scope": "layer-wise",
                "networks": ["input_encoder", "output_decoder"],
                "fine_tune": True,
            }

        self.train_data = train_data
        self.test_data = test_data
        self.data_summary = fine_tuning_settings["data_summary"]
        self.pred_horizon = fine_tuning_settings["forecasting_steps"]
        self.agent = agent
        self.network = agent.policy
        self.environments = environments
        self.env_implementation = fine_tuning_settings["env_implementation"]
        self.fine_tuning_settings = fine_tuning_settings
        self.log_dir = log_dir
        self.info = info

    def prune_model(self):
        input_encoder_pruning_mask, output_decoder_pruning_mask = [], []
        for net in self.info["networks"]:
            if net == "input_encoder":
                input_encoder_pruning_mask = self.prune_input_encoder()
            elif net == "output_decoder":
                output_decoder_pruning_mask = self.prune_output_decoder()
            else:
                raise ValueError("Unknown network: {}".format(net))

        if self.info["fine_tune"]:
            self.fine_tune(input_encoder_pruning_mask, output_decoder_pruning_mask)

        test_mae, test_mse, test_rmse = compute_metrics_multi_step(
            env=self.environments['eval_env'],
            policy=self.network,
            env_implementation=self.env_implementation,
            data_summary=self.data_summary,
            ts_data=self.test_data,
            pred_horizon=self.pred_horizon,
            step=0,
            log_dir=self.log_dir,
        )
        test_results = {
            "test_mae": test_mae,
            "test_mse": test_mse,
            "test_rmse": test_rmse,
        }
        pruning_masks = {
            "input_encoder": input_encoder_pruning_mask,
            "output_decoder": output_decoder_pruning_mask,
        }

        return test_results

    def prune_input_encoder(self):
        pruning_masks = {}
        for input_enc_layer in self.network._actor_network.layers[0].layers[0].layers:
            if isinstance(input_enc_layer, tf.keras.layers.Dense):
                init_weights, init_bias = input_enc_layer.get_weights()
                total_num_weights = np.prod(init_weights.shape)
                pruning_mask = np.ones(init_weights.shape)
                # sort weights by magnitude (low to high)
                sorted_weights = np.sort(init_weights).flatten()
                pruning_weight_values = sorted_weights[:int(np.floor(total_num_weights * self.info["pruning_rate"]))]
                # get the indices in init_weights with values in pruning_weight_values
                pruning_indices = np.where(np.isin(init_weights, pruning_weight_values))
                # set weights of the layer to zero at the indices
                init_weights[pruning_indices] = 0
                input_enc_layer.set_weights([init_weights, init_bias])
                # set elements in pruning mask to zero at the indices
                pruning_mask[pruning_indices] = 0
                pruning_masks[input_enc_layer.name] = pruning_mask

        return pruning_masks

    def prune_output_decoder(self):
        pruning_masks = {}
        for output_dec_layer in self.network._actor_network.layers[0].layers:
            if isinstance(output_dec_layer, tf.keras.layers.Dense):
                init_weights, init_bias = output_dec_layer.get_weights()
                total_num_weights = np.prod(init_weights.shape)
                pruning_mask = np.ones(init_weights.shape)
                # sort weights by magnitude (low to high)
                sorted_weights = np.sort(init_weights).flatten()
                pruning_weight_values = sorted_weights[:int(np.floor(total_num_weights * self.info["pruning_rate"]))]
                # get the indices in init_weights with values in pruning_weight_values
                pruning_indices = np.where(np.isin(init_weights, pruning_weight_values))
                # set weights of the layer to zero at the indices
                init_weights[pruning_indices] = 0
                output_dec_layer.set_weights([init_weights, init_bias])
                # set elements in pruning mask to zero at the indices
                pruning_mask[pruning_indices] = 0
                pruning_masks[output_dec_layer.name] = pruning_mask

        return pruning_masks

    def fine_tune(self, input_encoder_pruning_mask, output_decoder_pruning_mask):
        rl_training_loop(
            self.log_dir,
            self.environments['train_env'],
            self.environments['train_env_eval'],
            self.environments['eval_env'],
            self.environments['eval_env_train'],
            self.agent,
            self.train_data,
            self.test_data,
            self.fine_tuning_settings['file_writer'],
            self.fine_tuning_settings['setup'],
            self.pred_horizon,
            self.fine_tuning_settings['rl_algorithm'],
            self.fine_tuning_settings['total_train_time_h'],
            self.fine_tuning_settings['total_eval_time_h'],
            self.fine_tuning_settings['max_attribute_val'],
            self.fine_tuning_settings['num_iter'],
            self.data_summary,
            self.env_implementation,
            self.fine_tuning_settings['multi_task'],
            max_train_steps=self.fine_tuning_settings['max_train_steps'],
            pruning=True,
            pruning_masks={
                'input_encoder': input_encoder_pruning_mask,
                'output_decoder': output_decoder_pruning_mask,
            }
        )
