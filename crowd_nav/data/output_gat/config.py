from crowd_nav.configs.config import BaseEnvConfig, BasePolicyConfig, BaseTrainConfig, Config


class EnvConfig(BaseEnvConfig):
    def __init__(self, debug=False):
        super(EnvConfig, self).__init__(debug)


class PolicyConfig(BasePolicyConfig):
    def __init__(self, debug=False):
        super(PolicyConfig, self).__init__(debug)
        self.name = 'model_predictive_rl'

        # gcn
        self.gcn.num_layer = 2
        self.gcn.X_dim = 32
        self.gcn.similarity_function = 'embedded_gaussian'
        self.gcn.layerwise_graph = False
        self.gcn.skip_connection = True

        self.model_predictive_rl = Config()
        self.model_predictive_rl.linear_state_predictor = False
        self.model_predictive_rl.planning_depth = 1
        self.model_predictive_rl.planning_width = 1
        self.model_predictive_rl.do_action_clip = False
        self.model_predictive_rl.motion_predictor_dims = [64, 5]
        self.model_predictive_rl.value_network_dims = [32, 100, 100, 1]
        self.model_predictive_rl.share_graph_model = False

        self.use_local_map_semantic = True
        self.use_Ours = True
        self.RGL_origin = True


        # hgt
        self.use_time = False
        self.time_window_size = 3

        self.hgt.conv_name = "gat"
        self.hgt.n_hid = 32
        self.hgt.n_head = 2
        self.hgt.n_layers = 2
        self.hgt.use_RTE = False
        self.hgt.X_dim_v = 256
        self.hgt.X_dim_s = 32
        self.hgt.value_network_dims = [256, 128, 64, 1]
        self.hgt.motion_predictor_dims = [32, 5]


class TrainConfig(BaseTrainConfig):
    def __init__(self, debug=False):
        super(TrainConfig, self).__init__(debug)

        self.train.freeze_state_predictor = False
        self.train.detach_state_predictor = False
        self.train.reduce_sp_update_frequency = False
        # We reuse the same variable for epsilon and for boltzmann temperature
        self.train.exploration_alg = "random_encoder"
        self.train.beta = 0.1
        self.train.schedule = "constant"
        self.train.rho = 0.0
        self.train.knn = 3
