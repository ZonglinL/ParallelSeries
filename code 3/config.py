import ml_collections
import torch


def get_config():

    config = ml_collections.ConfigDict()

    # transformer settings
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 2048
    config.transformer.num_heads = 8
    config.transformer.num_layers = 2
    config.transformer.dropout = 0.01

    # embedding settings
    config.embeddings = ml_collections.ConfigDict()
    config.embeddings.in_channels = 1
    config.embeddings.out_channels = 512
    config.embeddings.conv_len = 3
    # config.embeddings.kernel_size = 3

    # basic settings

    config.window = 72 * 7
    config.decoder_window = 24 * 7 * 1
    config.to_predict = 24 * 7 * 2
    config.dropout = 0.05
    config.criterion = torch.nn.MSELoss()
    config.classifier = 'token'
    config.representation_size = None
    config.init_type = 'normal'
    config.eval_batch_size = 64
    config.train_batch_size = 32
    config.epoch = 100
    config.nesterov = False
    config.repeats = 1
    config.multi_conv = False
    config.data_name = 'ETTh1'
    config.sparse = 'prob'
    config.factor = 5
    config.scale = True
    config.train_share = 0.9
    config.val = False
    config.decoder_layers = 1
    config.root = '.\_results'
    config.scheduler = 'On'

    config.train_dir = r'C:\Users\SssaK\Desktop\Paper\train'
    config.test_dir = r'C:\Users\SssaK\Desktop\Paper\test'
    config.output_dir = r'C:\Users\SssaK\Desktop\Paper\output'

    if config.data_name == 'energy':
        config.data_dir = r'C:\Users\SssaK\Desktop\Paper\energydata_complete.csv'
        config.feature_dim = 25
        config.train_len = 17761   # math.floor(data.shape[0]*0.9)
        config.learning_rate = 5e-6
    elif config.data_name == 'pm25':
        config.data_dir = r'C:\Users\SssaK\Desktop\Paper\pm25_rawdata.csv'
        config.feature_dim = 6
        config.train_len = 37581   # math.floor(data.shape[0]*0.9)
        config.learning_rate = 1e-6
    elif config.data_name == 'sml':
        config.data_dir = r'C:\Users\SssaK\Desktop\Paper\sml_rawdata.csv'
        config.feature_dim = 13
        config.train_len = 3723   # math.floor(data.shape[0]*0.9)
        config.learning_rate = 3e-5
    elif config.data_name == 'ETTh1':
        config.data_dir = r'C:\Users\SssaK\Desktop\Paper\ETT1.csv'
        config.feature_dim = 6
        config.train_len = 1   # math.floor(data.shape[0]*0.9)
        config.learning_rate = 1e-4
    elif config.data_name == 'ETTh2':
        config.data_dir = r'C:\Users\SssaK\Desktop\Paper\ETT2.csv'
        config.feature_dim = 6
        config.train_len = 17421   # math.floor(data.shape[0]*0.9)
        config.learning_rate = 5e-3
    elif config.data_name == 'ETTm1':
        config.data_dir = r'C:\Users\SssaK\Desktop\Paper\ETTm1.csv'
        config.feature_dim = 6
        config.train_len = 69681  # math.floor(data.shape[0]*0.9)
        config.learning_rate = 3e-3
    elif config.data_name == 'weather':
        config.data_dir = r'C:\Users\SssaK\Desktop\Paper\wth_complete.csv'
        config.feature_dim = 11
        config.train_len = 31557  # math.floor(data.shape[0]*0.9)
        config.learning_rate = 1e-4

    config.weight_decay = .00
    config.momentum = .9

    config.device = torch.device(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    return config
