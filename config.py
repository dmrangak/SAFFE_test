#last update march 6 8pm
from pathlib import Path

def get_config():
    return {
        "batch_size": 4,
        "num_epochs": 40,
     #   "lr": 10**-4,
        "lr": 10**-3,
        "d_model": 768,
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "experiment_name": "runs/tmodel",
        "num_heads" : 8,
        "drop_prob" : 0.1,
        "ffn_hidden" : 2048,
        "num_layers" : 1,
        "kn_vocab_size":1214,
        "sequence_length_audio":50,
        "sequence_length_image":1214
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])