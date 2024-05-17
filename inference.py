#%%
import os
import io
from PIL import Image
import torch
import argparse
import importlib

import modules
from modules.evaluation import evaluate
from modules.utility import set_random_seed

import warnings
warnings.filterwarnings('ignore')
#%%
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("../wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

project = "XXX" # put your WANDB project name
entity = "XXX" # put your WANDB username

run = wandb.init(
    project=project, 
    entity=entity, 
    tags=["inference"], # put tags of this python project
)
#%%
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--imputation', default=False, type=str2bool)
    
    parser.add_argument('--ver', type=int, default=0, 
                        help='model version number')
    parser.add_argument('--dataset', type=str, default='abalone', 
                        help="""
                        Dataset options: 
                        covtype, loan, kings, banknote, concrete, 
                        redwine, whitewine, breast, letter, abalone
                        """)
    parser.add_argument("--missing_type", default="None", type=str,
                        help="how to generate missing: None(complete data), MCAR, MAR, MNARL, MNARQ") 
    parser.add_argument("--missing_rate", default=0.3, type=float,
                        help="missing rate") 
    
    parser.add_argument("--tau", default=1, type=float,
                        help="user defined temperature for privacy controlling") 
    parser.add_argument("--bins", default=50, type=int,
                        help="the number of bins used for quantization")
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    
    """model load"""
    base_name = f"{config['missing_type']}_{config['bins']}_{config['dataset']}"
    if config["missing_type"] != "None":
        base_name = f"{config['missing_rate']}_" + base_name
    model_name = f"MaCoDE_{base_name}"
    artifact = wandb.use_artifact(
        f"{project}/{model_name}:v{config['ver']}",
        type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    model_name = [x for x in os.listdir(model_dir) if x.endswith(f"{config['seed']}.pth")][0]
    
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    set_random_seed(config["seed"])
    wandb.config.update(config)
    #%%
    dataset_module = importlib.import_module('datasets.preprocess')
    importlib.reload(dataset_module)
    CustomDataset = dataset_module.CustomDataset

    """dataset"""
    train_dataset = CustomDataset(
        config,
        train=True)
    test_dataset = CustomDataset(
        config,
        EmpiricalCDFs=train_dataset.EmpiricalCDFs,
        train=False)
    #%%
    model_module = importlib.import_module("modules.model")
    importlib.reload(model_module)
    model = getattr(model_module, "MaCoDE")(
        config, train_dataset.EncodedInfo, device
    ).to(device)

    if config["cuda"]:
        model.load_state_dict(
            torch.load(
                model_dir + "/" + model_name
            )
        )
    else:
        model.load_state_dict(
            torch.load(
                model_dir + "/" + model_name,
                map_location=torch.device("cpu"),
            )
        )
    model.eval()
    #%%
    count_parameters = lambda model: sum(p.numel() for p in model.parameters())
    num_params = count_parameters(model)
    print(f"Number of Parameters: {num_params / 1000000:.1f}M")
    wandb.log({"Number of Parameters": num_params / 1000000})
    #%%
    n = len(train_dataset.raw_data)
    syndata = model.generate_synthetic_data(n, train_dataset, config["tau"])
    #%%
    if config["missing_type"] == "None":
        results, fig = evaluate(syndata, train_dataset, test_dataset, config)
        results = results._asdict()
    else:
        results = {}
        results["syn_reg"] = modules.metric_MLu.MLu_reg_withmissing(
            test_dataset, syndata
        )
        results["syn_cls"] = modules.metric_MLu.MLu_cls_withmissing(
            test_dataset, syndata
        )
        
    for x, y in results.items():
        print(f"{x}: {y:.3f}")
        wandb.log({f"{x}": y})
    
    # for idx, fig in enumerate(fig):
    #     buf = io.BytesIO()
    #     fig.savefig(buf, format='png')
    #     buf.seek(0)
    #     image = Image.open(buf)
    #     wandb.log({f'Marginal Histogram {idx}': wandb.Image(image)})
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
    #%% 
if __name__ == "__main__":
    main()
#%%