from pathlib import Path
import torch
import json
from core.options.loader import write_yaml

def save_weights(model_path, model, log_history=None, model_name="new", 
                 config=None):
    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)
    
    for name, param in model.named_children():
        torch.save(
            param.state_dict(), 
            model_path.joinpath(f"{model_name}_SAEHD_{name}.pkl")
        )

    if log_history is not None:
        log_path = model_path.joinpath("history.json")
        log_str = json.dumps(log_history)
        log_path.write_bytes(log_str.encode())
    if config is not None:
        write_yaml(config, model_path.joinpath("train_opt.yaml"))



def load_weights(model_path, model, model_name="new", finetune_start=False):
    model_path = Path(model_path)

    for name, param in model.named_children():
        chkpt = torch.load(model_path.joinpath(f"{model_name}_SAEHD_{name}.pkl"))
        if not (finetune_start and "inter" in name):
            print(f"Loading {name}")
            param.load_state_dict(chkpt, strict=False)

    log_path = model_path.joinpath("history.json")
    if log_path.is_file():
        log_history = json.loads(log_path.read_bytes())
    else:
        log_history = {
            "current_iters" : 0, 
            "src_loss_history" : {}, 
            "dst_loss_history" : {} 
        }
    return model, log_history