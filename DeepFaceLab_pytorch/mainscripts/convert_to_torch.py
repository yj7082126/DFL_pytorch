from pathlib import Path
import pickle
import yaml

import torch
from core.models import models

face_dict = {
    "f" : "full_face", 
    "wf" : "whole_face"
}

def set_weights(submodel, weight_name):
    state_dict = submodel.state_dict()
    
    if weight_name.is_file():
        weight = pickle.loads(weight_name.read_bytes())

        for key in state_dict.keys():
            keylist = key.split(".")
            for i in range(1, len(keylist)):
                if keylist[i].isnumeric():
                    keylist[i] = "_" + keylist[i]
                else:
                    keylist[i] = "/" + keylist[i]
            keylist = "".join(keylist) + ":0"

            new_value = weight[keylist]
            if "conv" in keylist and "weight" in keylist:
                new_value = torch.from_numpy(new_value).permute(3,2,0,1)
            elif "dense" in keylist and "weight"  in keylist:
                new_value = torch.from_numpy(new_value).permute(1,0)
            else:
                new_value = torch.from_numpy(new_value)

            state_dict[key] = new_value
    return state_dict

def converter(origin_path, torch_path, config_name, model_name="new"):
    origin_path = Path(origin_path)
    torch_path  = Path(torch_path)
    torch_path.mkdir(parents=True, exist_ok=True)
    config_path = origin_path.joinpath(config_name)
    options_dict = pickle.loads(config_path.read_bytes())

    resolution = options_dict['resolution']
    face_type = face_dict.get(options_dict['face_type'], "whole_face")
    model_type = options_dict['archi']
    ae_dims = options_dict['ae_dims']
    e_dims = options_dict['e_dims']
    d_dims = options_dict['d_dims']
    d_mask_dims = options_dict['d_mask_dims']
    if "-" in model_type:
        likeness = 'u' in model_type.split("-")[-1]
        double_res = 'd' in model_type.split("-")[-1]
    else:
        likeness = 'u' in model_type
        double_res = False

    if model_type.startswith('df'):
        model = models.DF(resolution, ae_dims, e_dims, d_dims, d_mask_dims, 
                            likeness=likeness, double_res=double_res)
    else:
        model = models.LIAE(resolution, ae_dims, e_dims, d_dims, d_mask_dims, 
                            likeness=likeness, double_res=double_res)

    for name, submodel in model.named_children():
        print(f"Converting : {name}")
        submodel.load_state_dict(set_weights(submodel, origin_path.joinpath(f"{model_name}_SAEHD_{name}.npy")))

    model_dict = {
        "resolution"  : int(resolution),
        "face_type"   : face_type, 
        "model_type"  : model_type,
        "ae_dims"     : int(ae_dims),
        "e_dims"      : int(e_dims),
        "d_dims"      : int(d_dims),
        "d_mask_dims" : int(d_mask_dims),
        "likeness"    : likeness, 
        "double_res"  : double_res
    }
    with torch_path.joinpath("model_opt.yaml").open("w") as fp:
        yaml.dump(model_dict, fp, sort_keys=False, indent=4)

    for name, submodel in model.named_children():
        torch.save(submodel.state_dict(), torch_path.joinpath(f"{model_name}_SAEHD_{name}.pkl"))

    print("Conversion finished")