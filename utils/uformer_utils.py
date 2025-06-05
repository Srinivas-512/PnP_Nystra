import torch
from collections import OrderedDict

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location='cpu')

    try:
        state_dict = checkpoint["state_dict"]

        model.load_state_dict(checkpoint["state_dict"])

    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model_state_dict = model.state_dict()

        filtered_state_dict = {}

        for key, value in new_state_dict.items():
            if key in model_state_dict:
                if value.size() == model_state_dict[key].size():
                    filtered_state_dict[key] = value

        model_state_dict.update(filtered_state_dict)
        model.load_state_dict(model_state_dict)