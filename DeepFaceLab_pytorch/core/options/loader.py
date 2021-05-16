from pathlib import Path
import yaml

class DictAsMember(object):
    def __init__(self, dict):
        self.dictionary = dict
        
    def __getattr__(self, name):
        value = self.dictionary[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value
    
    def __getstate__(self):
        return self.dictionary

def read_yaml(filepath):
    config = yaml.load(Path(filepath).read_bytes(), Loader=yaml.FullLoader)
    return DictAsMember(config)

def write_yaml(config, filepath):
    with Path(filepath).open("w") as fp:
        yaml.dump(config.dictionary, fp, sort_keys=False, indent=4)