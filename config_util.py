import yaml
from collections.abc import MutableMapping

import operator

def getitem(obj, key, default=None):
    try:
        return operator.getitem(obj, key)
    except (KeyError, IndexError, TypeError):
        return default

class Config(MutableMapping):
    def __init__(self, yaml_file=None, **kwargs):
        self._data = {}
        
        # Load from YAML file if provided
        if yaml_file:
            with open(yaml_file, "r") as f:
                self._data.update(yaml.safe_load(f) or {})
        
        # Merge additional keyword arguments
        self.merge(kwargs)
    
    def merge(self, other):
        """Merges another dictionary or Config instance into the current config."""
        if isinstance(other, Config):
            other = other._data
        
        def recursive_merge(target, source):
            for key, value in source.items():
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    recursive_merge(target[key], value)
                else:
                    target[key] = value
        
        recursive_merge(self._data, other)
    
    # def __getattr__(self, name):
    #     try:
    #         return self._data[name]
    #     except KeyError:
    #         raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    # def __setattr__(self, name, value):
    #     if name == "_data":
    #         super().__setattr__(name, value)
    #     else:
    #         self._data[name] = value
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __setitem__(self, key, value):
        self._data[key] = value
    
    def __delitem__(self, key):
        del self._data[key]
    
    def __iter__(self):
        return iter(self._data)
    
    def __len__(self):
        return len(self._data)
    
    def __repr__(self):
        # return f"Config({self._data})"
        return f"Config:{yaml.dump(self._data)}"


if __name__ == "__main__":
    config = Config("configs/tokenizer_ps16_d12c768ps8ts2.yaml")
    print(config)
