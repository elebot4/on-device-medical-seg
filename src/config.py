"""
Poor Man's Config - borrowed from nanoGPT
Example usage:
$ python train.py config/2d_axi.py --batch_size=4 --learning_rate=0.001
"""

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        # assume it's the name of a config file
        assert not arg.startswith('--')
        config_file = arg
        print(f"Loading config from {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # assume it's a --key=value override
        assert arg.startswith('--')
        key, val = arg.split('=', 1)
        key = key[2:]  # remove --
        
        # attempt to eval it (e.g. if bool, number, or etc)
        try:
            # handle special cases
            if val.lower() in ('true', 'false'):
                val = val.lower() == 'true'
            else:
                val = literal_eval(val)
        except (ValueError, SyntaxError):
            pass  # if that's not the case, just use the string
        
        # cross fingers this is a valid override
        exec(f'{key} = {repr(val)}')
        print(f"Override: {key} = {repr(val)}")