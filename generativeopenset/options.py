import os
import subprocess
import json
from pprint import pprint


def save_options(options):
    # Include the version of the code that saved the options
    # (in case the meaning of an option changes in the future)
    if 'version' not in options:
        options['version'] = get_code_version()
    if not os.path.exists(options['result_dir']):
        print("Creating result directory {}".format(options['result_dir']))
        os.makedirs(options['result_dir'])

    filename = os.path.join(options['result_dir'], 'params.json')
    with open(filename, 'w') as fp:
        print("Saving options to {}".format(filename))
        to_save = options.copy()
        # Do not save result_dir; always read it from the command line
        del to_save['result_dir']
        json.dump(to_save, fp, indent=2, sort_keys=True)


def load_options(options):
    print("{} with options:".format(options['result_dir']))

    param_path = get_param_path(options['result_dir'])
    old_opts = json.load(open(param_path))

    options.update(old_opts)
    options['result_dir'] = os.path.expanduser(options['result_dir'])

    pprint(options)
    return options


import os



def get_param_path(result_dir):
    params_path = os.path.join(result_dir, 'params.json')
    default_params_path = os.path.join(result_dir, 'default_params.json')

    # Check for params.json
    if os.path.exists(params_path):
        return params_path

    # Check for default_params.json
    if os.path.exists(default_params_path):
        return default_params_path

    # If neither file is found, print all files in the directory
    print(f"Could not find params.json or default_params.json in {result_dir}. Listing all files in the directory:")
    for file in os.listdir(result_dir):
        print(file)

    raise ValueError(f"Could not find params.json or default_params.json in {result_dir}")



def get_current_epoch(result_dir):
    checkpoints_path = os.path.join(result_dir, 'checkpoints')
    filenames = os.listdir(checkpoints_path)
    model_filenames = [f for f in filenames if f.endswith('.pth')]
    if not model_filenames:
        return 0
    def filename_to_epoch(filename):
        tokens = filename.rstrip('.pth').split('_')
        try:
            return int(tokens[-1])
        except ValueError:
            return 0
    return max(filename_to_epoch(f) for f in model_filenames)


def get_code_version():
    cwd = os.path.dirname(__file__)
    try:
        output = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd)
    except subprocess.CalledProcessError:
        print("Warning: Failed git rev-parse, current code version unknown")
        return "unknown"
    return output.strip().decode('utf-8')
