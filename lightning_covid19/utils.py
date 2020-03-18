#!/usr/bin/env python
import subprocess
import sys
import random
import torch
import numpy as np


def run(cmd, stderr=subprocess.STDOUT):
    out = None
    try:
        out = subprocess.check_output([cmd], shell=True,
                                      stderr=subprocess.STDOUT, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        print(f'ERROR {e.returncode}: {cmd}\n\t{e.output}', flush=True, file=sys.stderr)
        raise e
    return out


def set_global_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
