from .cars import Cars
from .cub import CUBirds
from .SOP import SOP
from .hotels import Hotels
from .import utils
from .base import BaseDataset


_type = {
    'cars': Cars,
    'cub': CUBirds,
    'SOP': SOP,
    'hotels': Hotels
}

def load(name, root, mode, transform = None, project_dir=None):
    return _type[name](root = root, mode = mode, transform = transform, project_dir=project_dir)
    
