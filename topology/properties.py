import numpy as np
from pathlib import Path
def get_project_root():
    """
    Return the root of the project.
    """
    from pathlib import Path
    return Path(__file__).parent

project_root = Path(__file__).parent
experiment_directory = "{}/.experiments".format(Path(__file__).parent.parent)
int_dtype = np.int32
float_dtype = np.float64
NUMPY_INT_DTYPE = np.int32
NUMPY_FLOAT_DTYPE = np.float64