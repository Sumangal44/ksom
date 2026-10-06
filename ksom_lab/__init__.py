"""ksom-lab — a clean, beautiful Kohonen Self-Organizing Map library."""

from .ksom_basic import BasicKSOM
from .som import SOM, TrainingHistory

__all__ = ["SOM", "TrainingHistory", "BasicKSOM"]
__version__ = "1.0.0"
