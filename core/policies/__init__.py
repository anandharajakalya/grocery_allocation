"""
Init
"""

from .base_policy import BasePolicy


# --- Simple UCB policies
from .ucb import UCB
from .ucbv_tuned import UCBVtuned

# From [Baransi et al, 2014]
from .besa import BESA