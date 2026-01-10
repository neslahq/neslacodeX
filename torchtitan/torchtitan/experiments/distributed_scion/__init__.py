# Copied from the torchtitan pull request: https://github.com/pytorch/torchtitan/pull/1630/files#

from .abstract_disco import AbstractDiSCO
from .disco import Disco
from .utils import (
    create_disco_optimizer_kwargs_from_optimizer_config,
    create_disco_param_groups,
    remove_orig_mod_and_weight_for_p_name,
)
