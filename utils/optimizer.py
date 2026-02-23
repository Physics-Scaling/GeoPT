import os
import torch
import numpy as np
import math


# ---------------------------
# Layer-wise LR Decay helpers
# ---------------------------
def _is_no_weight_decay(param_name: str, param: torch.nn.Parameter) -> bool:
    # 1D params are typically norm weights, and bias should not decay
    if param.ndim == 1:
        return True
    if param_name.endswith(".bias"):
        return True
    lname = param_name.lower()
    if any(k in lname for k in ["norm", "ln", "layernorm", "rmsnorm"]):
        return True
    return False


def _infer_num_layers_from_names(names):
    """
    Infer number of layers by scanning parameter names like 'blocks.0.', 'layers.11.' etc.
    Returns max_layer_id+1, or 0 if not found.
    """
    max_id = -1
    pat = re.compile(r"(blocks|layers)\.(\d+)\.")
    for n in names:
        m = pat.search(n)
        if m:
            max_id = max(max_id, int(m.group(2)))
    return max_id + 1 if max_id >= 0 else 0
