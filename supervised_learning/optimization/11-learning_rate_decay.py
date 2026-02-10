#!/usr/bin/env python3
"""
Learning Rate Decay Module
"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay.
    """
    decay_factor = 1 + decay_rate * (global_step // decay_step)

    alpha_updated = alpha / decay_factor
    return alpha_updated
