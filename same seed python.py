# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 12:58:13 2025

@author: User
"""

import numpy as np

# Create a Generator using MT19937
bitgen = np.random.MT19937(seed=123)
rng = np.random.Generator(bitgen)

# Use this generator to sample
print(rng.uniform(0, 1, 1))  # Now uses seeded MT19937