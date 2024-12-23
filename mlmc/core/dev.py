import numpy as np

from mlmc.core.estimators import standard_mc
from mlmc.core.options import AsianOption

nsamp = 1000
h = 0.1
option = AsianOption()
result = standard_mc(nsamp, h, option)