# Data transfer from pytorch to DAPHNE and back, via files.

import numpy as np
import torch
from daphne.context.daphne_context import DaphneContext

t1 = torch.from_numpy(np.array([-1.0, -2.0, -3.0], dtype=np.float64).reshape(-1, 1))

dctx = DaphneContext()

(dctx.from_pytorch(t1, shared_memory=False).print().compute())