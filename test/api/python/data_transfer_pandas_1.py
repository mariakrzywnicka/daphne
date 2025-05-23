#!/usr/bin/python

# Copyright 2021 The DAPHNE Consortium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Data transfer from pandas to DAPHNE and back, via files.
# pd.DataFrame

import pandas as pd
from daphne.context.daphne_context import DaphneContext

df = pd.DataFrame({"abc": [1, 2, 3], "def": [-1.1, -2.2, -3.3], "ghi": ["red", "green", "blue"]})

dctx = DaphneContext()

dctx.from_pandas(df, shared_memory=False).print().compute(type="files")