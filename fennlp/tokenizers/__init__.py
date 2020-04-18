#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""

import os
import importlib
# automatically import any Python files in the tokenizers/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        dataset_name = file[: file.find(".py")]
        module = importlib.import_module("fennlp.tokenizers." + dataset_name)