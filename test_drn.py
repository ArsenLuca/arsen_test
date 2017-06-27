# -*- coding: utf-8 -*-
"""
    get drn symbols
"""

import mxnet as mx
from arsen_toolbox.mx_tools.symbols.dilated_resnet import get_symbol

if __name__ == "__main__":
    drn = get_symbol(1000, 18, "3, 224, 224")
    mx.viz.plot_network(drn).view()
