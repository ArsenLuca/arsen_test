# -*- coding: utf-8 -*-
"""
    get drn symbols
"""

import mxnet as mx
from arsen_toolbox.mx_tools.symbols.dilated_resnet import get_symbol

if __name__ == "__main__":
    drn = get_symbol(1000, 18, "3, 224, 224")
    input_shapes = {"data": (1, 3, 224, 224)}
    #internals = drn.get_internals()
    #arg_shapes, out_shapes, aux_shapes = interals.infer_shape(**input_shapes)
    print drn.infer_shape(**input_shapes)
    #print dict(zip(internals.list_outputs(), out_shapes))
    #print aux_shapes
