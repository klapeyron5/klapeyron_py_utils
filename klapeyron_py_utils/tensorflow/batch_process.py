import numpy as np
from klapeyron_py_utils.tensorflow.imports import import_tensorflow
tf = import_tensorflow()


def batch_process(fn):
    specs = list(fn.function_spec.flat_input_signature)
    # assert all(isinstance(x, TensorSpec) for x in specs)
    @tf.function(input_signature=specs+[tf.TensorSpec([], tf.uint16)])
    def wrapper(data, bs):
        bs = tf.cast(bs, tf.int32)
        len = tf.shape(fn(data))[0]
        i = 0
        sind = i * bs
        find = (i + 1) * bs
        if find < len:
            outs = fn(data)[sind:find]
            i += 1
            sind = i * bs
            find = (i + 1) * bs
            while find < len:
                batch_out = fn(data)[sind:find]
                outs = tf.concat([outs,batch_out], 0)
                i += 1
                sind = i * bs
                find = (i + 1) * bs
            outs = tf.concat([outs,fn(data)[sind:]], 0)
        else:
            outs = fn(data)
        return outs
    return wrapper


def unittest():
    @batch_process
    @tf.function(input_signature=[tf.TensorSpec([None],tf.int32)])
    def func(x):
        return x + 1

    inp = np.array([98,1,3,5,8])
    o = func(inp, 2).numpy()
    ans = inp+1
    assert np.array_equal(o,ans)
