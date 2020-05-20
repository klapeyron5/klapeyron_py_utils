from klapeyron_py_utils.tensorflow.imports import import_tensorflow
tf = import_tensorflow(3)


def get_out_ch(sh):
    if len(sh) == 1:
        out_ch = sh
    else:
        out_ch = sh[-1]
    return out_ch


def init_normal(sh, name=None):
    initial = tf.random.truncated_normal(sh, stddev=0.05, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def init_zeros(sh, name=None, trainable=True):
    initial = tf.zeros(sh, dtype=tf.float32)
    return tf.Variable(initial, name=name, trainable=trainable)


def init_ones(sh, name=None, trainable=True):
    initial = tf.ones(sh, dtype=tf.float32)
    return tf.Variable(initial, name=name, trainable=trainable)


class BN_v0(tf.Module):
    def __init__(self, in_ch: int):
        super(BN_v0, self).__init__()
        self.epsilon = 1e-5
        self.decay = 0.99
        self.beta = init_zeros([in_ch], name='beta')
        self.gamma = init_ones([in_ch], name='gamma')
        self.pop_mean = init_zeros([in_ch], trainable=False, name='pop_mean')
        self.pop_var = init_ones([in_ch], trainable=False, name='pop_var')

    def __call__(self, batch, training):
        if training:
            batch_mean, batch_var = tf.nn.moments(batch, axes=[0, 1, 2])
            pop_mean_upd = self.pop_mean * self.decay + batch_mean * (1 - self.decay)
            self.pop_mean.assign(pop_mean_upd)
            pop_var_upd = self.pop_var * self.decay + batch_var * (1 - self.decay)
            self.pop_var.assign(pop_var_upd)
            bn = tf.nn.batch_normalization(batch, mean=batch_mean, variance=batch_var,
                                           offset=self.beta, scale=self.gamma, variance_epsilon=self.epsilon)
        else:
            bn = tf.nn.batch_normalization(batch, mean=self.pop_mean, variance=self.pop_var,
                                           offset=self.beta, scale=self.gamma, variance_epsilon=self.epsilon)
        return bn


class Layer_conv_bn_relu(tf.Module):
    def __init__(self, conv_kernel_sh, conv_stride, conv_padding='SAME'):
        """
        :param conv_kernel_sh: HWIO
        """
        super(Layer_conv_bn_relu, self).__init__()
        out_ch = conv_kernel_sh[-1]

        self.conv_kernel = init_normal(conv_kernel_sh)
        self.conv_strides = [1, conv_stride, conv_stride, 1]
        self.conv_padding = conv_padding

        self.bn = BN_v0(out_ch)

        self.OUTPUT_CHANNELS = self.conv_kernel.shape[-1]

    @tf.function(input_signature=[
        tf.TensorSpec([None, None, None, None], tf.float32),
        tf.TensorSpec((), tf.bool),
    ])
    def __call__(self, x, training):
        conv = tf.nn.conv2d(x, filters=self.conv_kernel, strides=self.conv_strides, padding=self.conv_padding)
        conv_bn = self.bn(conv, training=training)
        conv_bn_relu = tf.nn.relu(conv_bn)
        return conv_bn_relu

    def get_reg_loss_l2(self):
        l2_loss = tf.nn.l2_loss(self.conv_kernel)
        return l2_loss


class Layer_bn_relu_conv(tf.Module):
    def __init__(self, conv_kernel_sh, conv_stride, conv_padding='SAME'):
        super(Layer_bn_relu_conv, self).__init__()
        in_ch = conv_kernel_sh[-2]

        self.bn = BN_v0(in_ch)

        self.conv_kernel = init_normal(conv_kernel_sh)
        self.conv_strides = [1, conv_stride, conv_stride, 1]
        self.conv_padding = conv_padding

        self.OUTPUT_CHANNELS = self.conv_kernel.shape[-1]

    def __call__(self, x, training):
        bn = self.bn(x, training=training)
        bn_relu = tf.nn.relu(bn)
        bn_relu_conv = tf.nn.conv2d(bn_relu, filters=self.conv_kernel, strides=self.conv_strides, padding=self.conv_padding)
        return bn_relu_conv

    def get_reg_loss_l2(self):
        l2_loss = tf.nn.l2_loss(self.conv_kernel)
        return l2_loss


class Layer_conv_bn(tf.Module):
    def __init__(self, conv_kernel_sh, conv_stride, conv_padding='SAME'):
        super(Layer_conv_bn, self).__init__()
        out_ch = conv_kernel_sh[-1]

        self.conv_kernel = init_normal(conv_kernel_sh)
        self.conv_strides = [1, conv_stride, conv_stride, 1]
        self.conv_padding = conv_padding

        self.bn = BN_v0(out_ch)

        self.OUTPUT_CHANNELS = self.conv_kernel.shape[-1]

    def __call__(self, x, training):
        conv = tf.nn.conv2d(x, filters=self.conv_kernel, strides=self.conv_strides, padding=self.conv_padding)
        conv_bn = self.bn(conv, training=training)
        return conv_bn

    def get_reg_loss_l2(self):
        l2_loss = tf.nn.l2_loss(self.conv_kernel)
        return l2_loss


class Layer_conv(tf.Module):
    def __init__(self, conv_kernel_sh, conv_stride, conv_padding='SAME'):
        super(Layer_conv, self).__init__()

        self.conv_kernel = init_normal(conv_kernel_sh)
        self.conv_strides = [1, conv_stride, conv_stride, 1]
        self.conv_padding = conv_padding

        self.OUTPUT_CHANNELS = self.conv_kernel.shape[-1]

    def __call__(self, x, training):
        conv = tf.nn.conv2d(x, filters=self.conv_kernel, strides=self.conv_strides, padding=self.conv_padding)
        return conv

    def get_reg_loss_l2(self):
        l2_loss = tf.nn.l2_loss(self.conv_kernel)
        return l2_loss


class SE(tf.Module):
    def __init__(self, input_depth: int, reduction=16, ):
        super(SE, self).__init__()

        inner_depth = input_depth // reduction
        self.squeeze = Layer_conv([1, 1, input_depth, inner_depth], 1)
        self.excitation = Layer_conv([1, 1, inner_depth, input_depth], 1)

    def __call__(self, x, training):
        squeeze = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        squeeze = self.squeeze(squeeze, training)
        squeeze = tf.nn.relu(squeeze)
        excitation = self.excitation(squeeze, training)
        excitation = tf.nn.sigmoid(excitation)
        return tf.multiply(x, excitation)


class Residual_block_compact_v3(tf.Module):
    """
    Figure 4, variant e: https://arxiv.org/pdf/1603.05027.pdf

    bn_relu_conv_layer
    bn_relu_conv_layer

    shortcat projection:
    conv_bn_layer

    add
    """
    def __init__(self, output_depth: int, input_depth: int = None, input_tensor: tf.Tensor = None):
        if input_tensor is None:
            assert input_depth is not None
        else:
            input_depth = input_tensor.shape[-1]
        if input_depth == output_depth:
            self.bottleneck = False
            self.projection = False
        elif input_depth * 2 == output_depth:
            self.bottleneck = True
            self.projection = True
        else:
            raise Exception('Wrong block depths for original ResNet18/34. '
                            'Your input_depth is ' + str(input_depth) + ' and output_depth is ' +
                            str(output_depth) + '.')

        block_stride = 1
        self.conv0 = Layer_bn_relu_conv([3, 3, input_depth, output_depth], block_stride)
        self.conv1 = Layer_bn_relu_conv([3, 3, output_depth, output_depth], block_stride)

        # bottleneck by 3x3 conv with stride 2
        if self.bottleneck:
            self.bottleneck_func = self.bottleneck_max_pool
        else:
            self.bottleneck_func = self.bottleneck_identity

        # Option B: identity when dimensions are equal and projection when are not.
        # Dimension are not only equal when it's bottleneck block,
        # so projections with stride 2 in that case
        if self.projection:
            self.init_projection_1x1conv(input_depth, output_depth, block_stride)
            self.projection_func = self.projection_1x1conv
        else:
            self.projection_func = self.projection_identity

        self.OUTPUT_CHANNELS = output_depth

    @tf.function(input_signature=[
        tf.TensorSpec([None, None, None, None], tf.float32),
        tf.TensorSpec((), tf.bool),
    ])
    def __call__(self, x, training):
        bottlenecked_x = self.bottleneck_func(x, training)

        conv0 = self.conv0(bottlenecked_x, training)
        conv1 = self.conv1(conv0, training)

        projection = self.projection_func(bottlenecked_x, training)
        res = tf.add(conv1, projection)

        return res

    def bottleneck_max_pool(self, x, training=False):
        out = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        return out

    def bottleneck_identity(self, x, training=False):
        return x

    def projection_1x1conv(self, x, training):
        out = self.projection_1x1conv(x, training)
        return out

    def init_projection_1x1conv(self, input_depth, output_depth, stride):
            self.projection_1x1conv = Layer_conv_bn([1, 1, input_depth, output_depth], stride)

    def projection_identity(self, x, training=False):
        return x

    def get_reg_loss_l2(self):
        l2_loss = tf.nn.l2_loss(0.0)
        l2_loss += tf.nn.l2_loss(self.conv0.get_reg_loss_l2())
        l2_loss += tf.nn.l2_loss(self.conv0.get_reg_loss_l2())
        if self.projection:
            l2_loss += tf.nn.l2_loss(self.projection_1x1conv.get_reg_loss_l2())
        return l2_loss


class SEResidual_block_compact_v3(tf.Module):
    """
    SE-PRE
    """
    def __init__(self, output_depth: int, input_depth: int = None, input_tensor: tf.Tensor = None, reduction: int = 16):
        if input_tensor is None:
            assert input_depth is not None
        else:
            input_depth = input_tensor.shape[-1]
        if input_depth == output_depth:
            self.bottleneck = False
            self.projection = False
        elif input_depth * 2 == output_depth:
            self.bottleneck = True
            self.projection = True
        else:
            raise Exception('Wrong block depths for original ResNet18/34. '
                            'Your input_depth is ' + str(input_depth) + ' and output_depth is ' +
                            str(output_depth) + '.')

        self.se = SE(input_depth, reduction)
        block_stride = 1
        self.conv0 = Layer_bn_relu_conv([3, 3, input_depth, output_depth], block_stride)
        self.conv1 = Layer_bn_relu_conv([3, 3, output_depth, output_depth], block_stride)

        # bottleneck by 3x3 conv with stride 2
        if self.bottleneck:
            self.bottleneck_func = self.bottleneck_max_pool
        else:
            self.bottleneck_func = self.bottleneck_identity

        # Option B: identity when dimensions are equal and projection when are not.
        # Dimension are not only equal when it's bottleneck block,
        # so projections with stride 2 in that case
        if self.projection:
            self.init_projection_1x1conv(input_depth, output_depth, block_stride)
            self.projection_func = self.projection_1x1conv
        else:
            self.projection_func = self.projection_identity

        self.OUTPUT_CHANNELS = output_depth

    @tf.function(input_signature=[
        tf.TensorSpec([None, None, None, None], tf.float32),
        tf.TensorSpec((), tf.bool),
    ])
    def __call__(self, x, training):
        bottlenecked_x = self.bottleneck_func(x, training)

        se = self.se(bottlenecked_x, training)

        conv0 = self.conv0(se, training)
        conv1 = self.conv1(conv0, training)

        projection = self.projection_func(bottlenecked_x, training)
        res = tf.add(conv1, projection)

        return res

    def bottleneck_max_pool(self, x, training=False):
        out = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        return out

    def bottleneck_identity(self, x, training=False):
        return x

    def projection_1x1conv(self, x, training):
        out = self.projection_1x1conv(x, training)
        return out

    def init_projection_1x1conv(self, input_depth, output_depth, stride):
            self.projection_1x1conv = Layer_conv_bn([1, 1, input_depth, output_depth], stride)

    def projection_identity(self, x, training=False):
        return x

    def get_reg_loss_l2(self):
        l2_loss = tf.nn.l2_loss(0.0)
        l2_loss += tf.nn.l2_loss(self.conv0.get_reg_loss_l2())
        l2_loss += tf.nn.l2_loss(self.conv0.get_reg_loss_l2())
        if self.projection:
            l2_loss += tf.nn.l2_loss(self.projection_1x1conv.get_reg_loss_l2())
        return l2_loss


class Stack_of_blocks(tf.Module):
    def __init__(self, N, block, input_input_depth, block_output_depth, **block_kwargs):
        assert issubclass(block, tf.Module)
        self.blocks = []
        block_input_depth = input_input_depth
        for i in range(N):
            b = block(output_depth=block_output_depth, input_depth=block_input_depth, **block_kwargs)
            self.blocks.append(b)
            block_input_depth = b.OUTPUT_CHANNELS

        self.OUTPUT_CHANNELS = self.blocks[-1].OUTPUT_CHANNELS

    def __call__(self, x, training):
        for block in self.blocks:
            x = block(x, training)
        return x

    def get_reg_loss_l2(self):
        l2_loss = tf.nn.l2_loss(0.0)
        for block in self.blocks:
            l2_loss += block.get_reg_loss_l2()
        return l2_loss


class FC_logits(tf.Module):
    def __init__(self, input_depth, output_depth):
        sh = (input_depth, output_depth)
        self.w = init_normal(sh)
        self.b = init_zeros([output_depth])

    def __call__(self, x, training=True):
        fc = tf.matmul(x, self.w) + self.b
        return fc

    def get_reg_loss_l2(self):
        l2_loss = tf.nn.l2_loss(self.w)
        return l2_loss


def ut_0():
    r = Residual_block_compact_v3(1, 1)
    import numpy as np
    x = np.array([
        [
            [[1], [2], [3]],
            [[4], [5], [6]],
            [[1], [0], [0]]
        ]
    ]).astype(np.float16)
    a = r(x, training=False).numpy()
    pass
