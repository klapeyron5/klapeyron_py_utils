from klapeyron_py_utils.tensorflow.imports import TfSetup
tf = TfSetup.import_tensorflow(3)
from klapeyron_py_utils.models.blocks.resnet import Layer_conv_bn_relu
from klapeyron_py_utils.models.blocks.resnet import SEResidual_block_compact_v3, Stack_of_blocks, FC_logits
from klapeyron_py_utils.models.configs.model_train_config import Model_Train_Config
from klapeyron_py_utils.models.configs.model_config import Model_Config


class SEResNet34_v2(tf.Module):

    OUTPUT_CHANNELS = 2
    bs = (None,)

    l2_loss = tf.nn.l2_loss(0.0)

    NAME = 'SEResNet34_v2'  # TODO

    def __init__(self, model_config: Model_Config, train_config: Model_Train_Config):

        self.input_shape = model_config.input_shape

        self.conv0 = Layer_conv_bn_relu((7, 7, self.input_shape[-1], 64), 2)

        Ns = [3, 4, 6, 3]
        blocks_output_depths = [64, 128, 256, 512]
        se_reduction = 16
        block = SEResidual_block_compact_v3
        self.stack_of_stacks_of_blocks = []

        input_input_depth = self.conv0.OUTPUT_CHANNELS
        for i, N, output_depth in zip(range(len(Ns)), Ns, blocks_output_depths):
            stack_of_blocks = Stack_of_blocks(N=N, block=block, input_input_depth=input_input_depth, block_output_depth=output_depth, reduction=se_reduction)
            self.stack_of_stacks_of_blocks.append(stack_of_blocks)
            input_input_depth = stack_of_blocks.OUTPUT_CHANNELS

        self.fc_logits = FC_logits(output_depth, self.OUTPUT_CHANNELS)

        self.dropout_drop_prob = train_config.dropout_drop_prob

        self.__call__ = tf.function(
            self.__call__,
            input_signature=[
                tf.TensorSpec(self.bs + self.input_shape, tf.float32),
            ])
        self.get_logits = tf.function(
            self.get_logits,
            input_signature=[
                tf.TensorSpec(self.bs + self.input_shape, tf.float32), tf.TensorSpec([], tf.bool),
            ])
        self.reg_loss = tf.function(
            self.reg_loss,
            input_signature=[])

    @tf.function
    def __call__(self, x):
        out = self.get_logits(x, False)
        out = tf.nn.softmax(out)
        return out

    @tf.function
    def get_logits(self, x, training):
        conv0 = self.conv0(x, training=training)
        pool0 = tf.nn.max_pool(conv0, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        for block in self.stack_of_stacks_of_blocks:
            pool0 = block(pool0, training)

        pool5 = tf.reduce_mean(pool0, axis=[1, 2])
        if training:
            pool5 = tf.nn.dropout(pool5, rate=self.dropout_drop_prob)
        fc_output = self.fc_logits(pool5, self.OUTPUT_CHANNELS)

        return fc_output

    @tf.function
    def reg_loss(self):
        l2_loss = tf.nn.l2_loss(0.0)

        l2_loss += self.conv0.get_reg_loss_l2()

        for block in self.stack_of_stacks_of_blocks:
            l2_loss += block.get_reg_loss_l2()

        l2_loss += self.fc_logits.get_reg_loss_l2()
        return l2_loss
