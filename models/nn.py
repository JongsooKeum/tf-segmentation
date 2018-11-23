import time
from abc import abstractmethod, ABCMeta
import tensorflow as tf
import numpy as np
from models.layers import conv_layer, boundary_refine_module, global_conv_module, up_scale, feature_pyramid_attention, global_attention_upsample
from tensorflow.contrib.slim.nets import resnet_v2 as resnet_v2
import os

slim = tf.contrib.slim


class SegNet(metaclass=ABCMeta):
    """Base class for Convolutional Neural Networks for segmentation."""

    def __init__(self, input_shape, num_classes, **kwargs):
        """
        model initializer
        :param input_shape: tuple, shape (H, W, C)
        :param num_classes: int, total number of classes
        """
        if input_shape is None:
            input_shape = [None, None, 3]
        self.X = tf.placeholder(tf.float32, [None] + input_shape)
        self.y = tf.placeholder(
            tf.int32, [None] + input_shape[:2] + [num_classes])
        self.is_train = tf.placeholder(tf.bool)
        self.num_classes = num_classes
        self.d = self._build_model(**kwargs)
        self.pred = self.d['pred']
        self.logits = self.d['logits']
        self.loss = self._build_loss(**kwargs)

    @abstractmethod
    def _build_model(self, **kwargs):
        """
        Build model.
        This should be implemented.
        """
        pass

    @abstractmethod
    def _build_loss(self, **kwargs):
        """
        build loss function for the model training.
        This should be implemented.
        """
        pass

    def predict(self, sess, dataset, verbose=False, **kwargs):
        """
        Make predictions for the given dataset.
        :param ses: tf.Session.
        :param dataset: DataSet.
        :param verbose: bool, whether to print details during prediction.
        :param kwargs: dict, extra arguments for prediction.
                -batch_size: int, batch size for each iteraction.
        :return _y_pred: np.ndarray, shape: shape of self.pred
        """

        batch_size = kwargs.pop('batch_size', 64)

        num_classes = self.num_classes
        pred_size = dataset.num_examples
        num_steps = pred_size // batch_size

        if verbose:
            print('Running prediction loop...')

        # Start prediction loop
        _y_pred = []
        start_time = time.time()
        for i in range(num_steps + 1):
            if i == num_steps:
                _batch_size = pred_size - num_steps * batch_size
            else:
                _batch_size = batch_size
            X, _ = dataset.next_batch(
                _batch_size, shuffle=False)
            # Compute predictions
            # (N, H, W, num_classes)
            y_pred = sess.run(self.pred, feed_dict={
                              self.X: X, self.is_train: False})
            _y_pred.append(y_pred)

        if verbose:
            print('Total prediction time(sec): {}'.format(
                time.time() - start_time))

        _y_pred = np.concatenate(_y_pred, axis=0)

        return _y_pred


class GCN(SegNet):
    """
    GCN class
    see: Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network
    https://arxiv.org/abs/1703.02719
    """
    def _build_model(self, **kwargs):
        d = dict()
        num_classes = self.num_classes
        pretrain = kwargs.pop('pretrain', True)
        frontend = kwargs.pop('frontend', 'resnet_v2_50')

        if pretrain:
            frontend_dir = os.path.join(
                'pretrained_models', '{}.ckpt'.format(frontend))
            with slim.arg_scope(resnet_v2.resnet_arg_scope()):
                logits, end_points = resnet_v2.resnet_v2_50(
                    self.X, is_training=self.is_train)
                d['init_fn'] = slim.assign_from_checkpoint_fn(model_path=frontend_dir,
                                                              var_list=slim.get_model_variables(frontend))
                resnet_dict = [
                '/block1/unit_2/bottleneck_v2',  # conv1
                '/block2/unit_3/bottleneck_v2',  # conv2
                '/block3/unit_5/bottleneck_v2',  # conv3
                '/block4/unit_3/bottleneck_v2'   # conv4
                ]
                convs = [end_points[frontend + x] for x in resnet_dict]
        else:
            # TODO build convNet
            raise NotImplementedError("Build own convNet!")
        if self.X.shape[1].value is None:
            # input size should be bigger than (512, 512)
            g_kernel_size = (15, 15)
        else:
            g_kernel_size = (self.X.shape[1].value//32-1, self.X.shape[2].value//32-1)

        with tf.variable_scope('layer5'):
            d['gcm1'] = global_conv_module(convs[3], num_classes, g_kernel_size)
            d['brm1_1'] = boundary_refine_module(d['gcm1'], num_classes)
            d['up16'] = up_scale(d['brm1_1'], 2)

        with tf.variable_scope('layer4'):
            d['gcm2'] = global_conv_module(convs[2], num_classes, g_kernel_size)
            d['brm2_1'] = boundary_refine_module(d['gcm2'], num_classes)
            d['sum16'] = d['up16'] + d['brm2_1']
            d['brm2_2'] = boundary_refine_module(d['sum16'], num_classes)
            d['up8'] = up_scale(d['brm2_2'], 2)

        with tf.variable_scope('layer3'):
            d['gcm3'] = global_conv_module(convs[1], num_classes, g_kernel_size)
            d['brm3_1'] = boundary_refine_module(d['gcm3'], num_classes)
            d['sum8'] = d['up8'] + d['brm3_1']
            d['brm3_2'] = boundary_refine_module(d['sum8'], num_classes)
            d['up4'] = up_scale(d['brm3_2'], 2)

        with tf.variable_scope('layer2'):
            d['gcm4'] = global_conv_module(convs[0], num_classes, g_kernel_size)
            d['brm4_1'] = boundary_refine_module(d['gcm4'], num_classes)
            d['sum4'] = d['up4'] + d['brm4_1']
            d['brm4_2'] = boundary_refine_module(d['sum4'], num_classes)
            d['up2'] = up_scale(d['brm4_2'], 2)

        with tf.variable_scope('layer1'):
            d['brm4_3'] = boundary_refine_module(d['up2'], num_classes)
            d['up1'] = up_scale(d['brm4_3'], 2)
            d['brm4_4'] = boundary_refine_module(d['up1'], num_classes)

        with tf.variable_scope('output_layer'):
            d['logits'] = conv_layer(d['brm4_4'], num_classes, (1, 1), (1, 1))
            d['pred'] = tf.nn.softmax(d['logits'], axis=-1)

        return d

    def _build_loss(self, **kwargs):
        """
        Build loss function for the model training.
        :return tf.Tensor.
        """
        # Calculate pixel-wise cross_entropy loss and Ignore Unknown region
        real = tf.cast(tf.greater_equal(self.y[...,0], 0), dtype=tf.float32)
        softmax_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=self.y, logits=self.logits, dim=-1)
        loss = tf.multiply(softmax_loss, real)
        return tf.reduce_mean(loss)