"""
masked.py
========
Contain MaskedConv2D layer and MaskedAvgPool2D pooling class with functional interfaces.
"""

import tensorflow as tf
import logging


# Should run on gpus. However I need to verify that.
class MaskedConv2D(tf.layers.Layer):
    """ 2D masked convolution layer (e.g. spatial convolution over images).
        
        This work/code is inspired by
        1) https://github.com/ehoogeboom/hexaconv
        2) https://gist.github.com/abhaikollara/430c0491c851cf0b05a852f1faa805d7
        
        This layer creates a masked convolution kernel that is convolved
        (actually cross-correlated) with the layer input to produce a tensor of
        outputs. If `use_bias` is True (and a `bias_initializer` is provided),
        a bias vector is created and added to the outputs. Finally, if
        `activation` is not `None`, it is applied to the outputs as well.
        
        Arguments (taken from tensorflow):
        filters: Integer, the dimensionality of the output space (i.e. the number
            of filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            height and width of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the height and width.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `"valid"` or `"same"` (case-insensitive).
        data_format: A string, one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first` corresponds to
            inputs with shape `(batch, channels, height, width)`.
        dilation_rate: An integer or tuple/list of 2 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        activation: Activation function. Set it to None to maintain a
            linear activation.
        use_bias: Boolean, whether the layer uses a bias.
        kernel_initializer: An initializer for the convolution kernel.
        bias_initializer: An initializer for the bias vector. If None, the default
            initializer will be used.
        kernel_regularizer: Optional regularizer for the convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        kernel_constraint: Optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The function
            must take as input the unprojected variable and must return the
            projected variable (which must have the same shape). Constraints are
            not safe to use when doing asynchronous distributed training.
        bias_constraint: Optional projection function to be applied to the
            bias after being updated by an `Optimizer`.
        trainable: Boolean, if `True` also add variables to the graph collection
            `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
        name: A string, the name of the layer.
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        
        super(MaskedConv2D, self).__init__(trainable=True,name=name,activity_regularizer=activity_regularizer,**kwargs)
        self.logger = logging.getLogger(__name__ + '.MaskedConv2D')
        
        self.filters = filters
        self.kernel_size = kernel_size
        # There is a discrepancy in the arguments for 'padding' and 'data_format' between
        # the layer classes of tensorflow and the 'conv2d' and 'bias_add' function
        # in tensorflow.nn. A translation is introduced so that the MaskedConv2D class
        # is consistent with the convention of the layer classes of tensorflow.
        if padding == 'valid':
            self.padding = 'VALID'
        elif padding == 'same':
            self.padding = 'SAME'
        else:
            raise ValueError('padding must be valid or same.')
        if data_format == 'channels_last':
            self.data_format = 'NHWC'
            self.strides = (1,) + strides + (1,)
        elif data_format == 'channels_first':
            self.data_format = 'NCHW'
            self.strides = (1, 1) + strides
        else:
            raise ValueError('data_formate must be channels_last or channels_first.')
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

    def build(self, input_shape):
        
        input_dim = tf.TensorShape(input_shape)
        if self.data_format == 'NHWC':
            in_channels = input_dim[3]
        elif self.data_format == 'NCHW':
            in_channels = input_dim[1]
        out_channels = self.filters
        # For now the mask are hardcoded (to run on gpu)! Later with build_kernel() from utils.py!
        if self.kernel_size == 3:
            self.mask = tf.Variable([[1.,1.,0.],[1.,1.,1.],[0.,1.,1.]], tf.float32)
        elif self.kernel_size == 5:
            self.mask = tf.Variable([[1.,1.,1.,0.,0.],[1.,1.,1.,1.,0.],[1.,1.,1.,1.,1.],[0.,1.,1.,1.,1.],[0.,0.,1.,1.,1.]], tf.float32)
        else:
            raise ValueError('kernel_size must be 3 or 5.')
                                                            
        kernel_shape = (in_channels, out_channels, self.kernel_size, self.kernel_size)
        self.kernel = self.add_variable(name='kernel',
                                        shape=kernel_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True,
                                        dtype=self.dtype)
        self.kernel = self.kernel * self.mask
        self.kernel = tf.transpose(self.kernel, perm=[2, 3, 0, 1])
                                                                                                
        if self.use_bias:
            self.bias = self.add_variable(name='bias',
                                          shape=(out_channels,),
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint,
                                          trainable=True,
                                          dtype=self.dtype)
        else:
            self.bias = None

    def call(self, inputs):
    
        outputs = tf.nn.conv2d(input=inputs,
                               filter=self.kernel,
                               strides=self.strides,
                               padding=self.padding,
                               data_format=self.data_format,
                               name=self.name)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias,data_format=self.data_format)
        if self.activation is not None:
                return self.activation(outputs)
        return outputs

def masked_conv2d(inputs,
                  filters,
                  kernel_size,
                  strides=(1, 1),
                  padding='valid',
                  data_format='channels_last',
                  dilation_rate=(1, 1),
                  activation=None,
                  use_bias=True,
                  kernel_initializer=None,
                  bias_initializer=tf.zeros_initializer(),
                  kernel_regularizer=None,
                  bias_regularizer=None,
                  activity_regularizer=None,
                  kernel_constraint=None,
                  bias_constraint=None,
                  trainable=True,
                  name=None,
                  reuse=None):
    
    """ Functional interface for 2D masked convolution layer (e.g. temporal convolution).
        
        Same arguments as in the MaskedConv2D class.
    """
    layer = MaskedConv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilation_rate=dilation_rate,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                trainable=trainable,
                name=name,
                _reuse=reuse,
                _scope=name)
                         
    return layer.apply(inputs)


# Should run on gpus. However I need to verify that.
class MaskedAvgPool2D(tf.layers.Layer):
    """ Masked average pooling layer for 2D inputs (e.g. images).
        
        This work/code is inspired by
        1) https://github.com/ehoogeboom/hexaconv
        2) https://gist.github.com/abhaikollara/430c0491c851cf0b05a852f1faa805d7
        
        Arguments (taken from tensorflow):
        pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width)
            specifying the size of the pooling window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the pooling operation.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        padding: A string. The padding method, either 'valid' or 'same'.
            Case-insensitive.
        data_format: A string. The ordering of the dimensions in the inputs.
            `channels_last` (default) and `channels_first` are supported.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first` corresponds to
            inputs with shape `(batch, channels, height, width)`.
        name: A string, the name of the layer.
    """
    def __init__(self,
                 pool_size=(3,3),
                 strides=(3,3),
                 padding='valid',
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        
        super(MaskedAvgPool2D, self).__init__()
        self.logger = logging.getLogger(__name__ + '.MaskedAvgPool2D')
        
        if type(pool_size) is tuple:
            if pool_size[0] != pool_size[1]:
                raise ValueError('pool_size tuple must be quadratic.')
            self.pool_size = pool_size[0]
        self.pool_size = pool_size
        # There is a discrepancy in the arguments for 'padding' and 'data_format' between
        # the layer classes of tensorflow and the 'conv2d' and 'bias_add' function
        # in tensorflow.nn. A translation is introduced so that the MaskedConv2D class
        # is consistent with the convention of the layer classes of tensorflow.
        if padding == 'valid':
            self.padding = 'VALID'
        elif padding == 'same':
            self.padding = 'SAME'
        else:
            raise ValueError('padding must be valid or same.')
        if data_format == 'channels_last':
            self.data_format = 'NHWC'
            self.strides = (1,) + strides + (1,)
        elif data_format == 'channels_first':
            self.data_format = 'NCHW'
            self.strides = (1, 1) + strides
        else:
            raise ValueError('data_formate must be channels_last or channels_first.')

    def build(self, input_shape):
    
        input_dim = tf.TensorShape(input_shape)
        if self.data_format == 'NHWC':
            in_channels = input_dim[3]
        elif self.data_format == 'NCHW':
            in_channels = input_dim[1]
        out_channels = in_channels
        # For now the mask are hardcoded (to run on gpu)! Later with build_kernel() from utils.py!
        if self.pool_size == 3:
            self.mask = tf.Variable([[1./7.,1./7.,0.],[1./7.,1./7.,1./7.],[0.,1./7.,1./7.]], tf.float32)
        elif self.pool_size == 5:
            self.mask = tf.Variable([[1./19.,1./19.,1./19.,0.,0.],[1./19.,1./19.,1./19.,1./19.,0.],[1./19.,1./19.,1./19.,1./19.,1./19.],[0.,1./19.,1./19.,1./19.,1./19.],[0.,0.,1./19.,1./19.,1./19.]], tf.float32)
        else:
            raise ValueError('pool_size must be (3,3) or (5,5).')

        kernel_shape = (in_channels, out_channels, self.pool_size, self.pool_size)
        self.kernel = self.add_variable(name='avgpool',
                                        shape=kernel_shape,
                                        initializer=tf.ones_initializer(),
                                        trainable=False,
                                        dtype=self.dtype)
        self.kernel = self.kernel * self.mask
        self.kernel = tf.transpose(self.kernel, perm=[2, 3, 0, 1])

    def call(self, inputs):
        
        outputs = tf.nn.conv2d(input=inputs,
                               filter=self.kernel,
                               strides=self.strides,
                               padding=self.padding,
                               data_format=self.data_format,
                               name=self.name)
        return outputs

def masked_avgpool2d(inputs,
                     pool_size=(3,3),
                     strides=(3,3),
                     padding='valid',
                     data_format='channels_last',
                     name=None):
    """ Functional interface for 2D masked average pooling layer.
        
        Same arguments as in the MaskedAvgPool2D class.
    """
    avgpool = MaskedAvgPool2D(
                    pool_size=pool_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    name=name)
                              
    return avgpool.apply(inputs)
