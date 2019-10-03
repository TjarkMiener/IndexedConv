"""
indexed.py
========
Contain IndexedConv2D layer and IndexedAvgPool2D/IndexedMaxPool2D pooling class with functional interfaces.
"""

import tensorflow as tf
import logging
import indexedconv.utils as utils


class IndexedMaxPool2D(tf.layers.Layer):
    """ Indexed max pooling layer for 2D inputs (e.g. images).
        
        This work/code is inspired by
        1) https://github.com/ehoogeboom/hexaconv
        2) https://gist.github.com/abhaikollara/430c0491c851cf0b05a852f1faa805d7
        
        Arguments (taken from tensorflow):
        indices: Matrix of indices.
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
                 indices,
                 pool_size=(3,3),
                 strides=(3,3),
                 padding='valid',
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        
        super(IndexedMaxPool2D, self).__init__()
        self.logger = logging.getLogger(__name__ + '.IndexedMaxPool2D')
        
        # Check utils function 'pool_index_matrix'
        indices_neighbours = utils.neighbours_extraction(indices,kernel_type='Hex')
        self.indices, self.mask = utils.prepare_mask(indices_neighbours)
        self.indices_ = tf.convert_to_tensor(self.indices,name='ind')
        self.mask_ = tf.convert_to_tensor(self.mask,name='mask')
        
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
            raise ValueError('data_format must be channels_last or channels_first.')

    def build(self, input_shape)
        #TODO

    def call(self, inputs):
        #TODO
        outputs=1
        return outputs

def indexed_maxpool2d(inputs,
                      indices,
                      pool_size=(3,3),
                      strides=(3,3),
                      padding='valid',
                      data_format='channels_last',
                      name=None):
    """ Functional interface for 2D indexed max pooling layer.
        
        Same arguments as in the IndexedMaxPool2D class.
    """
    maxpool = IndexedMaxPool2D(
                    indices=indices,
                    pool_size=pool_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    name=name)
                              
    return maxpool.apply(inputs)


class IndexedAvgPool2D(tf.layers.Layer):
    """ Indexed average pooling layer for 2D inputs (e.g. images).
        
        This work/code is inspired by
        1) https://github.com/ehoogeboom/hexaconv
        2) https://gist.github.com/abhaikollara/430c0491c851cf0b05a852f1faa805d7
        
        Arguments (taken from tensorflow):
        indices: Matrix of indices.
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
                 indices,
                 pool_size=(3,3),
                 strides=(3,3),
                 padding='valid',
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        
        super(IndexedAvgPool2D, self).__init__()
        self.logger = logging.getLogger(__name__ + '.IndexedAvgPool2D')
        
        # Check utils function 'pool_index_matrix'
        indices_neighbours = utils.neighbours_extraction(indices,kernel_type='Hex')
        self.indices, self.mask = utils.prepare_mask(indices_neighbours)
        self.indices_ = tf.convert_to_tensor(self.indices,name='ind')
        self.mask_ = tf.convert_to_tensor(self.mask,name='mask')
        
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
            raise ValueError('data_format must be channels_last or channels_first.')

    def build(self, input_shape)
        #TODO

    def call(self, inputs):
        #TODO
        outputs=1
        return outputs

def indexed_avgpool2d(inputs,
                      indices,
                      pool_size=(3,3),
                      strides=(3,3),
                      padding='valid',
                      data_format='channels_last',
                      name=None):
    """ Functional interface for 2D indexed average pooling layer.
        
        Same arguments as in the IndexedAvgPool2D class.
    """
    avgpool = IndexedAvgPool2D(
                    indices=indices,
                    pool_size=pool_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    name=name)
                              
    return avgpool.apply(inputs)


class IndexedConv2D(tf.layers.Layer):
    """ 2D indexed convolution layer (e.g. spatial convolution over images).
        
        This work/code is inspired by
        1) https://github.com/ehoogeboom/hexaconv
        2) https://gist.github.com/abhaikollara/430c0491c851cf0b05a852f1faa805d7
        
        Applies a convolution over an input tensor where neighborhood relationships
        between elements are explicitly provided via an `indices` tensor.

        The output value of the layer with input size :math:`(N, C_{in}, L)` and output
        :math:`(N, C_{out}, L)` can be described as:

        .. math::

            \begin{array}{ll}
            out(N_i, C_{out_j})  = bias(C_{out_j})
                           + \sum_{{c}=0}^{C_{in}-1}
                             \sum_{{i}=0}^{L-1}
                             \sum_{{k}=0}^{K} weight(C_{out_j}, c, k) * input(N_i, c, indices(i, k))
            \end{array}

        where

        | `indices` is a L x K tensor, where `K` is the size of the convolution kernel,
        | providing the indices of the `K` neighbors of input element `i`.
        | A -1 entry means zero-padding.
        
        This layer creates an indexed convolution kernel that is convolved
        (actually cross-correlated) with the layer input to produce a tensor of
        outputs. If `use_bias` is True (and a `bias_initializer` is provided),
        a bias vector is created and added to the outputs. Finally, if
        `activation` is not `None`, it is applied to the outputs as well.
        
        Arguments (taken from tensorflow):
        indices: Matrix of indices.
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
                 indices,
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
        
        super(IndexedConv2D, self).__init__(trainable=True,name=name,activity_regularizer=activity_regularizer,**kwargs)
        self.logger = logging.getLogger(__name__ + '.IndexedConv2D')
        
        
        indices_neighbours = utils.neighbours_extraction(indices,kernel_type='Hex')
        self.indices, self.mask = utils.prepare_mask(indices_neighbours)
        self.indices_ = tf.convert_to_tensor(self.indices,name='ind')
        self.mask_ = tf.convert_to_tensor(self.mask,name='mask')
        
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
            raise ValueError('data_format must be channels_last or channels_first.')
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
                                                            
        kernel_shape = (self.kernel_size, self.kernel_size, in_channels, out_channels)
        self.kernel = self.add_variable(name='kernel',
                                        shape=kernel_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True,
                                        dtype=self.dtype)
        
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
        #TODO
        print("Start 'call' IndexedConv")
        nbatch = input.shape[0]
        print(input.shape)
        param = tf.constant([[[0.,0.,9.],[0.,1.,9.], [0.,2.,9.], [0.,3.,9.], [0.,4.,9.], [0.,5.,9.], [0.,6.,9.], [0.,7.,9.]],
                     [[1.,0.,9.],[1.,1.,9.], [1.,2.,9.], [1.,3.,9.], [1.,4.,9.], [1.,5.,9.], [1.,6.,9.], [1.,7.,9.]]]) # B x N x C(2 x 8 x 3)
        #param = tf.constant([[[1,10], [2,20], [3,30], [4,40], [5,50],[6,60], [7,70]],[[11,12], [21,22], [31,32], [41,42], [51,52],[61,62], [71,72]]],dtype=tf.float32)
        #param = tf.constant([[[0.,0.,9.],[0.,1.,9.], [0.,2.,9.], [0.,3.,9.], [0.,4.,9.], [0.,5.,9.], [0.,6.,9.],

        #             [[1.,0.,9.],[1.,1.,9.], [1.,2.,9.], [1.,3.,9.], [1.,4.,9.], [1.,5.,9.], [1.,6.,9.], [1.,7.,9.]]]) # B x N x C(2 x 8 x 3)
        print("param shape:     ",param.shape)

        #gather元を選ぶテンソル(2batch x 4点)
        #indices = tf.constant([[0, 0, 0, 0, 1, 2, 3],
        #                       [0, 0, 0, 1, 0, 3, 4],
        #                       [0, 0, 0, 2, 3, 0, 5],
        #                       [0, 1, 2, 3, 4, 5, 6],
        #                       [1, 0, 3, 4, 0, 6, 0],
        #                       [2, 3, 0, 5, 6, 0, 0],
        #                       [3, 4, 5, 6, 0, 0, 0]],
        #                       [[0, 0, 0, 0, 1, 2, 3],
        #                       [0, 0, 0, 1, 0, 3, 4],
        #                       [0, 0, 0, 2, 3, 0, 5],
        #                       [0, 1, 2, 3, 4, 5, 6],
        #                       [1, 0, 3, 4, 0, 6, 0],
        #                       [2, 3, 0, 5, 6, 0, 0],
        #                       [3, 4, 5, 6, 0, 0, 0]])    # B x N
        indices = tf.constant([[1,0,0,4,0,0,0,0,0],[1,3,4,6,0,0,0,0,0]])

        print("indices shape:   ",indices.shape)
        #gatherする
        result = tf.batch_gather(param, indices)
        result = tf.reshape(result,[result.shape[0],3,3,result.shape[-1]])
        print("result shape:    ",result.shape)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        print("param\n",sess.run(param))     #入力の確認
        print("indices\n",sess.run(indices)) #indiciesの確認
        print("result\n",sess.run(result))   #gatherの結果
        
        print("End 'call' IndexedConv")
        outputs=1
        return outputs
    
def indexed_conv2d(inputs,
                  indices,
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
    
    """ Functional interface for 2D indexed convolution layer (e.g. temporal convolution).
        
        Same arguments as in the IndexedConv2D class.
    """
    layer = IndexedConv2D(
                indices=indices,
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
