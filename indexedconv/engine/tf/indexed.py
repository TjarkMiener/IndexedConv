"""
indexed.py
========
Contain IndexedConv2D layer and IndexedAvgPool2D/IndexedMaxPool2D pooling class with functional interfaces.
"""


import tensorflow as tf
import logging
import indexedconv.utils as utils
import indexedconv
import time
import numpy as np
import torch 


# In[2]:
def neighbours_extraction(index_matrix, kernel_type='Hex', radius=2, stride=1, dilation=1, retina=False):
    """Builds the matrix of indices from an index matrix based on a kernel.
    The matrix of indices contains for each pixel of interest its neighbours, including itself.
    Args:
        index_matrix (torch.Tensor): Matrix of index for the images, shape(1, 1, matrix.size).
        kernel_type (str): The kernel shape, Hex for hexagonal Square for a square and Pool for a square of size 2.
        radius (int): The radius of the kernel.
        stride (int): The stride.
        dilation (int): The dilation. A dilation of 1 means no dilation.
        retina (bool): Whether to build a retina like kernel. If True, dilation must be 1.
    Returns:
        A torch.Tensor - the matrix of the neighbours.
    Example:
        >>> index_matrix = [[0, 1, -1], [2, 3, 4], [-1, 5, 6]]
        [[0, 1, -1],
        [2,  3, 4],
        [-1, 5, 6]]
        >>> kernel_type = 'Hex'
        >>> radius = 1
        >>> kernel
        [[1, 1, 0],
        [ 1, 1, 1],
        [ 0, 1, 1]]
        >>> stride = 1
        >>> neighbours = neighbours_extraction(index_matrix, kernel_type, radius, stride)
        [[-1, -1, -1,  0,  1,  2,  3],
        [ -1, -1,  0,  1, -1,  3,  4],
        [ -1,  0, -1,  2,  3, -1,  5],
        [  0,  1,  2,  3,  4,  5,  6],
        [  1, -1,  3,  4, -1,  6, -1],
        [  2,  3, -1,  5,  6, -1, -1],
        [  3,  4,  5,  6, -1, -1, -1]]
    """
    if retina:
        dilation = 1
    padding = radius * dilation * 2
    stride = stride
    bound = radius * dilation * 2 if radius > 0 else 1
    if kernel_type == 'Pool':
        kernel = np.ones((2, 2), dtype=bool)
        stride = 2
        bound = 1
        padding = 0
        center = 0
    elif retina:
        kernel = utils.build_kernel(kernel_type, 1, radius).astype(bool)
        for i in range(1, radius):
            sub_kernel = np.zeros_like(kernel).astype(bool)
            sub_kernel[i:sub_kernel.shape[0]-i, i:sub_kernel.shape[1]-i] = utils.build_kernel(kernel_type, 1, radius - i).astype(bool)
            kernel = kernel + sub_kernel
        center = int((np.count_nonzero(kernel) - 1) / 2)
    else:
        kernel = utils.build_kernel(kernel_type, radius, dilation).astype(bool)
        center = int((np.count_nonzero(kernel) - 1) / 2)

    neighbours = []

    idx_mtx = np.ones((index_matrix.size(-2)+padding, index_matrix.size(-1)+padding), dtype=int) * (-1)
    offset = int(padding/2)
    if offset == 0:
        idx_mtx = index_matrix[0, 0, :, :].numpy()
    else:
        idx_mtx[offset:-offset, offset:-offset] = index_matrix[0, 0, :, :].numpy()

    for i in range(radius, idx_mtx.shape[0]-bound-radius, stride):
        for j in range(radius, idx_mtx.shape[1]-bound-radius, stride):
            patch = idx_mtx[i:i+kernel.shape[0], j:j+kernel.shape[1]][kernel]
            if patch[center] == -1:
                continue
            neighbours.append(patch)

    neighbours = np.asarray(neighbours).T
    #neighbours = torch.from_numpy(neighbours).long()

    return neighbours

def create_index_matrix(nbRow, nbCol, injTable):
    r"""Creates the matrix of index of the pixels of the images of any shape stored as vectors.
    Args:
        nbRow (int): The number of rows of the index matrix.
        nbCol (int): The number of cols of the index matrix.
        injTable (numpy.array): The injunction table, i.e. the list of the position of every pixels of the vector image
            in a vectorized square image.
    Returns:
        A torch.Tensor containing the index of each pixel represented in a matrix.
    Example:
        >>> image = [0, 1, 2, 3, 4, 5, 6]  # hexagonal image stored as a vector
        >>> # in the hexagonal space                  0 1
        >>> #                                        2 3 4
        >>> #                                         5 6
        >>> # injunction table of the pixel position of a hexagonal image represented in the axial addressing system
        >>> injTable = [0, 1, 3, 4, 5, 7, 8]
        >>> index_matrix = [[0, 1, -1], [2, 3, 4], [-1, 5, 6]]
        [[0, 1, -1],
        [2,  3, 4],
        [-1, 5, 6]]
    """
    logger = logging.getLogger(__name__ + '.create_index_matrix')
    index_matrix = torch.full((int(nbRow), int(nbCol)), -1)
    for i, idx in enumerate(injTable):
        idx_row = int(idx // nbRow)
        idx_col = int(idx % nbCol)
        index_matrix[idx_row,idx_col] = i

    index_matrix.unsqueeze_(0)
    index_matrix.unsqueeze_(0)
    return index_matrix

def create_batch_index(batch, kernel_pixels):
    batch_column = tf.reshape(tf.range(batch),[batch,1,1,1])
    batch_column = tf.tile(batch_column,[1,1,kernel_pixels,1])
    zero_column = tf.reshape(tf.constant(0),[1,1,1,1])
    zero_column = tf.tile(zero_column, [batch,1,kernel_pixels,2])
    return tf.concat([batch_column,zero_column], axis=3)

class IndexedConv(tf.layers.Layer):
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
                 inputs,
                 filters,
                 kernel_size,
                 strides=(1,1),
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
        
        super(IndexedConv, self).__init__(trainable=True,name=name,activity_regularizer=activity_regularizer,**kwargs)
        self.logger = logging.getLogger(__name__ + '.IndexedConv')
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_radius = int((self.kernel_size-1)/2)
        
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
        strides = (strides,strides) if type(strides) != tuple else strides
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
        
        # padding
        self.P = int((self.kernel_size-1)/2) if self.padding == 'SAME' else 0
        #print('pad', self.P)
        self.pad = tf.constant([[0,0], [self.P, self.P], [self.P, self.P], [0,0]])
        
        input_dim = tf.TensorShape(input_shape).as_list()
        #print(input_dim)
        if self.data_format == 'NHWC':
            self.height = input_dim[1] + self.P*2
            self.width = input_dim[2] + self.P*2
            self.in_channels = input_dim[3]
        elif self.data_format == 'NCHW':
            self.height = input_dim[2] + self.P*2
            self.width = input_dim[3] + self.P*2
            self.in_channels = input_dim[1]
        self.out_channels = self.filters
        self.batch_size = input_dim[0]
        self.in_size = input_dim[2] + self.P*2
        
        indices_ = create_index_matrix(self.height,self.width, np.arange(self.height*self.width))
        #print('indices', '\n', indices_)
        indices_neighbours = neighbours_extraction(indices_, kernel_type='Hex', stride=self.strides[2], radius=self.kernel_radius)
        #print('indices_neighbours', '\n',indices_neighbours)
        
        self.indices, self.mask = utils.prepare_mask(indices_neighbours)
        #print('self.indices', '\n',self.indices)
        #print('self.mask', '\n',self.mask)
        self.indices_ = tf.cast(tf.convert_to_tensor(indices_neighbours,name='ind'), tf.int32)
        self.kernel_pixels = self.indices_.get_shape().as_list()[0]
        self.out_size = self.indices_.get_shape().as_list()[1] # just works for images with same height and widht
        #print('self.indices_', '\n',self.indices_)
        self.mask_ = tf.convert_to_tensor(self.mask,name='mask')
        #print('self.mask_', '\n',self.mask_)
        
          
        # For now the mask are hardcoded (to run on gpu)! Later with build_kernel() from utils.py!
        if self.kernel_size == 3:
            self.mask = tf.Variable([[1.,1.,0.],[1.,1.,1.],[0.,1.,1.]], tf.float32)
        elif self.kernel_size == 5:
            self.mask = tf.Variable([[1.,1.,1.,0.,0.],[1.,1.,1.,1.,0.],[1.,1.,1.,1.,1.],[0.,1.,1.,1.,1.],[0.,0.,1.,1.,1.]], tf.float32)
        else:
            raise ValueError('kernel_size must be 3 or 5.')
                                                            
        kernel_shape = (self.in_channels, self.out_channels, self.kernel_size, self.kernel_size)
        self.kernel = self.add_variable(name='kernel',
                                        shape=kernel_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True,
                                        dtype=self.dtype)
        self.kernel_hex = self.kernel * self.mask
        #self.kernel_hex = tf.transpose(self.kernel, perm=[2, 3, 0, 1])

        if self.use_bias:
            self.bias = self.add_variable(name='bias',
                                              shape=(self.out_channels,),
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint,
                                          trainable=True,
                                          dtype=self.dtype)
        else:
            self.bias = None
        
        self.mask2 = utils.build_kernel('Hex', radius = self.kernel_radius)
        self.mask2 = [self.mask2 != 0] 
        self.mask2 = np.tile(self.mask2, (self.in_channels*self.out_channels,1))
        self.mask2 = np.reshape(self.mask2, (self.in_channels, self.out_channels,1,self.kernel_size**2))

    def call(self, inputs):
        
        inputs = tf.pad(inputs, self.pad)
        print('inputs:', inputs)
        batch = tf.shape(inputs)[0]  # needed to build the graph
        inputs_ = tf.layers.flatten(inputs)    # flatten the inputs
        print('inputs_:', inputs_)
        inputs_ = tf.reshape(inputs_, [batch,1, self.in_size**2,self.in_channels])
        print('inputs_:', inputs_)

        indices1 = tf.transpose(self.indices_, [1,0])
        indices2 = tf.reshape(indices1, [1,self.out_size,self.kernel_pixels,1])
        indices2 = tf.tile(indices2, [batch,1,1,1])
        indices2 = tf.pad(indices2, tf.constant([[0,0], [0, 0], [0, 0], [2,0]]), constant_values=0)
        indices2 = tf.add(indices2, create_batch_index(batch, self.kernel_pixels))

        col = tf.gather_nd(inputs_, indices2)
        col_ =  tf.cast(col, dtype=tf.float32)

        kernel = tf.reshape(self.kernel_hex, [self.in_channels, self.out_channels,1,self.kernel_size**2])
        kernel = tf.boolean_mask(kernel, self.mask2)
        kernel_ = tf.cast(tf.reshape(kernel, [1, self.kernel_pixels,self.in_channels, self.out_channels]), dtype=tf.float32)
        
        outputs = tf.nn.conv2d(col_, kernel_, strides = (1,1,1,1), padding='VALID')
        outputs = tf.reshape(outputs,[batch,int(self.out_size**(1/2)),int(self.out_size**(1/2)),self.out_channels])
        
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias,data_format=self.data_format)
        if self.activation is not None:
                return self.activation(outputs)
            
        return outputs
    
def indexed_conv2d(inputs,
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
    layer = IndexedConv(
                inputs=inputs,
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


class IndexedAveragePool2d(tf.layers.Layer):
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
                 pool_size=(3,3),
                 strides=(3,3),
                 padding='valid',
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        
        super(IndexedAveragePool2d, self).__init__()
        self.logger = logging.getLogger(__name__ + '.IndexedAveragePool2d')
        
        self.pool_size = pool_size if type(pool_size) == int else pool_size[0]
        self.pool_radius = int((self.pool_size-1)/2)
        
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
        strides = (strides,strides) if type(strides) == int else None
        if data_format == 'channels_last':
            self.data_format = 'NHWC'
            self.strides = (1,) + strides + (1,)
        elif data_format == 'channels_first':
            self.data_format = 'NCHW'
            self.strides = (1, 1) + strides
        else:
            raise ValueError('data_format must be channels_last or channels_first.')


    def build(self, input_shape):
        # padding
        self.P = int((self.pool_size-1)/2) if self.padding == 'SAME' else 0
        self.pad = tf.constant([[0,0], [self.P, self.P], [self.P, self.P], [0,0]])
        
        input_dim = tf.TensorShape(input_shape).as_list()
        print(input_dim)
        if self.data_format == 'NHWC':
            self.height = input_dim[1] + self.P*2
            self.width = input_dim[2] + self.P*2
            self.in_channels = input_dim[3]
        elif self.data_format == 'NCHW':
            self.height = input_dim[2] + self.P*2
            self.width = input_dim[3] + self.P*2
            self.in_channels = input_dim[1]
        self.out_channels = self.in_channels
        self.batch_size = input_dim[0]
        self.in_size = input_dim[2] + self.P*2
        
        indices_ = create_index_matrix(self.height,self.width, np.arange(self.height*self.width))
        indices_neighbours = neighbours_extraction(indices_, kernel_type='Hex', stride=self.strides[2], radius=self.pool_radius)
        
        self.indices, self.mask = utils.prepare_mask(indices_neighbours)
        self.indices_ = tf.cast(tf.convert_to_tensor(self.indices,name='ind'), tf.int32)
        self.kernel_pixels = self.indices_.get_shape().as_list()[0]
        print('self.indices', '\n',self.indices_)
        self.out_size = self.indices_.get_shape().as_list()[1] # just works for images with same height and widht
        self.mask_ = tf.convert_to_tensor(self.mask,name='mask')
        

    def call(self, inputs):
        
        inputs = tf.pad(inputs, self.pad)
        batch = tf.shape(inputs)[0]  # needed to build the graph
        inputs_ = tf.layers.flatten(inputs)    # flatten the inputs
        inputs_ = tf.reshape(inputs_, [batch,1, self.in_size**2,self.in_channels])
        
        indices1 = self.indices_
        indices1 = tf.transpose(indices1, [1,0])
        indices2 = tf.reshape(indices1, [1,self.out_size,self.kernel_pixels,1])
        indices2 = tf.tile(indices2, [batch,1,1,1])
        indices2 = tf.pad(indices2, tf.constant([[0,0], [0, 0], [0, 0], [2,0]]), constant_values=0)
        indices2 = tf.add(indices2, create_batch_index(batch, self.kernel_pixels))

        col = tf.gather_nd(inputs_, indices2)
        col_ = tf.cast(col, dtype=tf.float32)/self.kernel_pixels
        
        outputs = tf.math.reduce_sum(col_, axis=2, keepdims=True)
        outputs = tf.reshape(outputs,[batch,int(self.out_size**(1/2)),int(self.out_size**(1/2)),self.out_channels])
        
        return outputs


def indexed_avgpool2d(inputs,
                      pool_size=(3,3),
                      strides=(3,3),
                      padding='valid',
                      data_format='channels_last',
                      name=None):
    """ Functional interface for 2D indexed average pooling layer.
        
        Same arguments as in the IndexedAvgPool2D class.
    """
    avgpool = IndexedAveragePool2d(
                    pool_size=pool_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    name=name)
                              
    return avgpool.apply(inputs)



class IndexedMaxPool2d(tf.layers.Layer):
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
                 pool_size=(3,3),
                 strides=(3,3),
                 padding='valid',
                 data_format='channels_last',
                 name=None,
                 **kwargs):
        
        super(IndexedMaxPool2d, self).__init__()
        self.logger = logging.getLogger(__name__ + '.IndexedMaxPool2d')
        
        self.pool_size = pool_size if type(pool_size) == int else pool_size[0]
        self.pool_radius = int((self.pool_size-1)/2)
        
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
        strides = (strides,strides) if type(strides) != tuple else strides
        if data_format == 'channels_last':
            self.data_format = 'NHWC'
            self.strides = (1,) + strides + (1,)
        elif data_format == 'channels_first':
            self.data_format = 'NCHW'
            self.strides = (1, 1) + strides
        else:
            raise ValueError('data_format must be channels_last or channels_first.')


    def build(self, input_shape):
        # padding
        self.P = int((self.pool_size-1)/2) if self.padding == 'SAME' else 0
        self.pad = tf.constant([[0,0], [self.P, self.P], [self.P, self.P], [0,0]])
        
        input_dim = tf.TensorShape(input_shape).as_list()
        if self.data_format == 'NHWC':
            self.height = input_dim[1] + self.P*2
            self.width = input_dim[2] + self.P*2
            self.in_channels = input_dim[3]
        elif self.data_format == 'NCHW':
            self.height = input_dim[2] + self.P*2
            self.width = input_dim[3] + self.P*2
            self.in_channels = input_dim[1]
        self.out_channels = self.in_channels
        self.batch_size = input_dim[0]
        self.in_size = input_dim[2] + self.P*2
        
        indices_ = create_index_matrix(self.height,self.width, np.arange(self.height*self.width))
        indices_neighbours = neighbours_extraction(indices_, kernel_type='Hex', stride=self.strides[2], radius=self.pool_radius)
        
        self.indices, self.mask = utils.prepare_mask(indices_neighbours)
        self.indices_ = tf.cast(tf.convert_to_tensor(self.indices,name='ind'), tf.int32)
        self.kernel_pixels = self.indices_.get_shape().as_list()[0]
        self.out_size = self.indices_.get_shape().as_list()[1] # just works for images with same height and widht
        

    def call(self, inputs):
        
        inputs = tf.pad(inputs, self.pad)
        batch = tf.shape(inputs)[0]  # needed to build the graph
        inputs_ = tf.layers.flatten(inputs)    # flatten the inputs
        inputs_ = tf.reshape(inputs_, [batch,1, self.in_size**2,self.in_channels])
        
        indices1 = self.indices_
        indices1 = tf.transpose(indices1, [1,0])
        indices2 = tf.reshape(indices1, [1,self.out_size,self.kernel_pixels,1])
        indices2 = tf.tile(indices2, [batch,1,1,1])
        indices2 = tf.pad(indices2, tf.constant([[0,0], [0, 0], [0, 0], [2,0]]), constant_values=0)
        indices2 = tf.add(indices2, create_batch_index(batch, self.kernel_pixels))

        col = tf.gather_nd(inputs_, indices2)
        col_ = tf.cast(col, dtype=tf.float32)
        
        outputs = tf.math.reduce_max(col_, axis=2, keepdims=True)
        outputs = tf.reshape(outputs,[batch,int(self.out_size**(1/2)),int(self.out_size**(1/2)),self.out_channels])
        
        return outputs


def indexed_maxpool2d(inputs,
                      pool_size=(3,3),
                      strides=(3,3),
                      padding='valid',
                      data_format='channels_last',
                      name=None):
    """ Functional interface for 2D indexed average pooling layer.
        
        Same arguments as in the IndexedAvgPool2D class.
    """
    maxpool = IndexedMaxPool2d(
                    pool_size=pool_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    name=name)
                              
    return maxpool.apply(inputs)

