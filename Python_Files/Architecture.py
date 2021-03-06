
# coding: utf-8

# In[1]:


import tensorflow as tf; tf.set_random_seed(42);


# In[2]:


def conv_2d(x, kernel_size = 4, stride = 1, out_channels = 64, is_conv = True, is_norm = True, normalization = 'instance', 
            is_act = True, activation = 'lrelu', leak_param = 0.2, padding = 'VALID'):
    
    """
    Arguments:
    x:             Input Tensor
    kernel_size:   Integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window
    stride:        Integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width
    out_channels:  Integer, the dimensionality of the output space
    is_conv:       Boolean, whether to perform convolution or not
    is_norm:       Boolean, whether to normalize the data or not
    normalization: Type of normalization, supports batch and instance normalization
    is_act:        Boolean, whether to apply non-linear activation functions or not
    activation:    Type of activation function, supports Leaky_ReLU, ReLU and Sigmoid and Tanh
    leak_param:    Integer, Leakiness to use in case of Leaky ReLU
    padding:       Zero padding around the image, supports VALID and SAME
    
    Returns:
    x:             Output Tensor
    """
    
#   Apply Non-linearities
    if is_act == True:
        if activation == 'lrelu': x = tf.nn.leaky_relu(x, leak_param)
        elif activation == 'relu': x = tf.nn.relu(x);
        elif activation == 'sigmoid': x = tf.nn.sigmoid(x)
        elif activation == 'tanh': x = tf.nn.tanh(x)
        else: print('Check you activation function again in Conv block!!')
    
#   Apply Convolution after doing mirror padding keeping the output size same as that of an input
    if is_conv == True:
        if(padding == 'VALID'):     
            x = tf.pad(x, [[0, 0], [(kernel_size-1)//2, (kernel_size-1)//2], [(kernel_size-1)//2, (kernel_size-1)//2], [0, 0]],
                       mode = 'REFLECT')         
        x = tf.layers.conv2d(x, filters = out_channels, kernel_size = kernel_size, strides = stride, padding = padding, 
                             use_bias = False, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d())
        
#   Apply Normalization
    if is_norm == True:
        if(normalization == 'instance'): x = tf.contrib.layers.instance_norm(x, epsilon = 1e-6)
        elif(normalization == 'batch'): x = tf.contrib.layers.batch_norm(x, epsilon = 1e-5, training = train_mode, 
                                                                         momentum = 0.9)
        else: print('Check your normalization function again !!')
    
    return x

  
def conv_2d_transpose(x, kernel_size, stride, out_channels, is_deconv = True, is_act = True, activation = 'relu', 
                      leak_param = 0.2, is_norm = True, normalization = 'instance', is_dropout = False, dropout = 0.5):
  
    """
    Arguments:
    x:             Input Tensor
    kernel_size:   Integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window
    stride:        Integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width
    out_channels:  Integer, the dimensionality of the output space
    is_deconv:       Boolean, whether to perform convolution or not
    is_norm:       Boolean, whether to normalize the data or not
    normalization: Type of normalization, supports batch and instance normalization
    is_act:        Boolean, whether to apply non-linear activation functions or not
    activation:    Type of activation function, supports Leaky_ReLU, ReLU and Sigmoid and Tanh
    leak_param:    Integer, Leakiness to use in case of Leaky ReLU
    is_dropout:    Boolean, whether to apply add dropout layer or not
    dropout:       A scalar Tensor with the same type as x. The probability that each element is kept.
    
    Returns:
    x:             Output Tensor
    """
    
#   Apply Non-linearities
    if is_act == True:
        if activation == 'lrelu': x = tf.nn.leaky_relu(x, leak_param)
        elif activation == 'relu': x = tf.nn.relu(x)
        elif activation == 'sigmoid': x = tf.nn.sigmoid(x)
        elif activation == 'tanh': x = tf.nn.tanh(x)
        else: print('Check you activation function again in Conv block!!')

#   Apply Deconvolution
    if is_deconv == True:
        x = tf.layers.conv2d_transpose(x, filters = out_channels, kernel_size = kernel_size, strides = stride, 
            padding = 'SAME', use_bias = False, kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d())

#   Apply Normalization
    if is_norm == True:
        if(normalization == 'instance'): x = tf.contrib.layers.instance_norm(x, epsilon = 1e-6)
        elif(normalization == 'batch'): x = tf.contrib.layers.batch_norm(x, epsilon = 1e-5, training = train_mode,
                                                                        momentum = 0.9)
        else: print('Check your normalization function again !!')
    
#   Add Dropout layer
    if is_dropout == True: x = tf.nn.dropout(x, keep_prob = 1-dropout)
    
#   NOTE: We don't want to turn off the dropout at the validation/test time in order to bring out the diversity in the images
#         generated by Generator. Moreover, instance normalization works best in case of GANs. Due to all these reasons, 
#         I didn't use placeholders for the boolean (dropout mode)!!
    
    return x


# In[3]:


def discriminator_patch_gan(input_ten, target_ten, out_channels = 64, use_sigmoid = False):
    
    """
    Arguments:
    input_ten:    Input Tensor
    target_ten:   Target Tensor
    out_channels: Integer, the dimensionality of the output space
    use_sigmoid:  Boolean, whether to use sigmoid at the last layer or not
    
    Returns:
    x:            Output Tensor
    """
    
#   Patch_gan discriminator is Conditional GAN, so training it requires conditionality on the input_tensor 
#   [image from the first domain which should get converted to second domain]
    x = tf.concat([input_ten, target_ten], axis = 3)

#   Number of convolutional layers used should be such that, the receptive field is 70*70
#   Authors tried out many different possible variants, but this config worked best for them!!
#   No normalization in the first layer

    x = conv_2d(x, 4, 2, out_channels, is_norm = False, is_act = False)
    x = conv_2d(x, 4, 2, out_channels*2); x = conv_2d(x, 4, 2, out_channels*4)
    x = conv_2d(x, 4, 2, out_channels*8); x = conv_2d(x, 4, 1, out_channels*8)
    
#   No normalization in the last layer
    x = conv_2d(x, 4, 1, 1, is_norm = False)
    
#   Only use sigmoid when probabilities are needed [generally we need un-normalized logits for the loss function]
    if use_sigmoid == True:
        x = tf.nn.sigmoid(x); print('Sigmoid activation in the discriminator')
    
    return x


# In[4]:


def generator_unet_128(input_ten, out_channels = 64):
  
    """
    Arguments:
    input_ten:    Input Tensor
    out_channels: Integer, the dimensionality of the output space
    
    Returns:
    o_2:          Output Tensor with tanh activation function applied
    """
    
#   Two layers of vanilla convolution in the initial part of the architecture just to increase the receptive field
#   No batch norm layer just after the input layer!
    i_1 = conv_2d(input_ten, 3, 1, out_channels*(1), is_norm = False, is_act = False)
    i_2 = conv_2d(i_1, 3, 1, out_channels*(1));
    
#   Architecture of the encoder [All are stride 2 convolutions]
    e_1 = conv_2d(i_2, 4, 2, out_channels*(1)); e_2 = conv_2d(e_1, 4, 2, out_channels*(2))
    e_3 = conv_2d(e_2, 4, 2, out_channels*(4)); e_4 = conv_2d(e_3, 4, 2, out_channels*(8))
    e_5 = conv_2d(e_4, 4, 2, out_channels*(8)); e_6 = conv_2d(e_5, 4, 2, out_channels*(8))
    e_7 = conv_2d(e_6, 4, 2, out_channels*(8), is_norm = False)

#   Architecture of the decoder [Adding Skip connections between the layers of encoder and decoder of same dimensionality]
#   NOTE: U-Net preserves the fine granularity in the output that will otherwise be lost if using stride 2 conv layers!! 
    d_1 = conv_2d_transpose(e_7, 4, 2, out_channels*(8), is_dropout = True); d_1 = tf.concat([d_1, e_6], axis = 3)
    d_2 = conv_2d_transpose(d_1, 4, 2, out_channels*(8), is_dropout = True); d_2 = tf.concat([d_2, e_5], axis = 3)
    d_3 = conv_2d_transpose(d_2, 4, 2, out_channels*(8), is_dropout = True); d_3 = tf.concat([d_3, e_4], axis = 3)
    d_4 = conv_2d_transpose(d_3, 4, 2, out_channels*(4)); d_4 = tf.concat([d_4, e_3], axis = 3)
    d_5 = conv_2d_transpose(d_4, 4, 2, out_channels*(2)); d_5 = tf.concat([d_5, e_2], axis = 3)
    d_6 = conv_2d_transpose(d_5, 4, 2, out_channels*(1)); d_6 = tf.concat([d_6, e_1], axis = 3)
    d_7 = conv_2d_transpose(d_6, 4, 2, out_channels*(1))
    
#   Last few vanilla convolutional layers without any stridding
    o_1 = conv_2d_transpose(d_7, 4, 1, out_channels*(1)); o_2 = conv_2d_transpose(o_1, 4, 1, 3, is_norm = False)
#   Tanh activation function to ensure the range of output to be inbetween -1 and 1!
    o_2 = tf.nn.tanh(o_2)

    return o_2

  
def generator_unet_256(input_ten, out_channels = 64):
    
    """
    Arguments:
    input_ten:    Input Tensor
    out_channels: Integer, the dimensionality of the output space
    
    Returns:
    o_3:          Output Tensor with tanh activation function applied
    """
    
#   Few layers of vanilla convolution in the initial part of the architecture just to increase the receptive field
#   No batch norm layer just after the input layer!
    i_1 = conv_2d(input_ten, 3, 1, out_channels*(1), is_norm = False, is_act = False)
    i_2 = conv_2d(i_1, 3, 1, out_channels*(1)); i_3 = conv_2d(i_2, 3, 1, out_channels*(2))

#   Architecture of the encoder [All are stride 2 convolutions]
    e_1 = conv_2d(i_3, 4, 2, out_channels*(2)); e_2 = conv_2d(e_1, 4, 2, out_channels*(4))
    e_3 = conv_2d(e_2, 4, 2, out_channels*(4)); e_4 = conv_2d(e_3, 4, 2, out_channels*(8))
    e_5 = conv_2d(e_4, 4, 2, out_channels*(8)); e_6 = conv_2d(e_5, 4, 2, out_channels*(8))
    e_7 = conv_2d(e_6, 4, 2, out_channels*(8))
    e_8 = conv_2d(e_7, 4, 2, out_channels*(8), is_norm = False)

#   Architecture of the decoder [Adding Skip coonections between the layers of encoder and decoder of same dimensionality]
#   NOTE: U-Net preserves the fine granularity in the output that will otherwise be lost if using stride 2 conv layers!!
    d_1 = conv_2d_transpose(e_8, 4, 2, out_channels*(8), is_dropout = True); d_1 = tf.concat([d_1, e_7], axis = 3)
    d_2 = conv_2d_transpose(d_1, 4, 2, out_channels*(8), is_dropout = True); d_2 = tf.concat([d_2, e_6], axis = 3)
    d_3 = conv_2d_transpose(d_2, 4, 2, out_channels*(8), is_dropout = True); d_3 = tf.concat([d_3, e_5], axis = 3)
    d_4 = conv_2d_transpose(d_3, 4, 2, out_channels*(8)); d_4 = tf.concat([d_4, e_4], axis = 3)
    d_5 = conv_2d_transpose(d_4, 4, 2, out_channels*(4)); d_5 = tf.concat([d_5, e_3], axis = 3)
    d_6 = conv_2d_transpose(d_5, 4, 2, out_channels*(4)); d_6 = tf.concat([d_6, e_2], axis = 3)
    d_7 = conv_2d_transpose(d_6, 4, 2, out_channels*(2)); d_7 = tf.concat([d_7, e_1], axis = 3)
    d_8 = conv_2d_transpose(d_7, 4, 2, out_channels*(2))
    
#   Last few vanilla convolutional layers without any stridding
    o_1 = conv_2d_transpose(d_8, 4, 1, out_channels*(1)); o_2 = conv_2d_transpose(o_1, 4, 1, out_channels*(1)) 
    o_3 = conv_2d_transpose(o_2, 4, 1, 3, is_norm = False); 
    
#   Tanh activation function to ensure the range of output to be inbetween -1 and 1!
    o_3 = tf.nn.tanh(o_3)

    return o_3


def generator_unet_512(input_ten, out_channels = 64):
    
    """
    Arguments:
    input_ten:    Input Tensor
    out_channels: Integer, the dimensionality of the output space
    
    Returns:
    o_4:          Output Tensor with tanh activation function applied
    """
    
#   Few layers of vanilla convolution in the initial part of the architecture just to increase the receptive field
#   No batch norm layer just after the input layer!
    i_1 = conv_2d(input_ten, 3, 1, out_channels*(1), is_norm = False, is_act = False)
    i_2 = conv_2d(i_1, 3, 1, out_channels*(1)); i_3 = conv_2d(i_2, 3, 1, out_channels*(2))
    i_4 = conv_2d(i_3, 3, 1, out_channels*(2))
    
#   Architecture of the encoder [All are stride 2 convolutions]
    e_1 = conv_2d(i_4, 4, 2, out_channels*(4)); e_2 = conv_2d(e_1, 4, 2, out_channels*(4))
    e_3 = conv_2d(e_2, 4, 2, out_channels*(4)); e_4 = conv_2d(e_3, 4, 2, out_channels*(8))
    e_5 = conv_2d(e_4, 4, 2, out_channels*(8)); e_6 = conv_2d(e_5, 4, 2, out_channels*(8))
    e_7 = conv_2d(e_6, 4, 2, out_channels*(8)); e_8 = conv_2d(e_7, 4, 2, out_channels*(16))
    e_9 = conv_2d(e_8, 4, 2, out_channels*(16), is_norm = False)

#   Architecture of the decoder [Adding Skip coonections between the layers of encoder and decoder of same dimensionality]
#   NOTE: U-Net preserves the fine granularity in the output that will otherwise be lost if using stride 2 conv layers!! 
    d_1 = conv_2d_transpose(e_9, 4, 2, out_channels*(16), is_dropout = True); d_1 = tf.concat([d_1, e_8], axis = 3)
    d_2 = conv_2d_transpose(d_1, 4, 2, out_channels*(8), is_dropout = True); d_2 = tf.concat([d_2, e_7], axis = 3)
    d_3 = conv_2d_transpose(d_2, 4, 2, out_channels*(8), is_dropout = True); d_3 = tf.concat([d_3, e_6], axis = 3)
    d_4 = conv_2d_transpose(d_3, 4, 2, out_channels*(8), is_dropout = True); d_4 = tf.concat([d_4, e_5], axis = 3)
    d_5 = conv_2d_transpose(d_4, 4, 2, out_channels*(8)); d_5 = tf.concat([d_5, e_4], axis = 3)
    d_6 = conv_2d_transpose(d_5, 4, 2, out_channels*(4)); d_6 = tf.concat([d_6, e_3], axis = 3)
    d_7 = conv_2d_transpose(d_6, 4, 2, out_channels*(4)); d_7 = tf.concat([d_7, e_2], axis = 3)
    d_8 = conv_2d_transpose(d_7, 4, 2, out_channels*(4)); d_8 = tf.concat([d_8, e_1], axis = 3)
    d_9 = conv_2d_transpose(d_8, 4, 2, out_channels*(2))
    
#   Last few vanilla convolutional layers without any stridding
    o_1 = conv_2d_transpose(d_9, 4, 1, out_channels*(2)); o_2 = conv_2d_transpose(o_1, 4, 1, out_channels*(1))
    o_3 = conv_2d_transpose(o_2, 4, 1, out_channels*(1)); o_4 = conv_2d_transpose(o_3, 4, 1, 3, is_norm = False)
    
#   Tanh activation function to ensure the range of output to be inbetween -1 and 1!
    o_4 = tf.nn.tanh(o_4)

    return o_4

