import tensorflow as tf
        

 
def prelu(x):
    with tf.name_scope('PRELU'):
        _alpha = tf.get_variable('prelu', shape=x.get_shape()[-1], dtype = x.dtype, initializer=tf.constant_initializer(0.0))
    return tf.maximum(0.0, x) + _alpha * tf.minimum(0.0, x)



def conv2d(input, name, num_output_channels=None, kernel_size=3, kernel_size2=None, strides=[1,1,1,1], padding='SAME', uniform=True, square_kernel=True, act='prelu', reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        if padding == 'SYMMETRIC':
            padding_size = int(kernel_size / 2)
            if square_kernel: padding_size2 = padding_size
            else: padding_size2 = int(kernel_size2 / 2)
            input = tf.pad(input, paddings=tf.constant([[0,0], [padding_size,padding_size], [padding_size2,padding_size2], [0,0]]), mode='SYMMETRIC')
            padding = 'VALID'
        if padding == 'REFLECT':
            padding_size = int(kernel_size / 2)
            if square_kernel: padding_size2 = padding_size
            else: padding_size2 = int(kernel_size2 / 2)
            input = tf.pad(input, paddings=tf.constant([[0,0], [padding_size,padding_size], [padding_size2,padding_size2], [0,0]]), mode='REFLECT')
            padding = 'VALID'            

        num_in_channels = input.get_shape()[-1].value
        if square_kernel: kernel_shape = [kernel_size, kernel_size, num_in_channels, num_output_channels]
        else: kernel_shape = [kernel_size, kernel_size2, num_in_channels, num_output_channels]
            
        biases = tf.get_variable('biases', shape=[num_output_channels], initializer=tf.constant_initializer(0.1))
        kernel =  tf.get_variable('weights', shape=kernel_shape, initializer=tf.contrib.layers.xavier_initializer(uniform=uniform))
                       
        outputs = tf.nn.conv2d(input, kernel, strides=strides, padding=padding)
        outputs = tf.nn.bias_add(outputs, biases)
        
        if act == 'prelu': outputs = prelu(outputs)
        elif act == 'relu': outputs = tf.nn.relu(outputs)
        elif act == 'tanh': outputs = tf.nn.tanh(outputs)
        elif act == 'sigmoid': outputs = tf.sigmoid(outputs)
        elif act == 'leakyrelu': outputs = tf.nn.leaky_relu(outputs)
        elif act == None: pass
        return outputs      
    
    
    
def pool2d(input, kernel_size, stride, name, padding='SAME', use_avg=True, reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        if use_avg: 
            return tf.nn.avg_pool(input, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding, name = name)
        else: 
            return tf.nn.max_pool(input, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding, name = name)
        


def fully_connected(input, num_outputs, name, act='relu', reuse=False):           
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        
        num_input_units = input.get_shape()[-1].value
        kernel_shape = [num_input_units, num_outputs]
        kernel = tf.get_variable('weights', shape=kernel_shape, initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', shape=[num_outputs], initializer=tf.constant_initializer(0.1))
        
        outputs = tf.matmul(input, kernel)
        outputs = tf.nn.bias_add(outputs, biases)

        if act == 'prelu': outputs = prelu(outputs)
        elif act == 'relu': outputs = tf.nn.relu(outputs)
        elif act == 'tanh': outputs = tf.nn.tanh(outputs)
        elif act == 'sigmoid': outputs = tf.sigmoid(outputs)
        elif act == 'leakyrelu': outputs = tf.nn.leaky_relu(outputs)
        elif act == None: pass
        return outputs
    
   
    
def inception_P(input, nbS1, nbS2, name, output_name, without_kernel_5=False, act='prelu', reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
        s1_0 = conv2d(input=input, num_output_channels=nbS1, kernel_size=1, act=act, name=name+'S1_0')
        s2_0 = conv2d(input=s1_0, num_output_channels=nbS2, kernel_size=3, act=act, name=name+'S2_0')
        s1_2 = conv2d(input=input, num_output_channels=nbS1, kernel_size=1, act=act, name=name+'S1_2')
        pool0 = pool2d(input=s1_2, kernel_size=2, stride=1, name=name+'pool0', use_avg=True)
        if not(without_kernel_5):
            s1_1 = conv2d(input=input, num_output_channels=nbS1, kernel_size=1, act=act, name=name+'S1_1')
            s2_1 = conv2d(input=s1_1, num_output_channels=nbS2, kernel_size=5, act=act, name=name+'S2_1')
        s2_2 = conv2d(input=input, num_output_channels=nbS2, kernel_size=1, act=act, name=name+'S2_2')

        if not(without_kernel_5):
            outputs = tf.concat(values=[s2_2, s2_1, s2_0, pool0], name=output_name, axis=3)
        else:
            outputs = tf.concat(values=[s2_2, s2_0, pool0], name=output_name, axis=3)
    return outputs



def inception_T(input, kernels, name, act='relu', reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
        a0 = conv2d(input=input, num_output_channels=int(kernels*0.65), kernel_size=1, name='a0', act=act)
        a1 = conv2d(input=a0, num_output_channels=kernels, kernel_size=5, name='a1', act=act)
        b0 = conv2d(input=input, num_output_channels=int(kernels*0.65), kernel_size=1, name='b0', act=act)
        b1 = conv2d(input=b0, num_output_channels=kernels, kernel_size=3, name='b1', act=act)
        c0 = conv2d(input=input, num_output_channels=int(kernels*0.65), kernel_size=1, name='c0', act=act)
        c1 = pool2d(input=c0, kernel_size=2, name='c1', stride=1, use_avg=True)
        d1 = conv2d(input=input, num_output_channels=int(kernels*0.7), kernel_size=1, name='d1', act=act)
    return tf.concat([a1, b1, c1, d1], 3)


    
def z_estimator(input, reddening, bins, name, net=0, multi_r_out=False, num_multi_r_out=None, reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
       
        if net != 1: act = 'prelu'
        else: act = 'relu'
         
        def CNN(name):
            with tf.variable_scope(name):
                if net == 0:  #Net_P             
                    conv0 = conv2d(input=input, num_output_channels=64, kernel_size=5, name='conv0', act=act)
                    conv0p = pool2d(input=conv0, kernel_size=2, stride=2, name='conv0p', use_avg=True)
                    i0 = inception_P(conv0p, 48, 64, name='I0_', output_name='inception_P0', act=act)
                    i1 = inception_P(i0, 64, 92, name='I1_', output_name='inception_P1', act=act)
                    i1p = pool2d(input=i1, kernel_size=2, name='inception_P1p', stride=2, use_avg=True)
                    i2 = inception_P(i1p, 92, 128, name='I2_', output_name='inception_P2', act=act)
                    i3 = inception_P(i2, 92, 128, name='I3_', output_name='inception_P3', act=act)
                    i3p = pool2d(input=i3, kernel_size=2, name='inception_P3p', stride=2, use_avg=True)
                    i4 = inception_P(i3p, 92, 128, name='I4_', output_name='inception_P4', act=act, without_kernel_5=True)
                    flat = tf.layers.Flatten()(i4)
                    concat = tf.concat([flat, tf.expand_dims(reddening, -1)], 1)

                elif net == 1:  #Net_T
                    conv1 = conv2d(input=input, num_output_channels=96, kernel_size=5, name='conv1', act=act)
                    conv2 = conv2d(input=conv1, num_output_channels=96, kernel_size=3, name='conv2', act='tanh')
                    pool2 = pool2d(input=conv2, kernel_size=2, stride=2, name='pool2', use_avg=True)
                    inc1 = inception_T(input=pool2, kernels=156, name='inc1', act=act)
                    inc2 = inception_T(input=inc1, kernels=156, name='inc2', act=act)
                    inc2b = inception_T(input=inc2, kernels=156, name='inc2b', act=act)
                    pool3 = pool2d(input=inc2b, kernel_size=2, stride=2, name='pool3', use_avg=True)
                    inc3 = inception_T(input=pool3, kernels=156, name='inc3', act=act)
                    inc3b = inception_T(input=inc3, kernels=156, name='inc3b', act=act)
                    pool4 = pool2d(input=inc3b, kernel_size=2, stride=2, name='pool4', use_avg=True)
                    inc4 = inception_T(input=pool4, kernels=156, name='inc4', act=act)
                    conv5 = conv2d(input=inc4, num_output_channels=96, kernel_size=3, name='conv5', act=act, padding='VALID')
                    conv6 = conv2d(input=conv5, num_output_channels=96, kernel_size=3, name='conv6', act=act, padding='VALID')
                    conv7 = conv2d(input=conv6, num_output_channels=96, kernel_size=3, name='conv7', act=act, padding='VALID')
                    dede = pool2d(input=conv7, kernel_size=2, stride=1, name='dede', use_avg=True)
                    flat = tf.layers.Flatten()(dede)
                    concat = tf.concat([flat, tf.expand_dims(reddening, -1)], 1)
                    
                elif net == 2:  #Net_S1         
                    conv0 = conv2d(input=input, num_output_channels=64, kernel_size=3, name='conv0', act=act) 
                    flat = tf.reduce_mean(conv0, (1, 2))
                    concat = tf.concat([flat, tf.expand_dims(reddening, -1)], 1)

                elif net == 3:  #Net_S2         
                    conv0 = conv2d(input=input, num_output_channels=64, kernel_size=3, name='conv0', act=act) 
                    conv0p = pool2d(input=conv0, kernel_size=2, stride=2, name='conv0p', use_avg=True)
                    conv1 = conv2d(input=conv0p, num_output_channels=64, kernel_size=3, name='conv1', act=act) 
                    flat = tf.reduce_mean(conv1, (1, 2))
                    concat = tf.concat([flat, tf.expand_dims(reddening, -1)], 1)

                elif net == 4: #Net_S3      
                    conv0 = conv2d(input=input, num_output_channels=64, kernel_size=3, name='conv0', act=act) 
                    conv0p = pool2d(input=conv0, kernel_size=2, stride=2, name='conv0p', use_avg=True)
                    conv1 = conv2d(input=conv0p, num_output_channels=64, kernel_size=3, name='conv1', act=act) 
                    conv1p = pool2d(input=conv1, kernel_size=2, stride=2, name='conv1p', use_avg=True)
                    conv2 = conv2d(input=conv1p, num_output_channels=64, kernel_size=3, name='conv2', act=act) 
                    flat = tf.reduce_mean(conv2, (1, 2))
                    concat = tf.concat([flat, tf.expand_dims(reddening, -1)], 1)
                return concat

        concat = CNN('CNN')
        
        act = 'relu'
        if net <= 1:
            fc0 = fully_connected(input=concat, num_outputs=1024, name = name+'fc0_estimator', act=act)
            fc1 = fully_connected(input=fc0, num_outputs=1024, name = name+'fc1_estimator', act=act)
        else:
            fc0 = fully_connected(input=concat, num_outputs=256, name = name+'fc0_estimator', act=act)
            fc1 = fully_connected(input=fc0, num_outputs=256, name = name+'fc1_estimator', act=act)
        
        if multi_r_out:
            fc2 = []
            for i in range(num_multi_r_out):
                fc2_i = fully_connected(input=fc1, num_outputs=bins, name = name+'fc2_estimator_'+str(i), act=None)
                fc2.append(fc2_i)
            fc_r = fully_connected(input=fc1, num_outputs=num_multi_r_out, name='fc_r', act=None)
            fc2.append(fc_r)
            fc2 = tf.concat(fc2, 1)                
        else:
            fc2 = fully_connected(input=fc1, num_outputs=bins, name = name+'fc2_estimator', act=None)
        return fc2
