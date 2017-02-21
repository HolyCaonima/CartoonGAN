import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as ly
import prettytensor as pt
import ops

#gen_activation_fn = tf.nn.relu
#discrim_activation_fn = leaky_rectify

def build_discriminator_template(version):
    num_filters = 64
    with tf.variable_scope('discriminator'):
        discriminator = pt.template('input')
        discriminator = discriminator.conv2d(5, num_filters)
        discriminator = discriminator.apply(ops.lrelu).max_pool(2, 2)

        discriminator = discriminator.dropout(0.5).conv2d(5, num_filters*2)
        discriminator = discriminator.batch_normalize().apply(ops.lrelu).max_pool(2, 2)

        discriminator = discriminator.dropout(0.5).conv2d(5, num_filters*4)
        discriminator = discriminator.batch_normalize().apply(ops.lrelu).max_pool(2, 2)

        discriminator = discriminator.dropout(0.5).conv2d(5, num_filters*8)
        discriminator = discriminator.batch_normalize().apply(ops.lrelu).max_pool(2, 2)

        # flatten for fc
        discriminator = discriminator.flatten()
        conv_activation = discriminator

        minibatch_disc = conv_activation.minibatch_disc(100)

        discriminator = discriminator.fully_connected(1024).apply(ops.lrelu)
        discriminator = discriminator.fully_connected(1024).apply(ops.lrelu)
        if version == 1 or version == 2:
            discriminator = discriminator.fully_connected(1024).apply(ops.lrelu)
            discriminator = discriminator.fully_connected(1024).apply(ops.lrelu)
            discriminator = discriminator.fully_connected(1024).apply(ops.lrelu)

        discriminator = discriminator.concat(1, [minibatch_disc])
        discriminator = discriminator.fully_connected(1)
        
    return discriminator

def build_generator_template(version, hidden_size):
    with tf.variable_scope('generator'):
        generator = pt.template('input')
        
        generator = generator.fully_connected(1024).apply(tf.nn.relu)
        generator = generator.fully_connected(1024).apply(tf.nn.relu)
        generator = generator.fully_connected(1024).apply(tf.nn.relu)
        if version == 1 or version == 2:
            generator = generator.fully_connected(1024).apply(tf.nn.relu)
            generator = generator.fully_connected(1024).apply(tf.nn.relu)
            generator = generator.fully_connected(1024).apply(tf.nn.relu)

        generator = generator.fully_connected(4*4*512).apply(tf.nn.relu)
        fc_out = generator

        generator = generator.reshape([-1, 4, 4, 512])
        generator = generator.upsample_conv(5, 512).batch_normalize().apply(tf.nn.relu)
        generator = generator.upsample_conv(5, 256).batch_normalize().apply(tf.nn.relu)
        generator = generator.upsample_conv(5, 128).batch_normalize().apply(tf.nn.relu)
        generator = generator.upsample_conv(5, 64).batch_normalize().apply(tf.nn.relu)

        generator = generator.conv2d(5, 3).apply(tf.nn.tanh)

        z_prediction = fc_out.fully_connected(1024).apply(tf.nn.relu)
        z_prediction = z_prediction.fully_connected(1024).apply(tf.nn.relu)
        z_prediction = z_prediction.fully_connected(1024).apply(tf.nn.relu)
        if version == 1 or version == 2:
            z_prediction = z_prediction.fully_connected(1024).apply(tf.nn.relu)
            z_prediction = z_prediction.fully_connected(1024).apply(tf.nn.relu)

        z_prediction = z_prediction.fully_connected(hidden_size)

    return generator, z_prediction
