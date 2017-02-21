
import math
import customDataGeter
import model
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as ly
import prettytensor as pt

from scipy.misc import imsave

class GAN(object):

    def __init__(self, version, clip_abs, hidden_size, batch_size, learning_rate, data_directory, log_directory):
        '''GAN Construction function

        Args:
            hidden_size: the hidden size for random Value
            batch_size: the img num per batch
            learning_rate: the learning rate

        Returns:
            A tensor that expresses the encoder network

        Notify: output size dependence
        '''
        self.img_size = [64, 64]
        self.version = version
        self.clip_abs = clip_abs
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.data_directory = data_directory
        self.log_directory = log_directory

        # build the graph
        self._build_graph()
        self.merged_all = tf.summary.merge_all()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.summary_writer = tf.summary.FileWriter(self.log_directory, self.sess.graph)


    def _build_graph(self):
        # build up the hidden Z
        z = tf.truncated_normal([self.batch_size, self.hidden_size], stddev=1)

        # the training step
        global_step = tf.Variable(0, trainable=False)

        # build template
        discriminator_template = model.build_discriminator_template(self.version)
        generator_template = model.build_generator_template(self.version, self.hidden_size)

        # instance the template
        self.g_out, z_prediction = pt.construct_all(generator_template, input=z)
        real_disc_inst = discriminator_template.construct(input=customDataGeter.input(self.data_directory, self.img_size, self.batch_size))
        fake_disc_inst = discriminator_template.construct(input=self.g_out)

        if self.version == 2:
            self.discriminator_loss = tf.reduce_mean(fake_disc_inst - real_disc_inst, name='discriminator_loss')
            self.generator_loss = tf.reduce_mean(-fake_disc_inst, name='generator_loss')
        if self.version == 1:
            self.discriminator_loss = tf.add(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_disc_inst, tf.ones_like(real_disc_inst))),
                                        tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_disc_inst, tf.zeros_like(fake_disc_inst))), name='discriminator_loss')
            self.generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_disc_inst, tf.ones_like(fake_disc_inst)), name='generator_loss')

        self.z_prediction_loss = tf.reduce_mean(tf.square(z - z_prediction), name='z_prediction_loss')

        # build the optimization operator (RMS no better than adam.)
        if self.version == 1:
            self.opt_d = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.discriminator_loss, global_step,
                                                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator'))
            self.opt_g = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.generator_loss, global_step,
                                                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator'))
        if self.version == 2:
            self.opt_d = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.discriminator_loss, global_step,
                                                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator'))
            self.opt_g = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.generator_loss, global_step,
                                                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator'))
                                                
        self.opt_z = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.z_prediction_loss, global_step,
                                            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator'))

        if self.version == 2:
            # define the clip op
            clipped_var_c = [tf.assign(var, tf.clip_by_value(var, -self.clip_abs, self.clip_abs)) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')]
            # merge the clip operations on critic variables
            with tf.control_dependencies([self.opt_d]):
                self.opt_d = tf.tuple(clipped_var_c)

    def update_params(self, current_step, d_step = 1, g_step = 1):
        #if self.version == 1:
        #    if current_step<=25 or current_step%500 == 0:
        #        d_step = 100
        #    else:
        #        d_step = 5
        # train citers 
        for j in range(d_step):
            self.sess.run(self.opt_d)

        for j in range(g_step):
            self.sess.run(self.opt_g)
            self.sess.run(self.opt_z)

    def get_loss(self):
        d_l, g_l = self.sess.run([self.discriminator_loss, self.generator_loss])
        return d_l, g_l

    def generate_and_save_images(self, num_samples, directory):
        '''Generates the images using the model and saves them in the directory

        Args:
            num_samples: number of samples to generate
            directory: a directory to save the images

        Notify: output size dependence
        '''
        imsize = self.img_size
        im_w = int(math.ceil(math.sqrt(num_samples)))
        big_img = np.zeros([im_w*imsize[1],im_w*imsize[0],3])
        imgs = self.sess.run(self.g_out)
        while imgs.shape[0]<num_samples:
            tmp = self.sess.run(self.g_out)
            imgs = np.concatenate((imgs, tmp), axis=0)
        for k in range(num_samples):
            slice_img = imgs[k].reshape(imsize[1], imsize[0], 3)
            big_img[(k/im_w)*imsize[1]:((k/im_w)+1)*imsize[1], (k%im_w)*imsize[0]:((k%im_w)+1)*imsize[0],:] = slice_img
            imgs_folder = os.path.join(directory, 'imgs')
            if not os.path.exists(imgs_folder):
                os.makedirs(imgs_folder) 
            imsave(os.path.join(imgs_folder, '%d.png') % k, slice_img)
        imsave(os.path.join(imgs_folder,"Agg.png"), big_img)
    
    def get_merged_image(self, num_samples):
        imsize = self.img_size
        im_w = int(math.ceil(math.sqrt(num_samples)))
        big_img = np.zeros([im_w*imsize[1],im_w*imsize[0],3])
        imgs = self.sess.run(self.g_out)
        while imgs.shape[0]<num_samples:
            tmp = self.sess.run(self.g_out)
            imgs = np.concatenate((imgs, tmp), axis=0)
        for k in range(num_samples):
            big_img[(k/im_w)*imsize[1]:((k/im_w)+1)*imsize[1], (k%im_w)*imsize[0]:((k%im_w)+1)*imsize[0],:] = imgs[k].reshape(imsize[1], imsize[0], 3)
        big_img = big_img.reshape([1,big_img.shape[0],big_img.shape[1],3])
        return big_img