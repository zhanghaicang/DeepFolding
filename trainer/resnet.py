import tensorflow as tf
import numpy as np
import tensorflow as tf
import util
from util import RunMode
import logging

#PADDING_FULL_LEN = 500
PADDING_FULL_LEN = 250

class Resnet:
    def __init__(self, sess, dataset, train_config, model_config):
        self.sess = sess
        self.dataset = dataset
        self.train_config = train_config
        self.model_config = model_config

        self.input_tfrecord_files = tf.placeholder(tf.string, shape=[None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.training = tf.placeholder(tf.bool)

        self.x1d_channel_dim = model_config['1d']['channel_dim']
        self.x2d_channel_dim = model_config['2d']['channel_dim']

    def cnn_with_2dfeature(self, x2d, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse) as scope:
            block_num = 8
            filters = 16
            kernel_size = [4, 4]
            act = tf.nn.relu
            #kernel_initializer = tf.truncated_normal_initializer(stddev=0.01)
            kernel_initializer = tf.glorot_normal_initializer()
            #kernel_initializer = None
            bias_initializer = tf.zeros_initializer()
            #kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
            kernel_regularizer = None
            bias_regularizer = None

            for i in np.arange(block_num):
                inputs = x2d if i == 0 else conv_
                conv_ = tf.layers.conv2d(inputs=inputs, filters=filters,
                        kernel_size=kernel_size, strides=(1,1), padding='same', activation=act,
                        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)

            logits = tf.layers.conv2d(inputs=conv_, filters=1,
                    kernel_size=kernel_size, strides=(1,1), padding='same',
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
                
            logits = tf.reshape(logits, (-1, tf.shape(logits)[1], tf.shape(logits)[2]))
            return tf.sigmoid(logits), logits
    
    def resn_with_2dfeature(self, x2d, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse) as scope:
            block_num = 8
            filters = 32
            kernel_size = [4, 4]
            act = tf.nn.relu
            #kernel_initializer = tf.truncated_normal_initializer(stddev=0.01)
            kernel_initializer = tf.glorot_normal_initializer()
            #kernel_initializer = None
            bias_initializer = tf.zeros_initializer()
            #kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
            kernel_regularizer = None
            bias_regularizer = None

            prev = tf.layers.conv2d(inputs=x2d, filters=filters,
                    kernel_size=kernel_size, strides=(1,1), padding='same',
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
            for i in np.arange(block_num):
                conv_ = act(prev)
                conv_ = tf.layers.conv2d(inputs=conv_, filters=filters,
                        kernel_size=kernel_size, strides=(1,1), padding='same',
                        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
                conv_ = act(conv_)
                conv_ = tf.layers.conv2d(inputs=conv_, filters=filters,
                        kernel_size=kernel_size, strides=(1,1), padding='same',
                        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
                prev = tf.add(conv_, prev)

            logits = tf.layers.conv2d(inputs=prev, filters=1,
                    kernel_size=kernel_size, strides=(1,1), padding='same',
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)

            logits = tf.reshape(logits, (-1, tf.shape(logits)[1], tf.shape(logits)[2]))
            return tf.sigmoid(logits), logits

    def resn(self, x1d, x2d, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse) as scope:
            act = tf.nn.relu

            filters_1d = self.model_config['1d']['filters']
            kernel_size_1d = self.model_config['1d']['kernel_size']
            block_num_1d = self.model_config['1d']['block_num']

            filters_2d = self.model_config['2d']['filters']
            kernel_size_2d = self.model_config['2d']['kernel_size']
            block_num_2d = self.model_config['2d']['block_num']

            #kernel_initializer = tf.glorot_normal_initializer()
            kernel_initializer = tf.variance_scaling_initializer()
            bias_initializer = tf.zeros_initializer()
            if self.train_config.l2_reg <= 0.0:
                kernel_regularizer = None
            else:
                kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=self.train_config.l2_reg)
            bias_regularizer = None

            prev_1d = tf.layers.conv1d(inputs=x1d, filters=filters_1d,
                    kernel_size=kernel_size_1d, strides=1, padding='same',
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=False)
            for i in np.arange(block_num_1d):
                conv_1d = act(prev_1d)
                conv_1d = tf.layers.conv1d(inputs=conv_1d, filters=filters_1d,
                        kernel_size=kernel_size_1d, strides=1, padding='same',
                        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=False)
                conv_1d = act(conv_1d)
                conv_1d = tf.layers.conv1d(inputs=conv_1d, filters=filters_1d,
                        kernel_size=kernel_size_1d, strides=1, padding='same',
                        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=False)

                prev_1d = tf.add(conv_1d, prev_1d)
            
            out_1d = tf.expand_dims(prev_1d, axis=3)
            ones = tf.ones((1, PADDING_FULL_LEN))
            #left_1d = tf.tensordot(out_1d, ones, [[3], [0]])
            left_1d = tf.einsum('abcd,de->abce', out_1d, ones)
            left_1d = tf.transpose(left_1d, perm=[0,1,3,2])
            right_1d = tf.transpose(left_1d, perm=[0,2,1,3])
            print '1d shape', left_1d.shape, right_1d.shape

            input_2d = tf.concat([x2d, left_1d, right_1d], axis=3)
            print '2d shape', input_2d.shape
            
            prev_2d = tf.layers.conv2d(inputs=input_2d, filters=filters_2d,
                    kernel_size=kernel_size_2d, strides=(1,1), padding='same',
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=False)
            for i in np.arange(block_num_2d):
                conv_2d = act(prev_2d)
                conv_2d = tf.layers.conv2d(inputs=conv_2d, filters=filters_2d,
                    kernel_size=kernel_size_2d, strides=(1,1), padding='same',
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=False)
                conv_2d = act(conv_2d)
                conv_2d = tf.layers.conv2d(inputs=conv_2d, filters=filters_2d,
                    kernel_size=kernel_size_2d, strides=(1,1), padding='same',
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=False)
                prev_2d =  tf.add(conv_2d, prev_2d)
                
            logits = tf.layers.conv2d(inputs=prev_2d, filters=1,
                    kernel_size=kernel_size_2d, strides=(1,1), padding='same',
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, use_bias=True)

            #logits = tf.reshape(logits, (-1, tf.shape(logits)[1], tf.shape(logits)[2]))
            logits = tf.squeeze(logits, 3)
            logits_tran = tf.transpose(logits, perm=[0, 2, 1])
            logits = (logits + logits_tran) / 2.0
            return tf.sigmoid(logits), logits

    def evaluate(self, mode):
        self.sess.run(self.iterator.initializer,\
                feed_dict={self.input_tfrecord_files:self.dataset.get_chunks(mode)})
        acc = []
        while True:
            try:
                pred, y, size = self.sess.run([self.pred, self.y, self.size])
                for y_, pred_, size_ in zip(y, pred, size):
                    #pred_ = (pred_ + np.transpose(pred_)) / 2.0
                    acc_ = util.TopAccuracy(pred_[:size_, :size_], y_[:size_, :size_])
                    acc.append(acc_)
            except tf.errors.OutOfRangeError:
                break
        acc = np.array(acc)
        acc = np.mean(acc, axis=0)
        acc_str = ' '.join(['%.4f ' % acc_ for acc_ in acc])
        logging.info('{:s} acc: {:s}'.format(mode, acc_str))
        return

    def build_input(self):
        with tf.device('/cpu:0'):
            def parser(record):
                keys_to_features = {
                'x1d' :tf.FixedLenFeature([], tf.string),
                'x2d' :tf.FixedLenFeature([], tf.string),
                'y'   :tf.FixedLenFeature([], tf.string),
                'size':tf.FixedLenFeature([], tf.int64)}
                parsed = tf.parse_single_example(record, keys_to_features)
                x1d = tf.decode_raw(parsed['x1d'], tf.float32)
                x2d = tf.decode_raw(parsed['x2d'] ,tf.float32)
                size = parsed['size']
                x1d = tf.reshape(x1d, tf.stack([size, -1]))
                x2d = tf.reshape(x2d, tf.stack([size, size, -1]))
                y = tf.decode_raw(parsed['y'],tf.int16)
                y = tf.cast(y, tf.float32)
                y = tf.reshape(y, tf.stack([size, size]))
                return x1d, x2d, y, size

            dataset = tf.data.TFRecordDataset(self.input_tfrecord_files)
            dataset = dataset.map(parser, num_parallel_calls=64)
            dataset = dataset.prefetch(1024)
            dataset = dataset.shuffle(buffer_size=512)
            dataset = dataset.padded_batch(self.train_config.batch_size,
                    padded_shapes=([PADDING_FULL_LEN, self.x1d_channel_dim],
                        [PADDING_FULL_LEN, PADDING_FULL_LEN, self.x2d_channel_dim],
                        [PADDING_FULL_LEN, PADDING_FULL_LEN], []),
                    padding_values=(0.0, 0.0, -1.0, np.int64(PADDING_FULL_LEN)))
            iterator = dataset.make_initializable_iterator()
            x1d, x2d, y, size = iterator.get_next()
            return  x1d, x2d, y, size, iterator

    def train(self):
        self.x1d, self.x2d, self.y, self.size, self.iterator = self.build_input()

        with tf.device('/gpu:0'):
            #self.pred, logits = self.discriminator_cnn(self.x2d)
            #self.pred, logits = self.discriminator_resn(self.x2d)
            self.pred, logits = self.resn(self.x1d, self.x2d)
            if self.train_config.down_weight >= 1.0:
                mask = tf.greater_equal(self.y, 0.0)
                labels = tf.boolean_mask(self.y, mask)
                logits = tf.boolean_mask(logits, mask)
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = logits))
            else:
                mask_pos = tf.equal(self.y, 1.0)
                label_pos = tf.boolean_mask(self.y, mask_pos)
                logit_pos = tf.boolean_mask(logits, mask_pos)
                loss_pos = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = label_pos, logits = logit_pos))

                mask_neg = tf.equal(self.y, 0.0)
                label_neg = tf.boolean_mask(self.y, mask_neg)
                logit_neg = tf.boolean_mask(logits, mask_neg)
                loss_neg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = label_neg, logits = logit_neg))
                self.loss = loss_neg * self.train_config.down_weight + loss_pos

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)#for batch normalization
            with tf.control_dependencies(update_ops):
                if self.train_config.op_alg == 'adam':
                    optim = tf.train.AdamOptimizer(self.train_config.learn_rate,
                            beta1=self.train_config.beta1).minimize(self.loss)
                elif self.train_config.op_alg == 'sgd':
                    optim = tf.train.GradientDescentOptimizer(
                            self.train_config.learn_rate).minimize(self.loss)

        tf.summary.scalar('train_loss', self.loss)
        merged_summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.train_config.summary_dir, self.sess.graph)
        tf.global_variables_initializer().run()
        steps = 0
        saver = tf.train.Saver()
        for epoch in np.arange(self.train_config.epoch):
            self.sess.run(self.iterator.initializer,\
                    feed_dict={self.input_tfrecord_files:self.dataset.get_chunks(RunMode.TRAIN)})
            train_loss = 0.0
            while True:
                try:
                    _, _loss, summary = self.sess.run([optim, self.loss, merged_summary])
                    train_loss += _loss
                    train_writer.add_summary(summary, steps)
                    steps += 1
                except tf.errors.OutOfRangeError:
                    break
            saver.save(self.sess, '{}/model'.format(self.train_config.model_dir),
                    global_step=epoch)
            logging.info('Epoch= {:d} train_loss= {:.4f}'.format(epoch, train_loss))
            self.evaluate(RunMode.VALIDATE)
            if self.train_config.test_file_prefix is not None:
                self.evaluate(RunMode.TEST)
        train_writer.close()

    def build_input_test(self):
        with tf.device('/cpu:0'):
            def parser(record):
                keys_to_features = {
                'x1d' :tf.FixedLenFeature([], tf.string),
                'x2d' :tf.FixedLenFeature([], tf.string),
                'name':tf.FixedLenFeature([], tf.string),
                'size':tf.FixedLenFeature([], tf.int64)}
                parsed = tf.parse_single_example(record, keys_to_features)
                x1d = tf.decode_raw(parsed['x1d'], tf.float32)
                x2d = tf.decode_raw(parsed['x2d'] ,tf.float32)
                size = parsed['size']
                x1d = tf.reshape(x1d, tf.stack([size, -1]))
                x2d = tf.reshape(x2d, tf.stack([size, size, -1]))
                name = parsed['name']
                return x1d, x2d, name, size

            dataset = tf.data.TFRecordDataset(self.input_tfrecord_files)
            dataset = dataset.map(parser, num_parallel_calls=64)
            dataset = dataset.prefetch(512)
            #dataset = dataset.shuffle(buffer_size=512)
            dataset = dataset.padded_batch(self.train_config.batch_size,
                    padded_shapes=([PADDING_FULL_LEN, self.x1d_channel_dim],
                        [PADDING_FULL_LEN, PADDING_FULL_LEN, self.x2d_channel_dim],
                        [], []),
                    padding_values=(0.0, 0.0, "", np.int64(PADDING_FULL_LEN)))
            iterator = dataset.make_initializable_iterator()
            x1d, x2d, name, size = iterator.get_next()
            return  x1d, x2d, name, size, iterator

    def predict(self, output_dir, model_path):
        x1d, x2d, name, size, iterator = self.build_input_test()
        preds, logits = self.resn(x1d, x2d)
        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)
        self.sess.run(iterator.initializer,
                feed_dict={self.input_tfrecord_files:self.dataset.get_chunks(RunMode.TEST)})
        while True:
            try:
                preds_, names_, sizes_, = self.sess.run([preds, name, size])
                for pred_, name_, size_ in zip(preds_, names_, sizes_):
                    pred_ = pred_[:size_, :size_]
                    #inds = np.triu_indices_from(pred_, k=1)
                    #pred_[(inds[1], inds[0])] = pred_[inds]
                    #pred_ = (pred_ + np.transpose(pred_)) / 2.0
                    output_path = '{}/{}.concat'.format(output_dir, name_)
                    np.savetxt(output_path, pred_)
            except tf.errors.OutOfRangeError:
                break
