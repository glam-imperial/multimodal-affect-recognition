from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import losses
import models
from data_provider import get_split as data_provider
from tensorflow.python.platform import tf_logging as logging

slim = tf.contrib.slim

# Create FLAGS
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '', 'pre-trained-model')
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0005, 'Initial learning rate.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0, 'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_integer('batch_size', 20, '''The batch size to use.''')
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4, 'How many preprocess threads to use.')
tf.app.flags.DEFINE_string('train_dir',
                            './check_points/',
                           '''Directory where to write event logs '''  # model save path
                           '''and checkpoint.''')
tf.app.flags.DEFINE_integer('max_steps', 1, 'Number of batches to run.')
tf.app.flags.DEFINE_integer('hidden_units', 256, 'Number of batches to run.')
tf.app.flags.DEFINE_integer('seq_length', 100, 'Number of batches to run.')
tf.app.flags.DEFINE_string('train_device', '/gpu:1', 'Device to train with.')
tf.app.flags.DEFINE_string('model', 'audio_model2',
                           '''Which model is going to be used: audio, video, or both ''')

tf.app.flags.DEFINE_string('dataset_dir',   'tfrecords/',    'The tfrecords directory') # tfrecord directory


def train(data_folder):

    tf.set_random_seed(1)
    g = tf.Graph()
    with g.as_default():
        
        # Load dataset.
        audio_frames, ground_truth, _ = data_provider(data_folder, True,
                                                     'train', FLAGS.batch_size,
                                                      seq_length=FLAGS.seq_length)
        
        # Define model graph.
        with slim.arg_scope([slim.layers.batch_norm, slim.layers.dropout],
                is_training=True):
                  prediction = models.get_model(FLAGS.model)(audio_frames,
                                                            hidden_units=FLAGS.hidden_units)
        
        for i, name in enumerate(['arousal', 'valence']): #, 'liking']):
            pred_single = tf.reshape(prediction[:, :, i], (-1,))
            gt_single = tf.reshape(ground_truth[:, :, i], (-1,))
            
            loss = losses.concordance_cc(pred_single, gt_single)
            tf.summary.scalar('losses/{} loss'.format(name), loss)
            
            mse = tf.reduce_mean(tf.square(pred_single - gt_single))
            tf.summary.scalar('losses/mse {} loss'.format(name), mse)

            tf.losses.add_loss(loss / 2.)

        #print(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        total_loss = tf.losses.get_total_loss()
        tf.summary.scalar('losses/total loss', total_loss)
        
        optimizer = tf.train.AdamOptimizer(FLAGS.initial_learning_rate, beta1= 0.9, beta2= 0.99)

        with tf.Session(graph=g) as sess:

            train_op = slim.learning.create_train_op(total_loss,
                                                     optimizer,
                                                     summarize_gradients=True)
            
            logging.set_verbosity(1)
            slim.learning.train(train_op,
                                FLAGS.train_dir,
                                save_summaries_secs=60,
                                save_interval_secs=120)


if __name__ == '__main__':
    train(FLAGS.dataset_dir)
