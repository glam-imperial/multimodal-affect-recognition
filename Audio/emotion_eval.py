from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import models
import math
import numpy as np
import os
import shutil
import time
import metrics

from data_provider import get_split
from pathlib import Path


slim = tf.contrib.slim
LOG_PATH = './log_audio.txt'
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 1, '''The batch size to use.''')
tf.app.flags.DEFINE_string('dataset_dir',
                           'tfrecords',
                           'The tfrecords directory.')  # specify the dataset path
tf.app.flags.DEFINE_string('checkpoint_dir',            # specify the saved model path
                           './check_points',
                           'The tfrecords directory.')
tf.app.flags.DEFINE_integer('hidden_units', 256, 'The number of examples in the test set.')
tf.app.flags.DEFINE_integer('seq_length', 100, 'The number of examples in the test set.')
tf.app.flags.DEFINE_integer('eval_interval_secs', 75, 'The seconds to wait until next evaluation.')
tf.app.flags.DEFINE_string('portion', 'Devel', '`Devel` or `Test` to evaluation on validation or test set.')
tf.app.flags.DEFINE_string('model', 'audio_model2',
                           '''Which model is going to be used: audio, video, or both ''')

slim = tf.contrib.slim


def evaluate(file2eval, model_path):
    g = tf.Graph()
    with g.as_default():

        total_nexamples = 0

        filename_queue = tf.FIFOQueue(capacity=1, dtypes=[tf.string])

        # Load dataset.
        audio_frames, labels, _ = get_split(filename_queue, False,
                                            FLAGS.portion, 1,
                                            seq_length=FLAGS.seq_length)

        # Define model graph.
        with slim.arg_scope([slim.layers.batch_norm, slim.layers.dropout],
                            is_training=False):
            predictions = models.get_model(FLAGS.model)(audio_frames,
                                                        hidden_units=FLAGS.hidden_units)

        coord = tf.train.Coordinator()
        variables_to_restore = slim.get_variables_to_restore()

        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            saver.restore(sess, model_path)
            tf.train.start_queue_runners(sess=sess, coord=coord)
            evaluated_predictions = []
            evaluated_labels = []

            print('Evaluating file : {}'.format(file2eval))
            nexamples = _get_num_examples(file2eval)
            total_nexamples += nexamples

            num_batches = int(math.ceil(nexamples / (float(FLAGS.seq_length))))

            sess.run(filename_queue.enqueue(file2eval))
            sess.run(filename_queue.enqueue(file2eval))
            for _ in range(num_batches):
                prediction_, label_ = sess.run([predictions, labels])
                evaluated_predictions.append(prediction_[0])
                evaluated_labels.append(label_[0])

            evaluated_predictions = np.vstack(evaluated_predictions)[:nexamples]
            evaluated_labels = np.vstack(evaluated_labels)[:nexamples]

            for i in range(sess.run(filename_queue.size())):
                sess.run(filename_queue.dequeue())
            if sess.run(filename_queue.size()) != 0:
                raise ValueError('Queue not empty!')
            coord.request_stop()

    return evaluated_predictions, evaluated_labels


def _get_num_examples(tf_file):
    c = 0
    for record in tf.python_io.tf_record_iterator(tf_file):
        c += 1

    return c

def copy2temporary(model_path):
    shutil.copy(model_path + '.data-00000-of-00001',
                './temporary_model/temporary.ckpt.data-00000-of-00001')
    shutil.copy(model_path + '.index', './temporary_model/temporary.ckpt.index')
    shutil.copy(model_path + '.meta', './temporary_model/temporary.ckpt.meta')

    return './temporary_model/temporary.ckpt'

# if you want to save the best model
def copy2best(model_path, inx):
    shutil.copy(model_path + '.data-00000-of-00001',
                './Best_Audio_{}.ckpt.data-00000-of-00001'.format(inx))
    shutil.copy(model_path + '.index', './Best_Audio_{}.ckpt.index'.format(inx))
    shutil.copy(model_path + '.meta', './Best_Audio_{}.ckpt.meta'.format(inx))

def deltemporary(model_path):
    os.remove(model_path + '.data-00000-of-00001')
    os.remove(model_path + '.index')
    os.remove(model_path + '.meta')

def main(_):
    dataset_dir = Path(FLAGS.dataset_dir)

    best, inx = 0.98, 1

    cnt = 0
    while True:

        model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        print('Current latest model: ' + model_path)
        model_path = copy2temporary(model_path)

        print(model_path)

        predictions, labels = None, None
        eval_model = metrics.metric_graph()
        eval_arousal = eval_model.eval_metric_arousal
        eval_valence = eval_model.eval_metric_valence
        eval_liking = eval_model.eval_metric_liking

        files = os.listdir(str(dataset_dir))
        portion_files = [str(dataset_dir / x) for x in files if FLAGS.portion in x]
        print(portion_files)
        for tf_file in portion_files:
            predictions_file, labels_file = evaluate(str(tf_file), model_path)
            print(tf_file)
            if predictions is not None and labels is not None:
                predictions = np.vstack((predictions, predictions_file))
                labels = np.vstack((labels, labels_file))
            else:
                predictions = predictions_file
                labels = labels_file

        print(predictions.shape)
        print(labels.shape)
        with tf.Session() as sess:

            e_arousal, e_valence, e_liking = sess.run([eval_arousal, eval_valence, eval_liking],
                                                      feed_dict={
                                                          eval_model.eval_predictions: predictions,
                                                          eval_model.eval_labels: labels
                                                      })
            eval_res = np.array([e_arousal, e_valence, e_liking])
            eval_loss = 2. - eval_res[0] - eval_res[1]
            print('Evaluation: %d, loss: _%.4f -- arousal: %.4f -- valence: %.4f -- liking: %.4f'
                  % (cnt, eval_loss, eval_res[0], eval_res[1], eval_res[2]))

            cnt += 1

        if eval_loss < best:
            print('================================================================================')
            copy2best(model_path, inx)

            log = open(LOG_PATH, 'a')
            log.write('Evaluated Model: ' + model_path + '\n')
            log.write('Evaluated loss: %.4f, arousal: %4.f, valence: %.4f\n' % (eval_loss, eval_res[0], eval_res[1]))
            log.write('========================================\n')
            inx += 1
            log.close()

        else:
            print(model_path)
            deltemporary(model_path)

        print('Finished evaluation! Now waiting for {} secs'.format(FLAGS.eval_interval_secs))
        time.sleep(FLAGS.eval_interval_secs)


if __name__ == '__main__':
    tf.app.run()