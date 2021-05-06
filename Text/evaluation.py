import numpy as np
import tensorflow as tf

import model
import data_utils_text as dl



LOGGING_PATH = 'check_points/log.txt'
MODEL_PATH = 'D:\Multimodal\Text\model.ckpt' # specify your own path
Embedding_PATH = 'word_embedding'

flags = tf.flags

# path related
flags.DEFINE_string('load_model',           MODEL_PATH,           '(optional) filename of the model to load. Useful for re-starting training from a checkpoint')


# model params
flags.DEFINE_integer('word_vocab_size', 1223, 'size of vocabulary')
flags.DEFINE_integer('rnn_size', 256, 'size of LSTM cell')
flags.DEFINE_integer('highway_layers', 2, 'size of highway layers')
flags.DEFINE_integer('word_embed_size', 300, 'embed_size')
flags.DEFINE_string('kernels', '[2, 3, 4]', 'CNN kernel width')
flags.DEFINE_string('kernel_features', '[100, 100, 100]', 'CNN kernel num')
flags.DEFINE_integer('rnn_layers', 1
                     , 'num of layers of RNN')


flags.DEFINE_integer('head_attention_layers',  2,  'num of heads of stack attention')
flags.DEFINE_string ('trnn_size',   '[256, 256]', 'stack attention trnn size')
flags.DEFINE_string ('trnn_layers', '[2, 2]', 'stack attention trnn layers')
flags.DEFINE_float('dropout', 0.0, 'dropout')
# optimization

flags.DEFINE_integer('num_unroll_steps', 1800, 'number of timesteps to unroll for')
flags.DEFINE_integer('batch_size', 32, 'number of sequences to train on in parallel')
flags.DEFINE_integer('batch_size_eval', 5, 'number of sequences to evaluate in parallel at eval time')
flags.DEFINE_integer('max_sent_length', 30, 'maximum word length')

FLAGS = flags.FLAGS


def evaluation(model_path):

    g = tf.Graph()
    with g.as_default():
        assert FLAGS.load_model != None

        dataset_tensors, labels_tensors = dl.make_batches()

        input_tensor_te, label_tensor_te, seq_tensor_te = dl.sequence_init(dataset_tensors, labels_tensors,
                                                                           FLAGS.num_unroll_steps, 'Test',
                                                                           allow_short_seq=True)

        eval_reader = dl.EvalDataReader(input_tensor_te, label_tensor_te, seq_tensor_te, FLAGS.batch_size_eval,
                                        FLAGS.num_unroll_steps, False)


        test_model = model.inference_graph(word_vocab_size=FLAGS.word_vocab_size,
                                                 kernels=eval(FLAGS.kernels),
                                                 kernel_features=eval(FLAGS.kernel_features),
                                                 rnn_size=FLAGS.rnn_size,
                                                 dropout=FLAGS.dropout,
                                                 num_rnn_layers=FLAGS.rnn_layers,
                                                 num_highway_layers=FLAGS.highway_layers,
                                                 num_unroll_steps=FLAGS.num_unroll_steps,
                                                 max_sent_length=FLAGS.max_sent_length,
                                                 embed_size=FLAGS.word_embed_size,
                                                trnn_size=eval(FLAGS.trnn_size),
                                                num_trnn_layers=eval(FLAGS.trnn_layers),
                                                num_heads=FLAGS.head_attention_layers)

        embedding_matrix = dl.loadPickle(Embedding_PATH, 'Embedding_300_fastText_training.pkl')

        predictions_arousal = test_model.predictions_arousal
        predictions_valence = test_model.predictions_valence
        predictions_liking = test_model.predictions_liking

        predictions = tf.concat([predictions_arousal, predictions_valence, predictions_liking], 1)

        eval_model = model.metric_graph()

        eval_arousal = eval_model.eval_metric_arousal
        eval_valence = eval_model.eval_metric_valence
        eval_liking = eval_model.eval_metric_liking

        saver = tf.train.Saver()

        with tf.Session() as sess:

            print('load model %s ...' % model_path)
            saver.restore(sess, model_path)
            print('done!')

            cnt = 0

            prev = None
            eval_y = None
            eval_x_total = None
            for mb in eval_reader.iter():

                eval_x_list, eval_y_list, eval_z_list = mb

                for eval_x, eval_z in zip(eval_x_list, eval_z_list):
                    cnt += np.sum(eval_z)
                    eval_tmp_preds = sess.run([predictions], feed_dict={
                        test_model.input: eval_x,
                        test_model.sequence_length: eval_z,
                        test_model.batch_size: eval_x.shape[0],
                        test_model.training: False,
                        test_model.word_embedding: embedding_matrix,
                        test_model.dropout_LSTM: 0.0,
                        test_model.dropout_text: 0.0,
                        test_model.dropout_atdnn: 0.0,
                        test_model.dropout_trnn: 0.0,
                        test_model.dropout_mlattention: 0.0
                    })

                    # print(s)
                    if prev is None:
                        prev = eval_tmp_preds[0]
                    else:
                        prev = np.vstack((prev, eval_tmp_preds[0]))

                prev = prev[:cnt]


                if eval_x_total is None:
                    eval_x_total = prev
                else:
                    eval_x_total = np.vstack((eval_x_total, prev))
                    # print(prev[:,2])
                if eval_y is None:
                    eval_y = np.array(eval_y_list).reshape([-1, 3])[:cnt]
                else:
                    eval_y = np.vstack((eval_y, np.array(eval_y_list).reshape([-1, 3])[:cnt]))
                prev = None
                cnt = 0


            e_arousal, e_valence, e_liking = sess.run([eval_arousal, eval_valence, eval_liking],
                                                      feed_dict={
                                                          eval_model.eval_predictions: eval_x_total,
                                                          eval_model.eval_labels: eval_y
                                                      })
            eval_res = np.array([e_arousal, e_valence, e_liking])
            eval_loss = 2. - eval_res[0] - eval_res[1]

            print('loss: %.4f -- arousal: %.4f -- valence: %.4f -- liking: %.4f'
                  % (eval_loss, eval_res[0], eval_res[1], eval_res[2]))

            print('done evaluation------------------------------------------\n')
    return eval_loss, eval_res[0], eval_res[1]

def main(_):
    loss, arousal, valence = [], [], []
    for inx in range(1,2):
        cur_loss, cur_arousal, cur_valence = evaluation(FLAGS.load_model + '-{}'.format(inx))
        loss.append(cur_loss)
        arousal.append(cur_arousal)
        valence.append(cur_valence)

    for inx in range(1):
        print('Model %d, loss: %.4f -- arousal: %.4f -- valence: %.4f' % ((inx + 1), loss[inx], arousal[inx], valence[inx]))


if __name__ == '__main__':
    tf.app.run()