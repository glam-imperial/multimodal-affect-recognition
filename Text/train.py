import os
import numpy as np
import tensorflow as tf
import sys

import model
import data_utils_text as dl

np.set_printoptions(threshold=sys.maxsize)

LOGGING_PATH = './' # please specify the following path
SAVE_PATH = 'model/model.ckpt' # specify your model save path
Embedding_PATH = 'word_embedding'

ArchivePathTrain = 'check_points/loss_archive_train.txt'
ArchivePathEval = 'check_points/loss_archive_eval.txt'

flags = tf.flags

# path related
flags.DEFINE_string('load_model', None,
                    '(optional) filename of the model to load. Useful for re-starting training from a checkpoint')

# model params
flags.DEFINE_integer('word_vocab_size', 1223, 'size of vocabulary')
flags.DEFINE_integer('rnn_size', 256, 'size of LSTM cell')
flags.DEFINE_integer('highway_layers', 2, 'size of highway layers')
flags.DEFINE_integer('word_embed_size', 300, 'embed_size')
flags.DEFINE_string('kernels', '[2, 3, 4]', 'CNN kernel width')
flags.DEFINE_string('kernel_features', '[100, 100, 100]', 'CNN kernel num')  # best is aquired with [128, 128, 64]
flags.DEFINE_integer('rnn_layers', 1, 'num of layers of RNN')

flags.DEFINE_integer('head_attention_layers', 2,  'num of heads of stack attention')
flags.DEFINE_string ('trnn_size',   '[256, 256]', 'stack attention trnn size')
flags.DEFINE_string ('trnn_layers', '[2, 2]', 'stack attention trnn layers')
flags.DEFINE_float('dropout', 0.0, 'dropout')

flags.DEFINE_float('learning_rate_decay', 0.5, 'learning rate decay')
flags.DEFINE_float('learning_rate', 0.0005, 'lr')
flags.DEFINE_float('decay_when', 1.0, 'decay if validation perplexity does not improve by more than this much')
flags.DEFINE_float('param_init', 0.05, 'initialize parameters at')
flags.DEFINE_integer('num_unroll_steps', 100, 'number of timesteps to unroll for')
flags.DEFINE_integer('batch_size', 20, 'number of sequences to train on in parallel')
flags.DEFINE_integer('batch_size_eval', 5, 'number of sequences to evaluate in parallel at eval time')
flags.DEFINE_integer('max_epochs', 200, 'number of full passes through the training data')
flags.DEFINE_float('max_grad_norm', 5.0, 'normalize gradients at')
flags.DEFINE_integer('max_sent_length', 30, 'maximum word length')



FLAGS = flags.FLAGS


def train():
    dataset_tensors, labels_tensors = dl.make_batches()

    input_tensor_tr, label_tensor_tr, seq_tensor_tr = dl.sequence_init(dataset_tensors, labels_tensors,
                                                                       FLAGS.num_unroll_steps, 'Train',
                                                                       allow_short_seq=False)
    input_tensor_te, label_tensor_te, seq_tensor_te = dl.sequence_init(dataset_tensors, labels_tensors,
                                                                       FLAGS.num_unroll_steps, 'Devel',
                                                                       allow_short_seq=True)



    train_reader = dl.TrainDataReader(input_tensor_tr, label_tensor_tr, seq_tensor_tr, FLAGS.batch_size,
                                      FLAGS.num_unroll_steps, False)

    eval_reader = dl.EvalDataReader(input_tensor_te, label_tensor_te, seq_tensor_te, FLAGS.batch_size_eval,
                                    FLAGS.num_unroll_steps, False)




    labels = tf.placeholder(tf.float32, [None, FLAGS.num_unroll_steps, 3], name='labels')

    train_model = model.inference_graph(word_vocab_size=FLAGS.word_vocab_size,
                                                 kernels=eval(FLAGS.kernels),
                                                 kernel_features=eval(FLAGS.kernel_features),
                                                 rnn_size=FLAGS.rnn_size,
                                                 dropout=FLAGS.dropout,
                                                 num_rnn_layers=FLAGS.rnn_layers,
                                                 num_highway_layers=FLAGS.highway_layers,
                                                 num_unroll_steps=FLAGS.num_unroll_steps,
                                                 max_sent_length=FLAGS.max_sent_length,
                                                 # batch_size= FLAGS.batch_size,
                                                 embed_size=FLAGS.word_embed_size,
                                                 trnn_size= eval(FLAGS.trnn_size),
                                                 num_trnn_layers= eval(FLAGS.trnn_layers),
                                                 num_heads = FLAGS.head_attention_layers)

    predictions_arousal = train_model.predictions_arousal
    predictions_valence = train_model.predictions_valence
    predictions_liking = train_model.predictions_liking


    predictions_AV = tf.concat([predictions_arousal, predictions_valence], 1)
    predictions = tf.concat([predictions_arousal, predictions_valence, predictions_liking], 1)

    embedding_matrix = dl.loadPickle(Embedding_PATH, 'Embedding_300_fastText_training.pkl')

    AV_losses = model.loss_graph_ccc_arousal_valence(predictions_AV, labels)
    eval_model = model.metric_graph()


    loss_av = AV_losses.AV_CCC

    eval_arousal = eval_model.eval_metric_arousal
    eval_valence = eval_model.eval_metric_valence
    eval_liking = eval_model.eval_metric_liking


    optimize_graph = model.training_graph(loss_av, FLAGS.learning_rate, FLAGS.max_grad_norm)
    train_op = optimize_graph.train_op

    saver = tf.train.Saver(max_to_keep=100)


    with tf.Session() as sess:


        sess.run(tf.initialize_all_variables())
        train_writer = tf.summary.FileWriter('.\logs\\train', graph=sess.graph)
        eval_writer = tf.summary.FileWriter('.\logs\\eval', graph=sess.graph)


        best, inx = 0.92, 1

        epoch = 0
        global_step = 0

        while epoch < FLAGS.max_epochs:

            batch = 1
            epoch += 1
            train_reader.make_batches()
            for minibatch in train_reader.iter():

                x, y = minibatch
                _, l = sess.run(
                    [train_op, loss_av],
                    feed_dict={
                        train_model.input: x,
                        labels: y,
                        train_model.sequence_length: [96] * x.shape[0],
                        train_model.batch_size: x.shape[0],
                        train_model.training: True,
                        train_model.word_embedding: embedding_matrix,
                        train_model.dropout_LSTM: 0.0,
                        train_model.dropout_text: 0.1,
                        train_model.dropout_atdnn: 0.3,
                        train_model.dropout_trnn: 0.3,
                        train_model.dropout_mlattention: 0.2
                    })

                with open(ArchivePathTrain, 'a') as apt:
                    apt.write(str(l) + ';' + str(global_step))
                    apt.write('\n')
                print('Epoch: %5d/%5d -- batch: %5d -- loss: %.4f' % (epoch, FLAGS.max_epochs, batch, l))

                summary = tf.Summary(value=[
                    tf.Summary.Value(tag="TRAIN_LOSS", simple_value=l)
                ])

                train_writer.add_summary(summary, global_step)

                if batch % 9 == 0:  # 7, change print from 7 to 9 20180725
                    print('-------------------Devel Set Start------------------------------')
                    cnt = 0

                    prev = None
                    eval_x_total = None
                    eval_y = None
                    for mb in eval_reader.iter():

                        eval_x_list, eval_y_list, eval_z_list = mb

                        for eval_x, eval_z in zip(eval_x_list, eval_z_list):
                            cnt += np.sum(eval_z)
                            eval_tmp_preds= sess.run([predictions], feed_dict={
                                train_model.input: eval_x,
                                train_model.sequence_length: eval_z,
                                train_model.batch_size: eval_x.shape[0],
                                train_model.training: False,
                                train_model.word_embedding: embedding_matrix,
                                train_model.dropout_LSTM: 0.0,
                                train_model.dropout_text: 0.0,
                                train_model.dropout_atdnn: 0.0,
                                train_model.dropout_trnn: 0.0,
                                train_model.dropout_mlattention: 0.0
                            })


                            if prev is None:
                                prev = eval_tmp_preds[0]
                            else:
                                prev = np.vstack((prev, eval_tmp_preds[0]))
                        prev = prev[:cnt]

                        if eval_x_total is None:
                            eval_x_total = prev
                        else:
                            eval_x_total = np.vstack((eval_x_total, prev))
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

                    with open(ArchivePathEval, 'a') as ape:
                        ape.write(str(eval_loss) + ';' + str(global_step))
                        ape.write('\n')

                    summary_eval = tf.Summary(value=[
                        tf.Summary.Value(tag="Eval_LOSS", simple_value=eval_loss)
                    ])

                    eval_writer.add_summary(summary_eval, global_step)

                    if eval_loss < best:
                        saver.save(sess, SAVE_PATH + '-{}'.format(inx))
                        inx += 1

                        log = open(LOGGING_PATH, 'a')
                        log.write('Model, ' + SAVE_PATH + '-{}'.format(inx) + '\n')
                        log.write('%s, Epoch: %d, Batch: %d, Loss: %.4f, Arousal: %.4f, Valence: %.4f\n' % ('Devel',
                                                                                                            epoch,
                                                                                                            batch,
                                                                                                            eval_loss,
                                                                                                            eval_res[0],
                                                                                                            eval_res[
                                                                                                                1]))
                        log.write('======================================================\n')
                        log.close()


                    print('Devel Set, Epoch: %5d/%5d -- batch: %5d -- loss: _%.4f -- arousal: %.4f -- valence: %.4f -- liking: %.4f'
                          % (epoch, FLAGS.max_epochs, batch, eval_loss, eval_res[0], eval_res[1], eval_res[2]))

                    print('---------------------Devel Finished----------------------')

                global_step += 1
                batch += 1

if __name__ == '__main__':
    train()



