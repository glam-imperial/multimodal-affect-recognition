from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim
train_set = [i for i in range(1, 35)]
devel_set = [i for i in range(1, 15)]  # dummy implementation, please feel free to del it or modify it

def get_split(dataset_dir, is_training=True, split_name='train', batch_size=32,
              seq_length = 96, debugging=False):
    """Returns a data split of the RECOLA dataset, which was saved in tfrecords format.

    Args:
        split_name: A train/test/valid split name.
    Returns:
        The raw audio examples and the corresponding arousal/valence
        labels.
    """


    if 'train' not in split_name:
        filename_queue = dataset_dir
    else:
        paths = []
        for inx in train_set:
            paths.append(dataset_dir + 'Train_' + (str(inx) if inx >= 10 else '0' + str(inx)) + '.tfrecords')
        filename_queue = tf.train.string_input_producer(paths, shuffle=is_training)

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example( # take one frame
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.string),
            'file_name': tf.FixedLenFeature([], tf.string),
            'audio_frame': tf.FixedLenFeature([], tf.string),
        }
    )


    audio_frame = tf.decode_raw(features['audio_frame'], tf.float32) # decode the audio feature of one frame

    label = tf.decode_raw(features['label'], tf.float32) # decode label of that frame
    file_name = features['file_name'] # file name
    label.set_shape([3]) # 3-D label
    audio_frame.set_shape([4410]) # raw feature of audio considering interloculor information
    file_name.set_shape([]) # string, file name, further use

    print(audio_frame)

    audio_frames, labels, file_names = tf.train.batch(
        [audio_frame, label, file_name], seq_length, num_threads=1, capacity=1000) # generate sequence, num_threads = 1,
                                                                        # guarantee the generation of sequences is correct
    labels = tf.expand_dims(labels, 0)                                  # i.e. frames of a sequence are in correct order and belong to same subject
    audio_frames = tf.expand_dims(audio_frames, 0)
    file_names = tf.expand_dims(file_names, 0)
    
    if is_training: # generate mini_batch of sequences
        audio_frames, labels, file_names = tf.train.shuffle_batch(   # shuffle_batch
            [audio_frames, labels, file_names], batch_size, 1000, 50, num_threads=1)
    else:
        audio_frames, labels, file_names = tf.train.batch(
            [audio_frames, labels, file_names], batch_size, num_threads=1, capacity=1000)


    frames = audio_frames[:, 0, :, :]
    labels = labels[:, 0, :]
    file_names = file_names[:, 0, :]
    
    masked_audio_samples =[]
    masked_labels = []
    
    for i in range(batch_size): # make sure sequences in a batch all belong to the same subject
        mask = tf.equal(file_names[i][0], file_names[i])
        
        fs = tf.boolean_mask(frames[i], mask)
        ls = tf.boolean_mask(labels[i], mask)
        
        ls = tf.cond(tf.shape(ls)[0] < seq_length,
                     lambda: tf.pad(ls, [[0,seq_length - tf.shape(ls)[0]],[0,0]], "CONSTANT"), 
                     lambda: ls)
        
        fs = tf.cond(tf.shape(fs)[0] < seq_length, 
                 lambda: tf.pad(fs, [[0,seq_length - tf.shape(fs)[0]],[0,0]], "CONSTANT"), 
                 lambda: fs)
        
        masked_audio_samples.append(fs)
        masked_labels.append(ls)


    masked_audio_samples = tf.stack(masked_audio_samples)
    masked_labels = tf.stack(masked_labels)
    
    masked_audio_samples = tf.reshape(masked_audio_samples, (batch_size, seq_length, 4410))
    masked_labels = tf.reshape(masked_labels, (batch_size, seq_length, 3))
    
    return masked_audio_samples, masked_labels, file_names
