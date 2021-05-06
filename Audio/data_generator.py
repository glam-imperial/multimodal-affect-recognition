import tensorflow as tf
import numpy as np
import librosa as lb

from pathlib import Path

train_set = [i for i in range(1, 35)]
test_set = [i for i in range(1, 17)]
dev_set = [i for i in range(1, 15)]
inx_dic = {'Train_set' : train_set, 'Test_set' : test_set, 'Devel_set' : dev_set}  


# specify the data path
root_dir = Path('/path/to/AVEC2017_SEWA/') 

# Path to save AVEC data
save_path = Path('tfrecords') 

target_sr = 22050
chunk_size = 2205

def get_samples(file_name):

    label_path = str(root_dir / 'labels' / '{}.csv'.format(file_name))
    turn_path = str(root_dir / 'turns' / '{}.csv'.format(file_name))
    labels = np.loadtxt(str(label_path), delimiter=';', dtype=str)
    turns = np.loadtxt(str(turn_path), delimiter = ';', dtype= np.float32)
    
    time = labels[:,1].astype(np.float32)
    arousal = np.reshape(labels[:,2], (-1,1)).astype(np.float32)
    valence = np.reshape(labels[:,3], (-1,1)).astype(np.float32)
    liking = np.reshape(labels[:,4], (-1,1)).astype(np.float32)
    
    audio_signal, sr = lb.core.load(str(root_dir / 'audio' / '{}.wav'.format(file_name)), sr=target_sr)
    audio_signal = np.pad(audio_signal, (0, chunk_size - audio_signal.shape[0] % chunk_size), 'constant')
    
    # consider interlocutor information
    target_interculator_audio = [np.zeros((1, 4410), dtype=np.float32) for _ in range(len(time))] 
    target_set = set()
    
    audio_frames = []
    for i, t in enumerate(time): # gather the original raw audio feature
        s = int(t * sr)
        e = s + 2205
        audio = np.reshape(audio_signal[s:e], (1, -1))
        
        audio_frames.append(audio.astype(np.float32))
    
    for turn in turns:
        st, end = int(round(float(turn[0]), 1) * 10), int(round(float(turn[1]), 1) * 10)
        for i in range(st, end + 1 if end + 1 < len(time) else len(time)):
            target_set.add(i)
            target_interculator_audio[i][0][:2205] = audio_frames[i] # the subject is speaking

    for i in range(len(time)):
        if i not in target_set:
            target_interculator_audio[i][0][2205:] = audio_frames[i] # the chatting partner is speaking

    return target_interculator_audio, np.hstack([arousal, valence, liking]).astype(np.float32)

def _int_feauture(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feauture(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_sample(writer, file_name):

    for i, (audio_frame, label) in enumerate(zip(*get_samples(file_name))): # serialize every frame
        example = tf.train.Example(features=tf.train.Features(feature={
                    'file_name': _bytes_feauture(file_name.encode()),
                    'label': _bytes_feauture(label.tobytes()),
                    'audio_frame': _bytes_feauture(audio_frame.tobytes()),
                }))

        writer.write(example.SerializeToString()) # write all frames of a subject to a file

def main(directory):
    
    directory.mkdir(exist_ok=True)
    for fname in ['Train', 'Devel', 'Test']:
        print('processing ' + fname + ' file')
        for inx in inx_dic[fname + '_set']:
            file_name = fname + '_' + (str(inx) if inx >= 10 else '0' + str(inx))
            print('Writing tfrecords for {} file'.format(file_name))
            writer = tf.python_io.TFRecordWriter(
                str(directory / (file_name + '.tfrecords')))
            serialize_sample(writer, file_name)
    
if __name__ == "__main__":
  main(save_path) 