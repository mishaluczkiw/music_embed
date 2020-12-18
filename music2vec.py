

import os
from zipfile import ZipFile
from slice import read_midi_file, slice_midi, plot_slices_bar
import matplotlib.pyplot as plt
import pickle
import sys
import pandas as pd
import numpy as np
import ast
import fluidsynth
import tensorflow as tf
from MIDI import midi2score
import time
import tqdm
from os.path import dirname, join, isfile 


from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Dot, Embedding, Flatten, GlobalAveragePooling1D, Reshape
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def read_midi_file(file, get_durations=False):
    midi = open(file,'rb').read()
    score = midi2score(midi)
    '''
    The argument is a list: the first item in the list is the "ticks"
    parameter, the others are the tracks. Each track is a list
    of score-events, and each event is itself a list.  A score-event
    is similar to an opus-event (see above), except that in a score:
     1) the times are expressed as an absolute number of ticks
        from the track's start time
     2) the pairs of 'note_on' and 'note_off' events in an "opus"
        are abstracted into a single 'note' event in a "score":
        ['note', start_time, duration, channel, pitch, velocity]
    score2opus() returns a list specifying the equivalent "opus".
    
    my_score = [
        96,
        [   # track 0:
            ['patch_change', 0, 1, 8],
            ['note', 5, 96, 1, 25, 96],
            ['note', 101, 96, 1, 29, 96]
        ],   # end of track 0
    ]
    my_opus = score2opus(my_score)
    '''

    # numTracks = len(score[2][:])

    ticks = score[0]

    #GOING THROUGH A SCORE WITHIN A PYTHON PROGRAM
    channels = {2,3,5,8,13}
    itrack = 1   # skip 1st element which is ticks
    all_notes = []
    all_durations = []
    all_starts = []
    while itrack < len(score):
        for event in score[itrack]:
            if event[0] == 'note':   # for example,
                all_notes.append(event[4])
                all_starts.append(event[1])
                all_durations.append(event[2])# do something to all notes
            # or, to work on events in only particular channels...

        itrack += 1

    all_quarters = np.array(all_durations)/ticks

    x_score = (np.array([np.array(all_starts)/ticks,np.array(all_starts)/ticks+all_quarters]))
    y_score = np.transpose(np.repeat(np.reshape(np.array(all_notes),(-1,1)),2,axis=1))

    if get_durations:
        durations = np.where(all_quarters >= 1 / 16, np.round(all_quarters * 16) / 16, all_quarters)
        durations = np.transpose(np.repeat(np.reshape(np.array(durations), (-1, 1)), 2, axis=1))
        return x_score, y_score, durations
    else:
        return x_score, y_score


def unzip(path, zips):
    for root, dirs, files in os.walk(path):
        for file in files:
            file_name = os.path.join(root, file)
            if file_name.endswith('.zip'):
                currentdir = os.path.join(root, file_name[:-4])
                currentdir = '\\\\?\\' + currentdir
                zips.append(currentdir)
                if not os.path.exists(currentdir):
                    os.makedirs(currentdir)
                with ZipFile('\\\\?\\' + file_name) as zipObj:
                    zipObj.extractall(currentdir)
                os.remove('\\\\?\\' + file_name)
                zips = unzip(currentdir, zips)
    return zips

def get_midi_file_paths(folder):
    midis = []
    num_mids = 0
    num_Mids = 0
    num_MIDs = 0
    num_midis = 0
    num_MIDis = 0
    noOfFiles = 0
    noOfMidis = 0
    for base, dirs, files in os.walk(folder):
        for file in files:
            noOfFiles += 1
            if file.endswith('.mid'):
                midis.append(os.path.join(base, file))
                noOfMidis += 1
                num_mids += 1
            elif file.endswith('.Mid'):
                midis.append(os.path.join(base, file))
                num_Mids += 1
            elif file.endswith('.MID'):
                midis.append(os.path.join(base, file))
                num_MIDs += 1
            elif file.endswith('.midi'):
                midis.append(os.path.join(base, file))
                num_midis += 1
            elif file.endswith('.MIDi'):
                midis.append(os.path.join(base, file))
                num_MIDis += 1
    print(noOfFiles)
    print(noOfMidis)
    print('.mid: %d\t .Mid: %d\t .MID: %d\t .midi: %d\t .MIDi: %d' % (num_mids, num_Mids, num_MIDs, num_midis, num_MIDis))
    print('total MIDI:', len(midis))
    return midis

# append slices to pickle file for all midi files in the given range
def generate_slices(start, end, file_prefix):
    i_bad_midis = [3403, 7826, 20202, 25152, 40659, 40715, 58949, 63606, 67136, 99019, 99034, 106954, 112028, 118286, 122886] # add 112051?
    for i in range(start, end):
        print('midi #%d of %d...' % (i + 1, len(midi_paths)), end='\r')
        # print('midi #%d of %d...' % (i + 1, len(midi_paths)))
        file = midi_paths[i]
        if i not in i_bad_midis:
            try:
                x_score, y_score, durations = read_midi_file('\\\\?\\' + file, get_durations=True)
                slices = slice_midi(x_score, y_score, durations)
                with open("%s_%d-%d" % (file_prefix, start, end), "ab") as f:
                    pickle.dump(slices, f)
            except:
                with open("bad_midis.txt", "a", encoding='utf-8') as f:
                    f.write('{}\n'.format(file))
        else:
            with open("bad_midis.txt", "a", encoding='utf-8') as f:
                f.write('{}\n'.format(file))

# get all slice counts for the given slice file and write counts to another file
def compile_slice_counts(start, file_in, file_out):
    all_slices = []
    iii=start
    with open(file_in, "rb") as f:
        while 1:
            print(iii,end='\r')
            try:
                all_slices.extend(pickle.load(f))
            except EOFError:
                break
            iii+=1

    all_slices = [str(s) for s in all_slices]
    counts = pd.Series(all_slices).value_counts()
    with open(file_out, "wb") as f:
        pickle.dump(counts, f)
    return counts

# combine slices from all files
# filter all slices with at least threshold (10) occurrences
def combine_all_slice_counts(slice_count_files, threshold=10):
    slices = pd.Series()
    for i, file in enumerate(slice_count_files):
        print('loading file %d of %d' % (i+1, len(slice_count_files)))
        with open(file, "rb") as f:
            counts = pickle.load(f)
            counts = counts[counts >= threshold]
            slices = slices.add(counts, fill_value=0)

    # sort all slices in descending order and remove empty sets
    slices = slices[(slices.index != 'set()') & (slices.index != '[]')]
    slices.sort_values(ascending=False, inplace=True)
    print('%d total slices with at least %d occurences' % (len(slices), threshold))
    return slices

# filter out all slices with durations > threshold
def filter_durations(slices,threshold=128):
    # max_durs = np.array([max(ast.literal_eval(s))[1] for s in slices.index])
    min_durs = np.array([min(ast.literal_eval(s))[1] for s in slices.index])
    # slices_good = slices[max_durs <= threshold]
    slices_good_min = slices[min_durs <= threshold]
    
    # return slices_good
    return slices_good_min

# filter out all slices with durations > threshold
def filter_durations_of_slice(slices,threshold=128):
    # max_durs = np.array([max(ast.literal_eval(s))[1] for s in slices.index])
    min_durs = np.array([min(s)[1] for s in slices])
    # slices_good = slices[max_durs <= threshold]
    slices_good_min = slices[min_durs <= threshold]
    
    # return slices_good
    return slices_good_min

# runs much faster than nested for loops with in_slice function
# slices contain list of (notes, duration) tuples in the order they occur in
# or optionally list of (notes, note start time, duration) tuples in order
def slice_midi(x_score, y_score, durations, include_start_times=True):
    slices = []
    for slice_start in range(int(np.ceil(np.max(x_score)))):
        # notes are not in the slice if their start time <= slice start and their end time >= slice end
        inote = ~((x_score[0] >= slice_start+1) | (x_score[1] <= slice_start))
        y = y_score[0, inote]
        x = x_score[0, inote]
        s = x_score[0,inote] -slice_start
        d = durations[0, inote]
        # d = x_score[1,inote] - x_score[0,inote] # duration
        if include_start_times:
            # include duration in tuple, note start set to slice start if note starts before the slice
            cur_slice = [(y[i], d[i], s[i]) for i in range(len(y))]
        else:
            cur_slice = [(y[i], d[i]) for i in range(len(y))]
        slices.append(cur_slice)
    return slices

def slice_midi_note(x_score, y_score):
    slices = []
    for slice_start in range(int(np.ceil(np.max(x_score)))):
        inote = ~((x_score[0] > slice_start+1) | (x_score[1] <= (slice_start)))
        slices.append(set(y_score[0,inote]))
    return slices

def plot_midi(x_score, y_score, slice_num = 0):
    # Slice by quarter
    max_quarter = np.ceil(np.max(x_score))
    t_slice = np.repeat(np.reshape(np.arange(0,max_quarter,1),(1,-1)),2,axis=0) # slice one quarter note at a time
    y_slice = np.repeat(np.reshape([np.min(y_score),np.max(y_score)],(-1,1)),np.size(t_slice,1),axis=1)

    # numSlices = np.size(t_slice,1)
    # music_vecs = [] # hold note, duration, residual flag
    # s_idx = 0 # slice index
    #for t_ in t_slice[0,:]:
    #    if t_

    plt.figure
    plt.plot(x_score,y_score,'k')
    # plt.plot(np.repeat(t_slice,2,1))
    plt.plot(t_slice,y_slice,'r')
    plt.xlim([max(0,slice_num-5),min(max_quarter,slice_num+5)])
    plt.ylabel('Note Pitch (60 = C4)')
    plt.xlabel('t in quarter notes')
    plt.show()


    
def play_slice(s,synthID):
    print("Playing slice")
    for iNote in range(len(s)):
        # notes.append(s[iNote][0])
        # dur.append(s[iNote][1])
        # start.append(s[iNote][2])
        
        note =int(s[iNote][0])
        duration_time = int(np.ceil((s[iNote][1])*1000))
        start_time = int(np.ceil((s[iNote][2])*1000))
        # print('note=',note)
        # print('dur',duration_time)
        # print('start_time',start_time)
        
        seq.note(time=start_time, absolute=False, duration=duration_time, channel=0, key=note, velocity=80, dest=synthID)
        
        time.sleep(0.25)
        
# Generate training data
def wordlist2seq(word_list,seq_len):
    # reshape word_list to be a multiple of sequence length
    add_zeros = seq_len-(len(word_list)%seq_len)
    temp_word_list = word_list.copy()    
    temp_word_list.extend(np.zeros(add_zeros,dtype = 'int64'))
    temp_out = np.reshape(temp_word_list,(-1,seq_len))
    return list(temp_out)        
        
        

class Word2Vec(Model):
  def __init__(self, vocab_size, embedding_dim,num_ns = 4):
    super(Word2Vec, self).__init__()
    self.target_embedding = Embedding(vocab_size, 
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding", )
    self.context_embedding = Embedding(vocab_size, 
                                       embedding_dim, 
                                       input_length=num_ns+1)
    self.dots = Dot(axes=(3,2))
    self.flatten = Flatten()

  def call(self, pair):
    target, context = pair
    we = self.target_embedding(target)
    ce = self.context_embedding(context)
    dots = self.dots([ce, we])
    return self.flatten(dots)

# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for vocab_size tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in dataset.
  for sequence in tqdm.tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence, 
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples 
    # with positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1, 
          num_sampled=num_ns, 
          unique=True, 
          range_max=vocab_size, 
          seed=SEED, 
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      negative_sampling_candidates = tf.expand_dims(
          negative_sampling_candidates, 1)

      context = tf.concat([context_class, negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels


def hash2slice(h,word_list,s):
    k = np.where(h == word_list)
    # slices = []
    # print(k)
    
    # print(int(k[0]))
    # for kk in range(len(k[0][:])):
        
    #     slices.append(s[int(k[0][kk])])
    return s[np.min(k)]   # return only most frequent slice
    # return slices


def unique(list1): 
  
    # intilize a null list 
    unique_list = [] 
    list2 = [] # store all elements of list, with duplicates in same format
    list3 = []
    idx = 0 # hold idx where unique occurs
    idx_array = {}
    # traverse for all elements 
    for x in list1: 
        list2.append(set(x))
        # list3.append(np.mod(hash(str(set(x))),10000))
        list3.append(np.mod(hash(str(set(x))),10000))

        
        # check if exists in unique_list or not        
        if x not in unique_list: 
            unique_list.append(x)

        idx += 1
    # print list 
    
    return list3


#%% Load all counts

# folder = 'C:\\Users\\User\\Documents\\Deep Learning\\Final\\music_embed\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]'
# folder = 'C:\\Users\\MaXentric\\Desktop\\Misha\\Deep_Learning\\final\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]'

# zips = unzip(folder, [])
# midi_paths = get_midi_file_paths(folder)

# batches = [(0, 15000), (15000, 30000), (30000, 45000), (45000, 60000),
#            (60000, 75000), (75000, 90000), (90000, 105000), (105000, len(midi_paths))]
# prefix = 'slices'
# counts_prefix = 'slice_counts'
# slice_count_files = []
# for batch in batches:
#     start, end = batch
#     generate_slices(start=start, end=end, file_prefix=prefix)
#     file_in = "%s_%d-%d" % (prefix, start, end)
#     file_out = "%s_%d-%d" % (counts_prefix, start, end)
#     slice_count_files.append(file_out)
#     slice_counts = compile_slice_counts(start, file_in, file_out)

# slices = combine_all_slice_counts(slice_count_files, threshold=10)
# # write all slice counts to a single file
# with open("all_slice_counts_gte_10", "wb") as f:
#     pickle.dump(slices, f)

# to load the binary file, use this:
with open("all_slice_counts_notes", "rb") as f:
    all_slice_counts = pickle.load(f)
    print("Loading all slice counts...")
# # plot top 1000 slices and save html widget 
# plot_slices_bar(slices[:1000], filename='slices_1000.html')
all_slices_counts_dur = pickle.load(open('all_slice_counts_gte_10','rb'))

good_slices = filter_durations(all_slices_counts_dur,1)
#%%


#%% Hash the entire vocab
s_good = []
# turn to list of arrays
for s_idx in range(len(good_slices)):
    s_good.append(ast.literal_eval(good_slices.index[s_idx]))
  
VOCABSIZE = 10000
# vocab = s_good[0:VOCABSIZE]
vocab = np.copy(s_good)


# vocab_hash = np.zeros((VOCABSIZE))
vocab_hash = np.zeros(len(s_good))


# for idx in range(VOCABSIZE):
for idx in range(len(s_good)):
    # vocab_hash[idx] = int(np.mod(hash(good_slices.index[idx]),VOCABSIZE)) # hash pandas object
    # vocab_hash[idx] = int(np.mod(hash(str(set(s_good[idx]))),VOCABSIZE)) # hash list as str
    vocab_hash[idx] = int(np.mod(hash(str(s_good[idx])),VOCABSIZE)) # hash list as str




#%% Play slice stuff
# comment out if not needed

seq = fluidsynth.Sequencer(time_scale=1000, use_system_timer=(False))

fs = fluidsynth.Synth()
# init and start the synthesizer as described above…
fs.start(driver="dsound") # might have to use another driver

# AUDIO_DRIVER_NAMES = ("alsa, coreaudio, dart, dsound, file, jack, oss, portaudio, pulseaudio, "
#                       "sdl2, sndman, waveout")

sfid = fs.sfload(r"FluidR3_GM.sf2") # use system patches

fs.program_select(0, sfid, 0, 0)
# fs.program_select(1, sfid, 0, 0) # use the same program for channel 2 for cheapness
    
    
synthID = seq.register_fluidsynth(fs)
current_time = seq.get_tick()

def seq_callback(time, event, seq, data):
    print('callback called!')

callbackID = seq.register_client("myCallback", seq_callback)

seq.timer(0, dest=callbackID)


#%% Import good midi slices


# import good midis
good_midis = pickle.load(open('good_midis','rb'))
# sfid = fs.sfload(join(dirname(__file__), "example.sf2"))
dir_file = r"C:\Users\MaXentric\Desktop\Misha\Deep_Learning\final"

# fpath3 = r'C:\Users\MaXentric\Desktop\Misha\Deep_Learning\final\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\0\009count.mid'
word_list = []
slice_list = [] # no starting time
slice_list3 =[] # with staring time
for i in range(1000): # specify how many files to append
    fpath = dir_file+ good_midis[i][55:len(good_midis)-4]
    
    if os.path.isfile(fpath): # check if file exists, if not skip
        print("loading midi file #",i)
        print("--> slices from: ", good_midis[i][55:len(good_midis)-4])
        
        
        x,y,d = read_midi_file(fpath, get_durations=True)
        
        s = slice_midi(x,y,d,include_start_times=False) # no starting time
        s3 = slice_midi(x,y,d,include_start_times=True) # with starting time, used to play
        s2 = pd.Series(s)
        
        # plot_midi(x,y)
        # rand_int = int(np.floor(np.random.uniform(0,len(s3),1)))
        # play_slice(s3[rand_int],synthID)
        # store all hashes
        for iSlice in range(len(s)):
            
            # val_hash = np.mod(hash(str(s2[iSlice])),VOCABSIZE)
            # val_hash = np.mod(hash(str(set(s[iSlice]))),VOCABSIZE)
            val_hash = np.mod(hash(str(s[iSlice])),VOCABSIZE)

            
            word_list.append(val_hash)
        
        slice_list.extend(s) # add slices without start time
        slice_list3.extend(s3) # add slices with start time
        # store all slices 
            # print(val_hash)
        # print(word_list[0:10])
print("len of word list:",len(word_list))
#%%
# filter out all slices with durations > threshold
def filter_durations_of_slice(slices,threshold=128):
    min_durs = min(s)
    
    
    # slices_good = slices[max_durs <= threshold]
    slices_good_min = slices[min_durs <= threshold]
    
    # return slices_good
    return slices_good_min


s2_good = filter_durations_of_slice(s,1)
print("len of good slices",len(s2_good))
# f1 = open('word_list10000.pckl','wb')
# f2 = open('s_10000.pckl','wb')
# f3 = open('s3_10000.pckl','wb')

# pickle.dump('word_list',f1)
# pickle.dump('s',f2)
# pickle.dump('s3',f3)

# f1.close()
# f2.close()
# f3.close()
#%% Inverse vocab

def hash2slice(h,word_list,s):
    k = np.where(h == word_list)
    # slices = []
    # print(k)
    
    # print(int(k[0]))
    # for kk in range(len(k[0][:])):
        
    #     slices.append(s[int(k[0][kk])])
    return s[np.min(k)]   # return only most frequent slice


inverse_vocab = {}
inverse_vocab3 = {}
idx_h = 0
for n_hash in word_list:
    inverse_vocab[n_hash] = hash2slice(n_hash,vocab_hash,vocab)
    inverse_vocab3[n_hash] = hash2slice(n_hash,word_list,slice_list3)
    idx_h += 1
    if idx_h%10000 == 0:
        print("Percent done: ",(idx_h*100)/len(word_list))
    

# idx = 0
# for n_hash in vocab_hash[:10]:
    
#     print(n_hash)
#     print(vocab[idx])
#     idx+=1
#     print(inverse_vocab[n_hash])

#%%  Generate training data
print("Generating Training data...")

sequence_length = 10
my_sequence = wordlist2seq(word_list,sequence_length)
#%% Convert to targets and contexts
SEED = 42
# Set the number of negative samples per positive context
num_ns = 4

tslices,cslices,lslices = generate_training_data(my_sequence,
                                                 window_size = 2, 
                                                 num_ns = 4, 
                                                 vocab_size = VOCABSIZE, 
                                                 seed= SEED)
# the distribution cslices spits out is Zipf distributed with 0 being the most common number
# needs to be mapped to the hashes that occur most often


cslices_hash = vocab_hash[np.array(cslices)] 

#% embed
BATCH_SIZE = 1024
BUFFER_SIZE = 10000
dataset = tf.data.Dataset.from_tensor_slices(((tslices, cslices_hash), lslices))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print(dataset)



dataset = dataset.cache().prefetch(buffer_size=BUFFER_SIZE)
print(dataset)



embedding_dim = 128
VOCABSIZE = len(vocab_hash)
word2vec = Word2Vec(VOCABSIZE, embedding_dim,num_ns = num_ns)
word2vec.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

# word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])

weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
#%%
import io
out_v = io.open('vectors_filt.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata_filt.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
  if  index == 0: continue # skip 0, it's padding.
  vec = weights[index] 
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(str(word) + "\n")
out_v.close()
out_m.close()
 


#%%
# #%%  create inverse vocab
# def hash2slice(h,word_list,s):
#     k = np.where(h == word_list)
#     # slices = []
#     # print(k)
    
#     # print(int(k[0]))
#     # for kk in range(len(k[0][:])):
        
#     #     slices.append(s[int(k[0][kk])])
#     return s[np.min(k)]   # return only most frequent slice


# inverse_vocab = {}
# for n_hash in vocab_hash:
#     inverse_vocab[n_hash] = hash2slice(n_hash,vocab_hash,vocab)

# idx = 0
# for n_hash in vocab_hash[:10]:
    
#     print(n_hash)
#     print(vocab[idx])
#     idx+=1
#     print(inverse_vocab[n_hash])



