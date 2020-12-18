import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from slice import read_midi_file, slice_midi, plot_midi, play_slice, plot_midi
import fluidsynth


def play_slice(s,synthID):
    print("Playing slice")
    for iNote in range(len(s)):
        # notes.append(s[iNote][0])
        # dur.append(s[iNote][1])
        # start.append(s[iNote][2])
        
        note =int(s[iNote][0])
        duration_time = int(np.ceil((s[iNote][2])*1000))
        start_time = int(np.ceil((s[iNote][1])*1000))

        # print('dur:',duration_time)
        # print('note:',note)
        # print('start:',start_time)
        print('note=',note)
        print('dur',duration_time)
        print('start_time',start_time)
        
        seq.note(time=start_time, absolute=False, duration=duration_time, channel=0, key=note, velocity=80, dest=synthID)

def play_slice_chord(s,synthID):
    print("Playing slice")
    for iNote in range(len(s)):
        # notes.append(s[iNote][0])
        # dur.append(s[iNote][1])
        # start.append(s[iNote][2])
        
        note =int(s[iNote][0])
        duration_time = int(np.ceil((s[iNote][1])*1000))
        start_time = 0
        

        # print('note=',note)
        # print('dur',duration_time)
        # print('start_time',start_time)
        
        seq.note(time=start_time, absolute=False, duration=duration_time, channel=0, key=note, velocity=80, dest=synthID)

def play_slice_plural(s,synthID):
    print("Playing slice")
    for iSlice in range(len(s)):
        for iNote in range(len(s[iSlice])):
            # notes.append(s[iNote][0])
            # dur.append(s[iNote][1])
            # start.append(s[iNote][2])
            
            note =int(s[iSlice][iNote][0])
            duration_time = int(np.ceil((s[iSlice][iNote][1])*1000))
            start_time = iSlice*1000
            
    
            # print('note=',note)
            # print('dur',duration_time)
            # print('start_time',start_time)
            
            seq.note(time=start_time, absolute=False, duration=duration_time, channel=0, key=note, velocity=80, dest=synthID)

def play_slice_plural_start(s,synthID):
    print("Playing slice")
    for iSlice in range(len(s)):
        for iNote in range(len(s[iSlice])):
            # notes.append(s[iNote][0])
            # dur.append(s[iNote][1])
            # start.append(s[iNote][2])
            
            note =int(s[iSlice][iNote][0])
            duration_time = int(np.ceil((s[iSlice][iNote][1])*1000))
            start_time = int(np.floor((s[iSlice][iNote][1])*1000))
            
    
            # print('dur:',duration_time)
            # print('note:',note)
            # print('start:',start_time)
            # print('note=',note)
            # print('dur',duration_time)
            # print('start_time',start_time)
            
            seq.note(time=start_time, absolute=False, duration=duration_time, channel=0, key=note, velocity=80, dest=synthID)



#%%
weights = pickle.load(open('weights_filtered.pckl','rb'))
vocab_filtered = pickle.load(open('vocab_filtered.pckl', 'rb')).tolist()
vocab_hash = pickle.load(open('vocab_hash_filtered.pckl', 'rb')).astype(int).tolist()
VOCABSIZE =10000

# read in midi dataset and choose one song
# VOCABSIZE = 1000
# ii = 100
# with open('midi_dataset_1000', 'rb') as f:
#     data = pickle.load(f)['test'] 
#
# song = data[ii]
# song = list(filter(lambda slice: slice != [], song))

good_midis = pickle.load(open('good_midis','rb'))
file = good_midis[0]

# dir_file = r"C:\Users\MaXentric\Desktop\Misha\Deep_Learning\final"
# fpath = dir_file+ good_midis[0][55:len(good_midis)-4]
fpath = r"C:\Users\MaXentric\Desktop\Misha\Deep_Learning\final\beeth\pathetique_1.mid"

x_score, y_score, durations = read_midi_file(fpath, get_durations=True)
song = slice_midi(x_score, y_score, durations,include_start_times=False)
song3 = slice_midi(x_score,y_score,durations,include_start_times=True)

#%% Hash the song
song_hashes = []
for n_hash in range(len(song3)):
    song_hashes.append(np.mod(hash(str(song[n_hash])),VOCABSIZE))
    

#%%
start = 30
croplen = 32
freq = 4
subset_og = song[start:start+croplen+1]
subset = subset_og.copy()

song_replace = song.copy()

# replace slices with embeddings
# for i in range(len(subset)):
    # if subset[i] in vocab_filtered:
        # subset[i] = weights[vocab_hash[vocab_filtered.index(subset[i])]]
    # else:
        # subset[i] = weights[vocab_hash[0]]
        # subset[i] = weights[0]
        # subset[i] = weights[vocab_hash[np.mod(hash(str(subset[i])), VOCABSIZE)]]

# replace every (freq)th slice with the most similar slice using cosine similarity with its embedding
for i in range(100):
    if i % freq == 0:
        # i_slice = vocab_filtered.index(subset[i])
        i_hash = np.mod(hash(str(song[i])),VOCABSIZE)
        
        # print(i_hash)
        k = np.where(i_hash == vocab_hash)
        # print(k[0])
        # print(song[i])
        # for ik in k[0]:
        #     print(ik)
        #     print(vocab[ik])
            
        # input('1')
        i_slice = k[0][0]
        
        
        emb = weights[i_slice]
        sim = cosine_similarity(emb.reshape(1, -1), weights).flatten()
        # print(sim)
        
        i_closest = int(np.where(sim == np.amax(sim[sim != np.amax(sim)]))[0])
        # print("Replace| ", song[i], "with|", vocab_filtered[i_closest])
        
        # print()
        # subset[i] = vocab_filtered[i_closest]
        song_replace[i] = vocab_filtered[i_closest]
        
        # input('2')
        # input()
        # print(np.amax(sim))
        # print(np.amax(sim[sim != np.amax(sim)]))
        # subset[i] = weights[i_closest]
        # print(i_closest)

# save original and new subsets
# with open('original_subset', 'rb') as f:
#     pickle.dump(subset_og, f)
# with open('new_subset', 'rb') as f:
#     pickle.dump(subset, f)

#%%
seq = fluidsynth.Sequencer(time_scale=1000, use_system_timer=(False))

fs = fluidsynth.Synth()
# init and start the synthesizer as described above…
fs.start(driver="dsound") # might have to use another driver
# fs.start(driver="file")

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



        
    # time.sleep(0.25)
#%%
# play_slice(song3[7],synthID)

# play_slice_chord(song[5],synthID)


# DEMO BEETHOVENS ORIGINAL
# play_slice_plural_start(song3[0:20],synthID)

# DEMO SONG WITH REPLACED SLICES
# play_slice_plural(song_replace[0:20],synthID)



# DEMO RANDOMLY GENERATED SONG USING COCONET
# play_slice_plural(slice_rand[0:20],synthID) #lol


