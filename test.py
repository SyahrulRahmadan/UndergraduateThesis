# Data manipulation
import numpy as np
import matplotlib.pyplot as plt

# Ekstraksi Fitur (MFCC)
import scipy
# import librosa
import python_speech_features as mfcc
import os
from scipy.io.wavfile import read

# Model training (GMM)
from sklearn.mixture import GaussianMixture as GMM
from sklearn import preprocessing
import pickle

# Live recording
import sounddevice as sd
import soundfile as sf


def get_MFCC(sr,audio):
    
    features = mfcc.mfcc(audio, sr, 0.025, 0.01, 13, appendEnergy = False)
    features = preprocessing.scale(features)
    
    return features

def get_features(source):
    
    files = [os.path.join(source,f) for f in os.listdir(source) if f.endswith('.wav')]
    
    features = []
    for f in files:
        sr,audio = read(f)
        vector   = get_MFCC(sr,audio)
        if len(features) == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

    return features

source_male = "./AudioSet/male_clips"
features_male = get_features(source_male)

gmm_male = GMM(n_components = 4, max_iter=100, covariance_type='diag', n_init = 3)
gmm_male.fit(features_male)

source_female = "./AudioSet/female_clips"
features_female = get_features(source_female)

gmm_female = GMM(n_components = 4, max_iter=100, covariance_type='diag', n_init = 3)
gmm_female.fit(features_female)

# Save models
pickle.dump(gmm_male, open("male.gmm", "wb" ))
pickle.dump(gmm_female, open("female.gmm", "wb" ))

def record_and_predict(sr=16000, channels=1, duration=3, filename='test.wav'):

    recording = sd.rec(int(duration * sr), samplerate=sr, channels=channels).reshape(-1)
    sd.wait()
    
    features = get_MFCC(sr,recording)
    scores = None

    log_likelihood_male = np.array(gmm_male.score(features)).sum()
    log_likelihood_female = np.array(gmm_female.score(features)).sum()

    if log_likelihood_male >= log_likelihood_female:
        return("Pria")
    else:
        return("Perempuan")

gmm_male = pickle.load(open('male.gmm','rb'))
gmm_female = pickle.load(open('female.gmm','rb'))

find = record_and_predict()
print(find)
