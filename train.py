# Data manipulation
import numpy as np
import matplotlib.pyplot as plt

# Feature extraction
import scipy
import librosa
import python_speech_features as mfcc
import os
from scipy.io.wavfile import read

# Model training
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
    
    # Split files
    files = [os.path.join(source,f) for f in os.listdir(source) if f.endswith('.wav')]
    len_train = int(len(files)*0.8)
    train_files = files[:len_train]
    test_files = files[len_train:]
    
    # Train features
    features_train = []
    for f in train_files:
        sr, audio = read(f)
        vector = get_MFCC(sr,audio)
        if len(features_train) == 0:
            features_train = vector
        else:
            features_train = np.vstack((features_train, vector))
            
    # Test features  
    features_test = []
    for f in test_files:
        sr, audio = read(f)
        vector = get_MFCC(sr,audio)
        if len(features_test) == 0:
            features_test = vector
        else:
            features_test = np.vstack((features_test, vector))
            
    return features_train, features_test

source = "./AudioSet/male_clips"
features_train_male, features_test_male = get_features(source)

gmm_male = GMM(n_components = 4, max_iter = 100, covariance_type = 'diag', n_init = 3)
gmm_male.fit(features_train_male)

source = "./AudioSet/female_clips"
features_train_female, features_test_female =  get_features(source)

gmm_female = GMM(n_components = 4, max_iter=100, covariance_type='diag', n_init = 3)
gmm_female.fit(features_train_female)

plt.figure(figsize=(15,10))
for i in range(1, 430000, 1000):
    plt.plot(features_train_male[i], c='b', linewidth=0.5, alpha=0.5)
    plt.plot(features_train_female[i], c='r', linewidth=0.5, alpha=0.5)
    plt.plot(features_test_male[i+1], c='b', label="Male", linewidth=0.5, alpha=0.5)
    plt.plot(features_test_female[i+1], c='r', label="Female", linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.title("MFCC features for Males and Females")
    plt.show()

output = []

for f in features_test_male:

    log_likelihood_male = np.array(gmm_male.score([f])).sum()
    log_likelihood_female = np.array(gmm_female.score([f])).sum()
    
    if log_likelihood_male > log_likelihood_female:
        output.append(0)
    else:
        output.append(1)

accuracy_male = (1 - sum(output)/len(output))
print(accuracy_male)

output = []

for f in features_test_female:
    log_likelihood_male = np.array(gmm_male.score([f])).sum()
    log_likelihood_female = np.array(gmm_female.score([f])).sum()
    
    if log_likelihood_male > log_likelihood_female:
        output.append(0)
    else:
        output.append(1)

accuracy_female = (sum(output)/len(output))
print(accuracy_female)








