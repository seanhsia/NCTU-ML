import numpy as np
import pandas as pd

'''
popularity = 20 - (sum of rank # in 1st, 2nd voting event in TW, JP)(should be improved since voting population of JP and TW is like, 15.6:1)

AvgDiff = average difficulty of song of the band it belongs to (hard or expert)

color: RGB in 0 ~ 255 HSL in 0 ~ 240 
(hue should not be used due to: 
1. both side of specturm being red 
2. when either brightness is either high or low or saturation is low, hue is pretty much meaningless)

attribute are features determined by a character's characteristic (slightly subjective, 1 ~ 10)

types : pure, cool, powerful, happy

all data are collected on 12/8(JP,version-wise)
'''

def tokenize(data, feat_name):
    feat = data.groupby(feat_name)
    feat = list(feat.size().index)
    tokenize_feat = {num:token for token, num in enumerate(feat)}
    return data[feat_name].map(tokenize_feat)

#Reduce features
data = pd.read_csv("BanGDream_Song_Data.csv", encoding='utf-8')

#Tokenize features
data['Band'] = tokenize(data, 'Band')

data.to_csv("BanGDream_song_tokenize.csv")

