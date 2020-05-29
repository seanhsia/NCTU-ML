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
data = pd.read_csv("BanGDream_Data.csv", encoding='utf-8')
data = data.drop(["Name", "No"] ,axis=1)
data = data.drop(["OriginalInGame", "CoverInGame", "SongInGame","AvgDiffExpert"] ,axis=1)

#Tokenize features
data['Band'] = tokenize(data, 'Band')
data['BandType'] = tokenize(data, 'BandType')
data['Part'] = tokenize(data, 'Part')

#Range Normalize
#RGB
data['HairR'] = data['HairR'] / 255
data['HairG'] = data['HairG'] / 255
data['HairB'] = data['HairB'] / 255
data['EyeR'] = data['EyeR'] / 255
data['EyeG'] = data['EyeG'] / 255
data['EyeB'] = data['EyeB'] / 255

#HSL
data['HairH'] = data['HairH']/240
data['HairS'] = data['HairS']/240
data['HairL'] = data['HairL']/240
data['EyeH'] = data['EyeH']/240
data['EyeS'] = data['EyeS']/240
data['EyeL'] = data['EyeL']/240

data.to_csv("BanGDream_tokenize.csv")
data.drop(["HairH", "EyeH"], axis=1).to_csv("BanGDream_noHue.csv")

