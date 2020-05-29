import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
data = pd.read_csv("googleplaystore.csv", encoding='utf-8')
feature_names = list(data)

#Preprocessing the noise
data = data.drop(['App','Reviews','Genres', 'Last Updated', 'Current Ver', 'Android Ver'], axis=1)
data['Installs'] = data['Installs'].str.rstrip('+')
data['Installs'] = data['Installs'].str.replace(",", "")
data['Installs'] = data['Installs'].convert_objects(convert_numeric=True)
data['Price'] = data['Price'].str.lstrip('$')
data['Price'] = data['Price'].convert_objects(convert_numeric=True)
data['Size'] = data['Size'].str.replace("K", "") 
data['Size'] = data['Size'].str.replace("M", "000")
data['Size'] = data['Size'].str.replace("G", "000000")
data['Size'] = data['Size'].convert_objects(convert_numeric=True)
data['Size'] = data['Size'].divide(1000)

reduce_feat_names = list(data)
#data.to_csv("numeric_googleplaystore.csv")

#tokenizing data
#Installs

#{0: 0, 1: 1, 5: 2, 10: 3, 50: 4, 100: 5, 500: 6, 1000: 7, 5000: 8, 10000: 9, 50000: 10, 100000: 11, 500000: 12, 1000000: 13, 5000000: 14, 10000000: 15, 50000000: 16, 100000000: 17, 500000000: 18, 1000000000: 19}
#Installs = data.groupby('Installs')
#Installs_num = list(Installs.size().index)
#Installs_num.sort()
#tokenize_installs = {num:token for token, num in enumerate(Installs_num)}

#Reduce target space to {0~100 is low(0), 100~10000 is medium(1), 10000~1000000 is high(2), 1000000~ is fucking high(3)}
tokenize_installs = {0: 0, 1:0 , 5: 0, 10: 0, 50: 0, 100: 1, 500: 1, 1000: 1, 5000: 1, 10000: 2, 50000: 2, 100000: 2, 500000: 2, 1000000: 3, 5000000: 3, 10000000: 3, 50000000: 3, 100000000: 3, 500000000: 3, 1000000000: 3}
data['Installs'] = data['Installs'].map(tokenize_installs)

#Category
#{'ART_AND_DESIGN': 0, 'AUTO_AND_VEHICLES': 1, 'BEAUTY': 2, 'BOOKS_AND_REFERENCE': 3, 'BUSINESS': 4, 'COMICS': 5, 'COMMUNICATION': 6, 'DATING': 7, 'EDUCATION': 8, 'ENTERTAINMENT': 9, 'EVENTS': 10, 'FAMILY': 11, 'FINANCE': 12, 'FOOD_AND_DRINK': 13, 'GAME': 14, 'HEALTH_AND_FITNESS': 15, 'HOUSE_AND_HOME': 16, 'LIBRARIES_AND_DEMO': 17, 'LIFESTYLE': 18, 'MAPS_AND_NAVIGATION': 19, 'MEDICAL': 20, 'NEWS_AND_MAGAZINES': 21, 'PARENTING': 22, 'PERSONALIZATION': 23, 'PHOTOGRAPHY': 24, 'PRODUCTIVITY': 25, 'SHOPPING': 26, 'SOCIAL': 27, 'SPORTS': 28, 'TOOLS': 29, 'TRAVEL_AND_LOCAL': 30, 'VIDEO_PLAYERS': 31, 'WEATHER': 32}
Category = data.groupby('Category')
Category = list(Category.size().index)
tokenize_category = {num:token for token, num in enumerate(Category)}
data['Category'] = data['Category'].map(tokenize_category)

#Content Rating
#{'Adults only 18+': 0, 'Everyone': 1, 'Everyone 10+': 2, 'Mature 17+': 3, 'Teen': 4, 'Unrated': 5} 
cont_rating = data.groupby('Content Rating')
cont_rating = list(cont_rating.size().index)
tokenize_cont_rating = {num:token for token, num in enumerate(cont_rating)}
data['Content Rating'] = data['Content Rating'].map(tokenize_cont_rating)

#Type
#{'Free' : 0, 'Paid' : 1}
Type = data.groupby('Type')
Type = list(Type.size().index)
tokenize_type = {num:token for token, num in enumerate(Type)}
data['Type'] = data['Type'].map(tokenize_type)

imp = Imputer(missing_values='NaN', strategy="most_frequent", axis=0)
#data_array = imp.fit_transform(data)
#filled_data = pd.DataFrame(data_array, index=data.index, columns=data.columns)
#filled_data.to_csv("tokenize_googleplaystore_bin.csv")

Installs = data.groupby('Installs')
Installs_num = list(Installs.size().index)
sep_data = list()
filled_data = list()

for i in range(len(Installs_num)):
    sep_data.append( Installs.get_group(Installs_num[i]))
    sep_data_array = imp.fit_transform(sep_data[i])
    filled_data.append(pd.DataFrame(sep_data_array, index=sep_data[i].index, columns=sep_data[i].columns))

all_data = pd.concat(filled_data, axis=0)
all_data.to_csv("tokenize_googleplaystore_filled.csv")
