import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#the following is a RandomForest training program for a model

training_data = pd.read_csv("train.csv")
# Any results you write to the current directory are saved as output.
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

# Our y will be the survived column so we'll add it then drop it from our X set
y = training_data['label']
X = training_data.drop(['label'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#display(X_train)
#display(X_test)
#display(y_test)
#display(y_train)

# First we'll tryout the basic Random Forest algorithm
#from sklearn.ensemble import RandomForestClassifier
#
rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=40, n_jobs=-1,
            oob_score=True, random_state=42, verbose=0, warm_start=False)

rfc.fit(X_train, y_train)

preds = rfc.predict(X_test)

#display(preds)
#The accuracy is usually quite high for RandomForest -- 90+%
print(accuracy_score(y_test, preds))

#the following draws the first 20 entries of the test file; the result will
#  be compared to the visually confirmed answers later

test_data = pd.read_csv("test.csv")
preds2 = rfc.predict(test_data.head(20))
display(preds2)



#the following code draws from the test.csv file to extract the first 20 
#   entries, each entry representing a written number 

#
from PIL import Image, ImageDraw
#
image1 = Image.new('RGB', (28, 28), (255, 255, 255)) # a white canvas
test_data = pd.read_csv('test.csv')
test_data.drop(['label'], axis = 1)
print(test_data.shape) # prints the dimension of the data just to be sure
display(test_data.loc[8:8])  # loc[8:8] extracts the 8th row of the table 

# This is drawing the number on the white canvas by the data
count = 1
for count in range(20):
    image1 = Image.new('RGB', (28, 28), (255, 255, 255))
    for x in range(28):
        for y in range(28):
            if test_data.iat[count, x * 28 + y] > 0:
                image1.putpixel((x,y),(0,0,0))
                draw = ImageDraw.Draw(image1)
                image1.save('image{0}.png'.format(count))

# Note that RGB value == 0 means it's black; RGB==255 means it's white

# The following uses another algorithm called K-Nearest Neighbour

from sklearn.neighbors import KNeighborsClassifier

df0 = pd.read_csv('train.csv')
df = df0[0:15000]
df.head()

from sklearn.model_selection import train_test_split

X = np.array(df.iloc[:, 2:785])
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# split into train and test
for i in range(5):

    # instantiate learning model (k = 2)
    knn = KNeighborsClassifier(n_neighbors=2)
    
    # fitting the model
    knn.fit(X_train, y_train)
    
    # predict the response
    
    pred = knn.predict(X_test)
    
    # evaluate accuracy
    print('\nThe percentage of correct estimate (for the training data)is %' + 
          str(accuracy_score(y_test, pred) * 100) + '')
    
    df1 = df0[15001:40000]
    df2 = df1.sample(n=10000)
    
    new_X = np.array(df2.iloc[:, 2:785])
    new_y = np.array(df2['label'])
    
    new_pred = knn.predict(new_X)
    print('\nThe new percentage of correct estimate (for the testing data)is %'
          + str(accuracy_score(new_y, new_pred) * 100) + '')
