#source: http://www.analyticbridge.com/profiles/blogs/random-forest-in-python
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("data/uitm-matrik.csv")    # make sure you're in the right directory if using iPython!
test = pd.read_csv("data/uitm-matrik-test.csv")


#cols = ['PROGRAM','CGPA_MASUK', 'TANGGUNGAN', 'GAJI', 'JANTINA','BANGSA','POSKOD','CACAT']
cols = ['CGPA_MASUK', 'TANGGUNGAN', 'GAJI']

colsRes = ['STATUS_GRAD']
#print(train)
trainArr = train.as_matrix(cols) #training array
#print(trainArr)
trainRes = train.as_matrix(colsRes) # training results
#print(trainRes)
## Training!

rf = RandomForestClassifier(n_estimators=10) # initialize

rf.fit(trainArr, trainRes) # fit the data to the algorithm

# note - you might get an warning saying you entered a 2 column

# vector..ignore it. If you know how to get around this warning,

# please comment! The algorithm seems to work anyway.

## Testing!

# put the test data in the same format!

testArr = test.as_matrix(cols)

results = rf.predict(testArr)

# something I like to do is to add it back to the data frame, so I can compare side-by-side

test['predictions'] = results

print("\nmean: %s \n\n" % (rf.score(trainArr, trainRes)))
print(test.head())
