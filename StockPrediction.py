import quandl,math,datetime
import numpy as np
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Get The Data From Quandl
df=quandl.get("WIKI/GOOGL",api_key="yTNaunPs6HxDyXYzZzsA")
df=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
#Create HL_PCT and PCT_CHANGE columns 
df['HL_PCT']=(df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100
df['PCT_change']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100

#remove unnecessary features
df=df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume',]]
#prediction column
forecast_col="Adj. Close"
#replace the NaNs with Outliers
df.fillna(-99999, inplace=True)

#Decide how much ahead to predict . Adj.Close for <forecast_out> days ahead
# will be a label or prediction (y) for <forecast_out> days ahead
forecast_out=int(math.ceil(0.05*len(df)))
df['label']=df[forecast_col].shift(-forecast_out)

#fix X s and ys
X=np.array(df.drop(['label'],1))
y=np.array(df['label'])

#rescale the values of X for processing time inprovement
X=preprocessing.scale(X)

#slice out the last <forecast_out> days for later prediction
x_to_predict=X[-forecast_out:]

#Reomove the above slice from actual train and test data
y=y[:-forecast_out]
X=X[:-forecast_out]

#prepare 0.2 fraction train data among all data where 0.8 will be train data
X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)
#setup a classifier
clf=LinearRegression(n_jobs=-1)
clf.fit(X_train,y_train)

#calculate the coefficient of determination of squared error
accuracy=clf.score(X_test,y_test)

#predict the values 
future_prediction=clf.predict(x_to_predict)

# forecast for all dates will be NaN by default ,but last will be updated later
df['forecast']=np.nan

#get date
last_date=df.iloc[-1].name
last_unix=last_date.timestamp()
one_day=86400
next_unix=last_unix+one_day
#create date->index for last rows
for i in future_prediction:
    next_date=datetime.datetime.fromtimestamp(next_unix)
    next_unix+=one_day
    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i]

#plotting
df['Adj. Close'].plot()
df['forecast'].plot()
print("The accuracy after crossvalidation is :",accuracy)
print("For {} days ahead, the predictions are :".format(forecast_out))
print(df[['forecast']][-forecast_out:])
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()
