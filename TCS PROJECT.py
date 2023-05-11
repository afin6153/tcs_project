#!/usr/bin/env python
# coding: utf-8

# ### Dataset information
# 
#  - battery_power - Total energy a battery can store in one time measured in mAh
#  - blue - Has bluetooth (1) or not (0)
#  - clock_speed - speed at which microprocessor executes instructions
#  - dual_sim - Has dual sim support (1) or not (0)
#  - fc - Front Camera mega pixels
#  - four_g - Has 4G (1) or not (0)
#  - int_memory - Internal Memory in Gigabytes
#  - m_dep - Mobile Depth in cm
#  - mobile_wt - Weight of mobile phone
#  - n_cores - Number of cores of processor
#  - pc - Primary Camera mega pixels
#  - px_height - Pixel Resolution Height
#  - px_width - Pixel Resolution Width
#  - ram - Random Access Memory in Mega Bytes
#  - sc_h - Screen Height of mobile in cm
#  - sc_w - Screen Width of mobile in cm
#  - talk_time - longest time that a single battery charge will last
#  - three_g - Has 3G (1) or not (0)
#  - touch_screen - Has touch screen (1) or not (0)
#  - wifi - Has wifi (1) or not (0)
#  - price_range - This is the target variable with value of 0(low cost), 1(medium cost), 2(high cost) and 3(very high cost).

# In[2]:


# Importing the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Reading the dataset
train = pd.read_csv('MobileTrain .csv')
test = pd.read_csv('MobileTest .csv')


# ## Descriptive Statistics

# In[4]:


#Displaying first 5 rows of the train dataset
train.head()


# In[5]:


#Displaying last 5 rows of the train dataset
train.tail()


# In[6]:


train.shape


# ## Basic details of the train dataset

# In[7]:


train.columns


# In[8]:


train.dtypes


# In[9]:


# basic information
train.info() 


# In[10]:


# Number of unique values in each feature
train.nunique()


# In[11]:


#Checking for null values
train.isnull().sum()


# In[12]:


#descriptive statistics
round(train.describe(),1).T


# In[13]:


def understand_data(train) :
    
    return(pd.DataFrame({"Datatype":train.dtypes,
                         "No of null values":train.isna().sum(),
                         "No of unique values":train.nunique(axis=0,dropna=True),
                         "Unique values": train.apply(lambda x: str(x.unique()),axis=0)}))
understand_data(train)


# In[14]:


#Checking for duplicate rows
train.duplicated().sum()


# # EDA Proccess

# In[15]:


# Count plot for Price range
sns.countplot(x='price_range',data=train, palette='CMRmap')
plt.title("Price Range")
plt.xlabel('0(low cost), 1(medium cost), 2(high cost) and 3(very high cost)')
plt.show() 


#  -  The dataset is balanced

# In[16]:


#Countplot for various columns in the dataset
for i in train:
    if (train[i].nunique())<=25:
        sns.countplot(x=train[i])
        plt.show()


# 1. In this dataset, the vast majority of phones are equipped with a front camera, with only a quarter lacking this feature.
# 2. It can be observed that approximately 94% of phones in this dataset have a camera.
# 3. The talktime among phones in this dataset is uniformly distributed.

# In[17]:


#Plotting histogram to understand more about numerical features
num_col= ['battery_power','clock_speed','fc','int_memory', 'm_dep', 'mobile_wt','n_cores','pc', 'px_height','px_width', 'ram', 'sc_h', 'sc_w', 'talk_time']
train.hist(num_col,figsize=(16,10), color = "lightblue", ec="red")
plt.show()


# In[18]:


#distplot on various columns of the dataset
plt.figure(figsize=(30,10))
plt.subplot(331)
sns.distplot(train["battery_power"])

plt.subplot(332)
sns.distplot(train["clock_speed"])

plt.subplot(333)
sns.distplot(train["int_memory"])

plt.subplot(334)
sns.distplot(train["mobile_wt"])

plt.subplot(335)
sns.distplot(train["px_height"])

plt.subplot(336)
sns.distplot(train["px_width"])

plt.subplot(337)
sns.distplot(train["ram"])

plt.subplot(338)
sns.distplot(train["talk_time"])

plt.subplot(339)
sns.distplot(train["m_dep"])

plt.show()


# In[19]:


#Analysis with respect to target variable "price_range"

for i in train.columns[:-1]:
    plt.figure(figsize=(10,10))
    sns.histplot(x=train[i], hue=train["price_range"], multiple="stack")
    print("Distribution of ",i ,"with respect to price range")
    plt.show()


# In[20]:


train['blue'].value_counts()


# In[21]:


labels=['No','Yes']
plt.figure(figsize=(5,5))
train['blue'].value_counts().plot(kind="pie",autopct='%.1f%%',labels=labels,figsize=(4,4))
plt.ylabel('Have Bluetooth')
plt.show()


# In[22]:


train['dual_sim'].value_counts()


# In[23]:


labels=['No','Yes']
plt.figure(figsize=(5,5))
train['dual_sim'].value_counts().plot(kind="pie",autopct='%.1f%%',labels=labels,figsize=(4,4))
plt.ylabel('Have Dual Sim')
plt.show()


# In[24]:


train['four_g'].value_counts()


# In[25]:


labels=['No','Yes']
plt.figure(figsize=(5,5))
train['four_g'].value_counts().plot(kind="pie",autopct='%.1f%%',labels=labels,figsize=(4,4))
plt.ylabel('Have 4G')
plt.show()


# In[26]:


train['three_g'].value_counts()


# In[27]:


labels=['No','Yes']
plt.figure(figsize=(5,5))
train['three_g'].value_counts().plot(kind="pie",autopct='%.1f%%',labels=labels,figsize=(4,4))
plt.ylabel('Have 3G')
plt.show()


# In[28]:


train['touch_screen'].value_counts()


# In[29]:


labels=['No','Yes']
plt.figure(figsize=(5,5))
train['touch_screen'].value_counts().plot(kind="pie",autopct='%.1f%%',labels=labels,figsize=(4,4))
plt.ylabel('Have Touch Screen')
plt.show()


# In[30]:


train['wifi'].value_counts()


# In[31]:


labels=['No','Yes']
plt.figure(figsize=(5,5))
train['wifi'].value_counts().plot(kind="pie",autopct='%.1f%%',labels=labels,figsize=(4,4))
plt.ylabel('Have Wifi')
plt.show()


# In[32]:


sns.countplot(x='n_cores',data=train, palette='CMRmap')
plt.title("Number of cores of processor")
plt.xlabel('Single,    Dual,      Triple,     Quad,    Penta,    Hexa,    Hepta,    Octa')
plt.show()


# * Bluetooth, dual sim, 4G, touchscreen, and wifi are features present in nearly 50% of the phones.
# * About 75% of the phones possess 3G.
# * The various types of cores are distributed almost evenly.

# In[33]:


#Line plot for all columns
a=train.drop('price_range',axis=1)
plt.figure(figsize=(14,8))
for i, j in enumerate(a):
    plt.subplot(4,5, i+1)
    sns.lineplot(y=train[j],x=train["price_range"],color ='#FFB90F')
plt.tight_layout()


# * The higher the price range higher the average ram size.
# * We can see that the higher the price range it tend to have a higher battery power except on the price range 1 to 2. It seems that from price range 1 to 2 there's no significance difference in the average  battery power 
# * Clock spped is highest for mobiles in lower price range.
# * The price range of 0 has more products with lower pixel width and pixel height while the highest price range has more products with higher pixel width and pixel height.
# * It seems that talk time doesn't really affect the price range.
# * Higher price range has a higher megapixel primary camera.

# In[34]:


# Checking for multicollinearity
plt.figure(figsize=(16,8))
sns.heatmap(train.corr(), cmap="YlGnBu", annot=True)
plt.show()


# * Almost all features exhibit low correlation with each other.
# * 'ram' is highly correlated with price range.

# ## Preprocessing

# ### Missing Values Handling

# In[35]:


train.isnull().sum()


# No missing values present in the dataset

# ### Outlier detection

# In[36]:


#boxplot of each column
train.boxplot(figsize=(20,22))
plt.show()


# In[37]:


num_col= ['battery_power','clock_speed','fc','int_memory', 'm_dep', 'mobile_wt','pc', 'px_height','px_width', 'ram', 'sc_h', 'sc_w', 'talk_time']
plt.figure(figsize=(15, 7))
for i in range(0, len(num_col)):
    plt.subplot(5, 3, i+1)
    sns.boxplot(x=train[num_col[i]],orient='v')
    plt.tight_layout()


# There are outliers in the columns 'fc' and 'px_height'. Since these are genuine values we need not replace or remove it.

# ## Feature Scaling

# In[38]:


# Min-max scaling
df = train.drop(['battery_power','clock_speed','fc','int_memory', 'm_dep', 'mobile_wt','n_cores','pc', 'px_height','px_width', 'ram', 'sc_h', 'sc_w', 'talk_time'],axis=1)
X = train.drop(['blue','dual_sim','four_g','three_g','touch_screen','wifi','price_range'], axis=1)
X


# In[39]:


from sklearn.preprocessing import MinMaxScaler  #importing the required library for MinMax scaling
minmax = MinMaxScaler(feature_range=(0,1))  #creating instance
X= minmax.fit_transform(X)  #Performing MinMax scaling
X


# In[40]:


X = pd.DataFrame(X)         
X = pd.DataFrame(X)         
X.columns = ['battery_power','clock_speed','fc','int_memory', 'm_dep', 'mobile_wt','n_cores','pc', 'px_height','px_width', 'ram', 'sc_h', 'sc_w', 'talk_time'] # Giving the columns their respective names
X


# In[41]:


df = pd.concat([df,X],axis = 1)


# In[42]:


df.head()


# ### splitting features and target

# In[43]:


# Split the dataset into features and target
x = df.drop('price_range', axis=1)
y = df['price_range']


# In[44]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest =train_test_split(x,y, test_size=0.25,random_state = 42)


# ## Modelling

# ### Logistic Regression

# In[45]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
model_lr = lr.fit(xtrain,ytrain)
ypred_lr = model_lr.predict(xtest)
# checking the validation of the model
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score,classification_report
print(classification_report(ytest,ypred_lr))
print('Accuracy score is:',accuracy_score(ytest,ypred_lr))
print('f1 score is:', f1_score(ytest, ypred_lr,average='weighted'))
al = accuracy_score(ytest,ypred_lr)
print(al)


# ### Decision Tree Classifier

# In[46]:


from sklearn.tree import DecisionTreeClassifier
#creating an instance
dt_clf = DecisionTreeClassifier(random_state =42)
#fitting the model
dt_clf.fit(xtrain,ytrain)
ypred_dt = dt_clf.predict(xtest)
print(classification_report(ytest,ypred_dt))
print('Accuracy score is:',accuracy_score(ytest,ypred_dt))
print('f1 score is:', f1_score(ytest, ypred_dt,average='weighted'))
ad = accuracy_score(ytest,ypred_dt)
print(ad)


# ### Random Forest Classifier

# In[47]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score,classification_report
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(xtrain,ytrain)
ypred_random = rf_clf.predict(xtest)
print(classification_report(ytest,ypred_random))
print('Accuracy score is:',accuracy_score(ytest,ypred_random))
print('f1 score is:', f1_score(ytest, ypred_random,average='weighted'))
ar = accuracy_score(ytest,ypred_random)
print(ar)


# ### KNN

# In[48]:


from sklearn.neighbors import KNeighborsClassifier
metric_k= []
neighbors = np.arange(3, 15)

for k in neighbors:
  knn=KNeighborsClassifier(n_neighbors=k)
  model_knn =knn.fit(xtrain,ytrain)
  y_pred_knn = model_knn.predict(xtest)
  acc=accuracy_score(ytest,y_pred_knn)
  metric_k.append(acc)
# accuracy array
metric_k 


# In[49]:


#plotting the accuracy for each k value
plt.plot(neighbors,metric_k,'o-')
plt.xlabel('k value')
plt.ylabel('accuracy')
plt.grid()


# In[50]:


# accuracy is more when k=13 so we can create model using k = 13
knn = KNeighborsClassifier(n_neighbors=13)
model_knn = knn.fit(xtrain,ytrain)
y_pred_knn = model_knn.predict(xtest)
print(classification_report(ytest,y_pred_knn))
print('Accuracy score is:',accuracy_score(ytest,y_pred_knn))
print('f1 score is:', f1_score(ytest, y_pred_knn,average='weighted'))
ak = accuracy_score(ytest,y_pred_knn)
print(ak)


# ### Gradient Boost Classifier

# In[51]:


from sklearn.ensemble import GradientBoostingClassifier
# Define the gradient boosting classifier model
gb = GradientBoostingClassifier(random_state=42)

# Train the model on the training data
gb_model=gb.fit(xtrain, ytrain)
y_pred_gb = gb_model.predict(xtest)
print(classification_report(ytest,y_pred_gb))
print('Accuracy score is:',accuracy_score(ytest,y_pred_gb))
print('f1 score is:', f1_score(ytest, y_pred_gb,average='weighted'))
ag = accuracy_score(ytest,y_pred_gb)
print(ag)


# ### Choosing Best Model 

# In[52]:


classifiers=["Logistic Regression","Random Forest","Decision Tree","Knn","Gradient Boost"]
# accuracy = [al,ar,ad,ak,ag]
accuracy = [0.91,0.87,0.80,0.47,0.88]
f1_score = [0.91,0.86,0.80,0.47,0.88]
df_af=pd.DataFrame({'model':classifiers,"accuracy":accuracy,"f1-score":f1_score})
sns.barplot(data=df_af,x="model",y="accuracy")
plt.xticks(rotation ='90')
plt.show()


# ### Logistic Regression is the best model

# #### We need to identify the most important features and rank the features based on their importance scores.
# #### We could Identify the most important features using feature selection techniques such as correlation analysis, mutual information, and feature importance scores from machine learning models.

# ### Identifying important features and ranking them
# #### Logistic Regression Model

# Logistic regression does not have an attribute for ranking feature. If you want to visualize the coefficients that you 
# can use to show feature importance.

# After the model is fitted, the coefficients are stored in the coef_ property.
# 
# The following code trains the logistic regression model, creates a data frame in which the attributes are stored with their respective coefficients, and sorts that data frame by the coefficient in descending order:

# In[54]:


importance_scores_lr = pd.DataFrame(data ={
    'Attribute': xtrain.columns,
    'Importance': np.abs(model_lr.coef_[0])
})

importance_scores_lr = importance_scores_lr.sort_values(by='Importance', ascending=False)


# In[55]:


# Create a DataFrame to store the feature names and their importance scores
feature_scores = pd.DataFrame({'Feature': x.columns,'Importance': np.abs(model_lr.coef_[0])})

# Sort the features based on their importance scores in descending order
feature_scores = feature_scores.sort_values(by='Importance', ascending=False)

# Rank the features based on their importance scores
feature_scores['Rank'] = np.arange(1, len(x.columns) + 1)

# Display the ranked features
print(feature_scores[['Rank', 'Feature', 'Importance']]) 


# In[56]:


importance_scores_lr = pd.DataFrame({'feature': x.columns, 'importance': np.abs(model_lr.coef_[0])})
importance_scores_lr = importance_scores_lr.sort_values('importance',ascending=False)
importance_scores_lr.plot.bar(x='feature', figsize=(8,5),fontsize=10)
plt.show()


# #### Gradient Boost Model

# In[60]:


# Compute the feature importance scores using a machine learning model
importance_scores_gb = gb_model.feature_importances_

# Create a DataFrame to store the feature names and their importance scores
feature_scores_gb = pd.DataFrame({'Feature': x.columns, 'Importance': importance_scores_gb})

# Sort the features based on their importance scores in descending order
feature_scores_gb = feature_scores_gb.sort_values(by='Importance', ascending=False)

# Rank the features based on their importance scores
feature_scores_gb['Rank'] = np.arange(1, len(x.columns) + 1)

# Display the ranked features
print(feature_scores_gb[['Rank', 'Feature', 'Importance']]) 


# In[61]:


# Plotting features according to its importance
importance_scores_gb = pd.DataFrame({'feature': x.columns, 'importance': np.round(gb_model.feature_importances_,3)})
importance_scores_gb = importance_scores_gb.sort_values('importance',ascending=False)
importance_scores_gb.plot.bar(x='feature', figsize=(8,5),fontsize=10)
plt.show()


# #### Random Forest Model

# In[62]:


# Compute the feature importance scores using a machine learning model
importance_scores_rf = rf_clf.feature_importances_

# Create a DataFrame to store the feature names and their importance scores
feature_scores_rf = pd.DataFrame({'Feature': x.columns, 'Importance': importance_scores_rf})

# Sort the features based on their importance scores in descending order
feature_scores_rf = feature_scores_rf.sort_values(by='Importance', ascending=False)

# Rank the features based on their importance scores
feature_scores_rf['Rank'] = np.arange(1, len(x.columns) + 1)

# Display the ranked features
print(feature_scores_rf[['Rank', 'Feature', 'Importance']]) 


# In[63]:


# Plotting features according to its importance
importance_scores_rf = pd.DataFrame({'feature': x.columns, 'importance': np.round(rf_clf.feature_importances_,3)})
importance_scores_rf = importance_scores_rf.sort_values('importance',ascending=False)
importance_scores_rf.plot.bar(x='feature', figsize=(8,5),fontsize=10)
plt.show()


# #### Decision Tree Classifier

# In[64]:


# Compute the feature importance scores using a machine learning model
importance_scores_dt = dt_clf.feature_importances_

# Create a DataFrame to store the feature names and their importance scores
feature_scores_dt= pd.DataFrame({'Feature': x.columns, 'Importance': importance_scores_dt})

# Sort the features based on their importance scores in descending order
feature_scores_dt = feature_scores_dt.sort_values(by='Importance', ascending=False)

# Rank the features based on their importance scores
feature_scores_dt['Rank'] = np.arange(1, len(x.columns) + 1)

# Display the ranked features
print(feature_scores_dt[['Rank', 'Feature', 'Importance']]) 


# In[65]:


# Plotting features according to its importance
importance_scores_dt = pd.DataFrame({'feature': x.columns, 'importance': np.round(dt_clf.feature_importances_,3)})
importance_scores_dt = importance_scores_dt.sort_values('importance',ascending=False)
importance_scores_dt.plot.bar(x='feature', figsize=(8,5),fontsize=10)
plt.show()


# ### Loading Test Dataset

# In[66]:


test = pd.read_csv('MobileTest .csv')
test.head()


# In[67]:


test.shape


# In[68]:


test.columns


# In[69]:


test.info()


# In[70]:


#Checking for null values
test.isnull().sum()


# No null values present in the test dataset

# In[71]:


# Min-max scaling
df_test1 = test.drop(['battery_power','clock_speed','fc','int_memory', 'm_dep', 'mobile_wt','n_cores','pc', 'px_height','px_width', 'ram', 'sc_h', 'sc_w', 'talk_time'],axis=1)
Y = test.drop(['id','blue','dual_sim','four_g','three_g','touch_screen','wifi'], axis=1)
Y


# In[72]:


from sklearn.preprocessing import MinMaxScaler  #importing the required library for MinMax scaling
minmax = MinMaxScaler(feature_range=(0,1))  #creating instance
Y = minmax.fit_transform(Y)  #Performing MinMax scaling
Y


# In[73]:


Y = pd.DataFrame(Y)         
Y.columns = ['battery_power','clock_speed','fc','int_memory', 'm_dep', 'mobile_wt','n_cores','pc', 'px_height','px_width','ram', 'sc_h', 'sc_w', 'talk_time'] # Giving the columns their respective names
Y


# In[74]:


df_test = Y
df_test = pd.concat([df_test,df_test1],axis = 1)


# In[75]:


df_test


# ### Prediction Using Test Data with Logistic Regression Model

# In[77]:


df_test =df_test.drop('id', axis=1)


# In[78]:


ypred_lr = model_lr.predict(df_test)
result =pd.DataFrame(ypred_lr)
print(result)


# In[79]:


result.nunique()


# In[ ]:




