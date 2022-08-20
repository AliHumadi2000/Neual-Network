#!/usr/bin/env python
# coding: utf-8

# # import necassry library that needed for this assignments

# In[1]:


import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns 
import warnings
warnings.simplefilter('ignore')
from sklearn.preprocessing import LabelEncoder


# # Forest Fire
# 
# ## problem statment 

# PREDICT THE BURNED AREA OF FOREST FIRES WITH NEURAL NETWORKS

# In[2]:


#load the forestfires dataset form the github repo 
df1=pd.read_csv("https://raw.githubusercontent.com/AliHumadi2000/Neual-Network/main/forestfires.csv")


# <h3 style="color:red">Data understaning and exploring 

# In[3]:


# visulize the first 10 columns 
df1.head(10)


# In[4]:


# basic infromatino about the data
df1.info()


# In[5]:


# shape of the dataset 
df1.shape
# 517 rows 
# 31 columns 


# In[6]:


# datatypes 
df1.dtypes


# In[7]:


# basic statastics of numerical features 
df1.describe().T


# In[8]:


# check if any missing or duplicated values 


# In[9]:


df1.isna().sum()


# In[10]:


df1.duplicated().sum()
# there is 8 duplicate values 


# In[11]:


df1[df1.duplicated()] # this is the duplicated values 


# In[12]:


df1=df1.drop_duplicates()
df1.shape


# # Data Visulization

# In[13]:


df1.size_category.value_counts()


# In[14]:


# plot the size of the fire category
sns.countplot(df1.size_category)
plt.show()


# In[15]:


# distibution of months and day 
sns.histplot(df1['month'])
plt.show()
# in aug and sep we can see the more records


# In[16]:


sns.histplot(df1.day)
plt.show()


# In[17]:


#distubtion of features
sns.displot(df1['FFMC'])
plt.show()


# In[18]:


sns.displot(df1['DMC'])
plt.show()
#not normaly 


# In[19]:


sns.displot(df1['DC'])
plt.show()


# In[20]:


sns.displot(df1['temp'])
plt.show()
#for temp


# In[21]:


#relation between Dc and temp
sns.scatterplot(df1['DC'],df1['temp'])
plt.show()


# In[22]:


#relation between fine fuel moisture code and temp
sns.scatterplot(df1['FFMC'],df1['temp'])
plt.show()


# In[23]:


#relation between fine fuel moisture code and DC
sns.scatterplot(df1['FFMC'],df1['DC'])
plt.show()


# In[24]:


#relation between temp and RH
sns.scatterplot(df1['RH'],df1['temp'])
plt.show()


# In[25]:


#relation between temp and wind
sns.scatterplot(df1['wind'],df1['temp'])
plt.show()


# # Data processing 

# In[26]:


#drop unwanted features
df1.drop(columns=['dayfri','daymon','daysat','daysun','daythu','daytue','daywed'],inplace=True)
df1.drop(columns=['monthapr','monthaug','monthdec','monthfeb','monthjan','monthjul','monthjun','monthmar','monthmay','monthnov','monthoct','monthsep'],inplace=True)


# In[27]:


df1


# In[28]:


#convert categorical to numerical 
label_encoder=LabelEncoder()
df1['M']=label_encoder.fit_transform(df1['month'])
df1['D']=label_encoder.fit_transform(df1['day'])


# In[29]:


df1


# In[30]:


df1.drop(columns='month',inplace=True)
df1.drop(columns='day',inplace=True)


# In[31]:


df1


# In[32]:


df1['size']=label_encoder.fit_transform(df1['size_category'])


# In[33]:


df1


# In[34]:


df1.drop(columns='size_category',inplace=True)


# In[35]:


df1


# In[36]:


X=df1.iloc[:,:-1]
Y=df1.iloc[:,-1]


# In[37]:


#features 
X


# In[38]:


#target
Y


# In[39]:


# Model

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(17,input_dim=11,activation='sigmoid'))
model.add(tf.keras.layers.Dense(8,activation='sigmoid'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))


# In[40]:


model.summary()


# In[41]:


# Compile model
model.compile(loss ='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[42]:


# Fit the model
model_fit=model.fit(X,Y,batch_size=70,validation_split=0.25,epochs=100)


# In[43]:


# Evaluating the model
score= model.evaluate(X,Y)
score


# In[44]:


model.metrics_names


# In[ ]:





# In[45]:


print("Accuracy of the model is:: ",score[1])


# In[46]:


model_fit.history.keys()


# In[47]:


# Plotting for training and testing data
plt.plot(model_fit.history['accuracy'],label='train')
plt.plot(model_fit.history['val_accuracy'],label='test')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim(0,1.1)
plt.legend(loc='best')


# In[48]:


# Plotting for training and testing data
plt.plot(model_fit.history['loss'],label='train')
plt.plot(model_fit.history['val_loss'],label='test')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.ylim(0,1)
plt.legend(loc='best')


# # Gas Turbine

# Problem statement: predicting turbine energy yield (TEY) using ambient variables as features.
# 
# Attribute Information:
# 
# 1. Variable (Abbr.) Unit Min Max Mean
# 2. Ambient temperature (AT) C â€“6.23 37.10 17.71
# 3. Ambient pressure (AP) mbar 985.85 1036.56 1013.07
# 4. Ambient humidity (AH) (%) 24.08 100.20 77.87
# 5. Air filter difference pressure (AFDP) mbar 2.09 7.61 3.93
# 6. Gas turbine exhaust pressure (GTEP) mbar 17.70 40.72 25.56
# 7. Turbine inlet temperature (TIT) C 1000.85 1100.89 1081.43
# 8. Turbine after temperature (TAT) C 511.04 550.61 546.16
# 9. Compressor discharge pressure (CDP) mbar 9.85 15.16 12.06
# 10. Turbine energy yield (TEY) MWH 100.02 179.50 133.51
# 11. Carbon monoxide (CO) mg/m3 0.00 44.10 2.37
# 12. Nitrogen oxides (NOx) mg/m3 25.90 119.91 65.29

# In[49]:


df2=pd.read_csv("https://raw.githubusercontent.com/AliHumadi2000/Neual-Network/main/gas_turbines.csv")


# In[50]:


df2


# The Target variable is Continuos, hence it will be a Regression problem.

# In[51]:


df2.info()


# In[52]:


df2.isna().sum()


# In[53]:


# Scaling the data
def scale(x):
    x=(x-x.min())/(x.max()-x.min())
    return x


# In[54]:


df_temp=df2


# In[55]:


df_temp


# In[56]:


y=df_temp.iloc[:,7]


# In[57]:


y


# In[58]:


df2.drop(columns='TEY',inplace=True)


# In[59]:


df2


# In[60]:


x=scale(df2.iloc[:,:])


# In[61]:


x


# In[62]:


# Feature--> 10


# In[63]:


# model creation

reg_model=tf.keras.models.Sequential()
reg_model.add(tf.keras.layers.Dense(35,input_dim=10,activation='relu'))
reg_model.add(tf.keras.layers.Dense(15,activation='relu'))
reg_model.add(tf.keras.layers.Dense(5,activation='relu'))
reg_model.add(tf.keras.layers.Dense(1,activation='linear'))


# In[64]:


reg_model.summary()


# In[65]:


# Compiling the model
reg_model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_squared_error'])


# In[66]:


# Fit the model
reg_model_fit=reg_model.fit(x,y,validation_split=0.2,epochs=100,batch_size=200)


# In[67]:


scores=reg_model.evaluate(x,y)
scores


# In[68]:


reg_model.metrics_names


# In[69]:


print("MSE ::",scores[1])


# In[70]:


reg_model_fit.history.keys()


# In[71]:


# Plotting for training and testing data
plt.plot(reg_model_fit.history['mean_squared_error'],label='train')
plt.plot(reg_model_fit.history['val_mean_squared_error'],label='test')
plt.title("Model MSE")
plt.xlabel("Epochs")
plt.ylabel("Mean_squared_error")
#plt.ylim(0,1)
plt.legend(loc='best')


# In[72]:


# Plotting for training and testing data
plt.plot(reg_model_fit.history['loss'],label='train')
plt.plot(reg_model_fit.history['val_loss'],label='test')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
#plt.ylim(0,1)
plt.legend(loc='best')


# In[73]:


x['predict']=reg_model.predict(x)


# In[74]:


x


# In[75]:


x['actual']=y


# In[76]:


x


# In[ ]:




