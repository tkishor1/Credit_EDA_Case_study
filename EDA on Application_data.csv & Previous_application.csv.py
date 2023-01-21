#!/usr/bin/env python
# coding: utf-8

# In[14]:


# For suppressing Warnings
import warnings
warnings.filterwarnings("ignore")


# In[108]:


#importing libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
sns.set_style('whitegrid')


# In[109]:


#importing dataset
df=pd.read_csv('application_data.csv')


# In[113]:


df.head(10)


# In[18]:


df.shape #  307511=rows , 122=columns


# In[19]:


df.info(verbose=True)


# In[20]:


df.isna().sum().sort_values(ascending=False).head(60)


# In[21]:


x=len(df)/2
x


# In[22]:


df.columns[df.isnull().sum() < x ]


# In[23]:


len(df.columns[df.isnull().sum()<x])


# In[24]:


df=df[df.columns[df.isnull().sum()<x]]
df.shape


# In[25]:


df.isna().sum().sort_values(ascending=False).head(60)


# In[26]:


df.describe().columns


# In[27]:


list(set(df.columns)-set(df.describe().columns))


# In[28]:


# Defining a function to find out the number and percentage of missing values in a Dataframe

def df_missing(df):
    
    missing_values_sum = df.isnull().sum()   
        
    missing_values_per = (df.isnull().sum()*100/len(df)).round(1) 
    
    missing_info = pd.concat([missing_values_sum,missing_values_per],axis=1) 
    
    missing_info.columns = ['Total', '% of NA values'] 
    
    missing_info.sort_values(by = '% of NA values',ascending=False )
           

        
    missing_info = missing_info[missing_info.iloc[:,1] != 0].sort_values('% of NA values', ascending=False)
    
    return missing_info


# In[29]:


df_missing(df)


# In[30]:


df['FLOORSMAX_AVG'].mode()


# In[31]:


df['FLOORSMAX_AVG'].describe()


# In[32]:


plt.figure(figsize=(7,5))
plt.hist(df['FLOORSMAX_AVG'],bins=10)
plt.title("Distribution of FLOORSMAX_AVG")
plt.show()


# In[33]:


df['FLOORSMAX_AVG'] = df['FLOORSMAX_AVG'].fillna(round(df['FLOORSMAX_AVG'].mean(),2))


# In[34]:


df['FLOORSMAX_AVG'].isna().sum()


# In[35]:


plt.figure(figsize=(7,5))
plt.hist(df['FLOORSMAX_AVG'])
plt.title("Distribution of FLOORSMAX_AVG")
plt.show()


# In[36]:


df_missing(df)


# In[37]:


cat_var = df[df_missing(df).index].select_dtypes('object')
cat_var.columns


# In[38]:


df["EMERGENCYSTATE_MODE"].value_counts(normalize=True).plot(kind='bar');


# In[39]:


df["EMERGENCYSTATE_MODE"].mode()[0]


# In[40]:


df["EMERGENCYSTATE_MODE"]= df["EMERGENCYSTATE_MODE"].fillna(df["EMERGENCYSTATE_MODE"].mode()[0])
df["EMERGENCYSTATE_MODE"].value_counts()


# In[41]:


df["EMERGENCYSTATE_MODE"].value_counts().plot(kind='bar')
plt.title("Count of EMERGENCYSTATE_MODE")
plt.xlabel("EMERGENCYSTATE_MODE")
plt.show()


# In[42]:


df_missing(df)


# In[43]:


df['OCCUPATION_TYPE'].head()


# In[44]:


plt.figure(figsize=(9,6))
sns.countplot(x = 'OCCUPATION_TYPE', data=df,palette='viridis',)
plt.title("Count of Individuals based on their Occupation Type",size=11)
plt.xticks(rotation = 90);


# In[45]:


average_income = pd.pivot_table(data = df,index="OCCUPATION_TYPE",aggfunc='mean')['AMT_INCOME_TOTAL']
average_income = round(average_income,2).sort_values(ascending = True)
average_income


# In[46]:


plt.figure(figsize=(8,6))
average_income.plot(kind='barh',title='Average Income vs Occupation')
plt.xlabel('Income',size=14);


# In[47]:


df['OCCUPATION_TYPE'].mode()


# In[48]:


cols = list(set(df.columns) - set(df.describe().columns))
df[cols]= df[cols].fillna(df.mode().iloc[0])


# In[49]:


df_missing(df) 


# In[50]:


null_var = df.isnull().sum()
null_var[null_var>0]


# In[51]:


df["AMT_ANNUITY"] = df["AMT_ANNUITY"].fillna(df["AMT_ANNUITY"].mean())
df["CNT_FAM_MEMBERS"] = df["CNT_FAM_MEMBERS"].fillna(df["CNT_FAM_MEMBERS"].mean())
df["DAYS_LAST_PHONE_CHANGE"] = df["DAYS_LAST_PHONE_CHANGE"].fillna(df["DAYS_LAST_PHONE_CHANGE"].mean())


# In[52]:


null_var = df.isnull().sum()
null_var[null_var>0]


# In[53]:


df_missing(df).index 


# In[114]:


df[df_missing(df).index] = df[df_missing(df).index].fillna(value=df[df_missing(df).index].mean())


# In[55]:


df_missing(df)


# In[56]:


df[df.duplicated()]


# In[57]:


df['NAME_CONTRACT_TYPE'].value_counts() 


# In[58]:


sns.countplot(df['NAME_CONTRACT_TYPE'])
plt.title('Count of contract Type',size=14)
plt.show()


# In[59]:


gender_info = df['CODE_GENDER'].value_counts(normalize=True)*100


# In[60]:


gender_info.plot(kind='bar')
plt.title("Count of Gender", size = 13)
plt.xlabel("Gender Type")
plt.show()  


# In[61]:


df['AMT_INCOME_TOTAL'].describe()


# In[62]:


df['AMT_INCOME_TOTAL'].plot(kind='box')
plt.title("Boxplot for Income ")
plt.show()  


# In[63]:


Q1=df['AMT_INCOME_TOTAL'].quantile(0.25)
Q3=df['AMT_INCOME_TOTAL'].quantile(0.75)

IQR=Q3-Q1

print("Q1 is " + str(Q1))
print("Q3 is " + str(Q3))
print("IQR is " + str(IQR))

Lower_Whisker = Q1- 1.5*IQR
Upper_Whisker = Q3 + 1.5*IQR

print("Lower_Whisker is " + str(Lower_Whisker)," and   Upper_Whisker is " + str(Upper_Whisker))


# In[64]:


df  = df[df['AMT_INCOME_TOTAL']< Upper_Whisker] 


# In[65]:


df['AMT_INCOME_TOTAL'].plot(kind='box') 
plt.title("Boxplot for Income ")
plt.show()


# In[66]:



df.shape 


# In[67]:


df['AMT_CREDIT'].describe()


# In[68]:


plt.figure(figsize=(8,8))
sns.distplot(df['AMT_CREDIT'])
plt.title("Distritbution of Amount Credit")
plt.show() 


# In[69]:


plt.figure(figsize=(12,5))
sns.boxplot(df['AMT_CREDIT'])
plt.title("Boxplot of Amount Credit")
plt.show()


# In[70]:


Q1=df['AMT_CREDIT'].quantile(0.25)
Q3=df['AMT_CREDIT'].quantile(0.75)

IQR=Q3-Q1

print("Q1 is " + str(Q1))
print("Q3 is " + str(Q3))
print("IQR is " + str(IQR))

Lower_Whisker = Q1- 1.5*IQR
Upper_Whisker = Q3 + 1.5*IQR

print("Lower_Whisker is " + str(Lower_Whisker)," and   Upper_Whisker is " + str(Upper_Whisker))


# In[71]:


Upper_Whisker


# In[72]:


df  = df[df['AMT_CREDIT']< Upper_Whisker] 


# In[73]:


plt.figure(figsize=(8,5))
sns.boxplot(df['AMT_CREDIT'])
plt.title("Boxplot of Amount Credit")
plt.show()


# In[74]:


df.shape


# In[75]:


df['AMT_GOODS_PRICE'].describe()


# In[76]:


plt.figure(figsize=(8,5))
sns.boxplot(df['AMT_GOODS_PRICE'])
plt.title("Boxplot of Amount of Goods Price")
plt.show()


# In[77]:


Q1=df['AMT_GOODS_PRICE'].quantile(0.25)
Q3=df['AMT_GOODS_PRICE'].quantile(0.75)

IQR=Q3-Q1

print("Q1 is " + str(Q1))
print("Q3 is " + str(Q3))
print("IQR is " + str(IQR))

Lower_Whisker = Q1- 1.5*IQR
Upper_Whisker = Q3 + 1.5*IQR

print("Lower_Whisker is " + str(Lower_Whisker)," and   Upper_Whisker is " + str(Upper_Whisker))


# In[78]:


df  = df[df['AMT_GOODS_PRICE']< Upper_Whisker] 


# In[79]:


plt.figure(figsize=(8,5))
sns.boxplot(df['AMT_GOODS_PRICE'])
plt.title("Boxplot of Amount of Goods Price")
plt.show()


# In[80]:


df['NAME_INCOME_TYPE'].value_counts()


# In[81]:


plt.figure(figsize=(10,8))
sns.countplot(x = 'NAME_INCOME_TYPE',data=df,hue = 'TARGET')
plt.title("Income Type of the Individuals  with respect to Target")
plt.xticks(rotation = 90)
plt.show()


# In[82]:


df['DAYS_BIRTH'].head()


# In[83]:


df['DAYS_BIRTH'] = df['DAYS_BIRTH'].apply(lambda x: round(abs(x/365)))


# In[84]:


df['DAYS_BIRTH'].value_counts().head()


# In[85]:


plt.figure(figsize=(10,8))
sns.distplot(df['DAYS_BIRTH'],bins = 20)
plt.title("Distribution of Age of Applicants")
plt.show()


# In[86]:


print("Minimum age of the Applicant is  " + str(df['DAYS_BIRTH'].min()))
print("Maximum age of the Applicant is  " + str(df['DAYS_BIRTH'].max()))


# In[87]:


df['NAME_EDUCATION_TYPE'].head()


# In[88]:


df['NAME_EDUCATION_TYPE'].value_counts().index


# In[89]:


Education  = {'Secondary / secondary special':'Secondary', 'Higher education':'Higher Edu','Incomplete higher' :"Inc Higher", 'Lower secondary':'Lower secondary', 'Academic degree':'Degree'}


# In[90]:


df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].map(Education)


# In[91]:


plt.figure(figsize=(8,5))
df['NAME_EDUCATION_TYPE'].value_counts().plot(kind='bar')
plt.title("Distribution of Education of Applicants")
plt.xlabel("Education Type")
plt.show()


# In[92]:


df['TARGET'].head() 


# In[93]:


df['TARGET'].value_counts()


# In[94]:


df['TARGET'].value_counts(normalize = True).plot(kind='bar')
plt.title("Number of repaid with issues")
plt.show()


# In[95]:


(df['TARGET']==0).sum()


# In[96]:


(df['TARGET']==1).sum()


# In[97]:


df['TARGET'].sample(5)


# In[98]:


df['TARGET'].value_counts()


# In[99]:


Ratio_of_repaid_with_issue = (df['TARGET']==0).sum() / (df['TARGET']==1).sum()
Ratio_of_repaid_with_issue = round(Ratio_of_repaid_with_issue)
Ratio_of_repaid_with_issue

print( 'The Ratio between the Repayments and the issue payments is  ' + str(Ratio_of_repaid_with_issue))


# In[100]:


train_0 = df.loc[df['TARGET'] == 0 ]
train_1 = df.loc[df['TARGET'] == 1 ]


# In[101]:


cat_col = list(set(df.columns) - set(df.describe().columns))
cat_col


# In[102]:


num_col = df.describe().columns
num_col


# In[103]:


len(num_col)


# In[104]:


def plotting(train, train0, train1, column):
    
    train = train
    train_0=train0
    train_1=train1
    col = column
    
    fig = plt.figure(figsize=(13,10))
    
    ax1=plt.subplot(221)
    train[col].value_counts().plot.pie(autopct = "%1.0f%%", ax=ax1)
    plt.title('Plotting data for the column: '+ column)
    
    ax2 = plt.subplot(222)
    sns.countplot(x=column, hue = 'TARGET', data = train, ax = ax2)
    plt.xticks(rotation=90)
    plt.title('Plotting data for target in terms of total count')
    
    ax3 = plt.subplot(223)
    df=pd.DataFrame()
    
    
    df['0'] = ((train_0[col].value_counts())/len(train_0))
    df['1'] = ((train_1[col].value_counts())/len(train_1))
    df.plot.bar(ax=ax3)
    plt.title('Plotting data for target in terms of percentage')
    
    fig.tight_layout() #or equivalently, "plt.tight_Layout()"
    
    plt.show()


# In[105]:


train_categorical = df.select_dtypes(include =['object']).columns
train_categorical


# In[106]:


for column in train_categorical:
    print("Plotting ", column)
    plotting(df, train_0, train_1, column)


# In[117]:


import numpy as np
corr = train_0.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)]= True  
f, ax = plt.subplots(figsize = (11,9))
with sns.axes_style("white"):
    ax = sns.heatmap(corr, mask=mask, vmax = 0.3, square = True);


# In[120]:


train_0.corr()


# In[121]:


train_0.corr().shape


# In[123]:


train_0.corr().abs()


# In[126]:


train_0.corr().abs().unstack()


# In[129]:


train_0.corr().abs().unstack().sort_values(kind = 'quicksort')


# In[131]:


train_0.corr().abs().unstack().sort_values(kind = 'quicksort').dropna()


# In[133]:


correlation_0 = train_0.corr().abs().unstack().sort_values(kind = 'quicksort').dropna()
correlation_0


# In[135]:


correlation_0 = correlation_0[correlation_0 != 1.0]
correlation_0


# In[137]:


correlation_0.tail(10)


# In[138]:


corr = train_1.corr()


# In[140]:


import numpy as np
corr = train_1.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)]= True   #automatically mask or hide
f, ax = plt.subplots(figsize = (11,9))
with sns.axes_style("white"):
    ax = sns.heatmap(corr, mask=mask, vmax = 0.3, square = True);


# In[141]:


correlation_1 = train_1.corr().abs()
correlation_1 = correlation_1.unstack().sort_values(kind = 'quicksort')
correlation_1 = correlation_1.dropna()
correlation_1 = correlation_1[correlation_1 != 1.0]
print(correlation_1.tail(10))


# In[142]:


train_numerical = df.select_dtypes(include = ['int64', 'float64']).columns
train_numerical


# In[143]:


df.index


# In[144]:


for column in train_numerical:
    plt.scatter(df.index,df[column])
    plt.title("Plot of "+column)
    plt.show()


# In[149]:


#Importing the previous application dataset
previous_application = pd.read_csv("previous_application.csv")
previous_application.head()


# In[150]:


previous_application.shape


# In[151]:


previous_application = previous_application.sample(25000) 
previous_application.head()


# In[152]:


previous_application.shape


# In[153]:


previous_application.columns


# In[154]:


previous_application.SK_ID_PREV.value_counts()


# In[155]:


previous_train = df.merge(previous_application, left_on = 'SK_ID_CURR', right_on = 'SK_ID_CURR', how = 'inner')


# In[156]:


previous_train.shape


# In[157]:


previous_train.head()


# In[159]:


previous_application.columns.value_counts().head()


# In[160]:


train_0 = df.loc[df['TARGET'] == 0]
train_1 = df.loc[df['TARGET'] == 1]


# In[161]:


ptrain_0 = previous_train.loc[previous_train['TARGET'] == 0]
ptrain_1 = previous_train.loc[previous_train['TARGET'] == 1]


# In[162]:


def plotting(column, hue):
    col = column
    hue = hue
    fig = plt.figure(figsize = (13,10))
    
    ax1 = plt.subplot(221)
    df[col].value_counts().plot.pie(autopct = "%.0f%%", ax=ax1)
    plt.title('Plotting data for the column: '+column)
    
    ax2 = plt.subplot(222)
    df1 = pd.DataFrame()
    df1['0'] = ((train_0[col].value_counts())/len(train_0))
    df1['1'] = ((train_1[col].value_counts())/len(train_1))
    df1.plot.bar(ax=ax2)
    plt.title('Plotting data for target in terms of total count')
    
    ax3 = plt.subplot(223)
    sns.countplot(x=col, hue = hue, data =ptrain_0, ax=ax3)
    plt.xticks(rotation=90)
    plt.title('Plotting data for Target=0 interms of percentage')
    
    ax4 = plt.subplot(224)
    sns.countplot(x=col, hue=hue, data=ptrain_1, ax=ax4)
    plt.xticks(rotation=90)
    
    fig.tight_layout()
    
    plt.show()


# In[164]:


plotting('NAME_EDUCATION_TYPE', 'NAME_CONTRACT_STATUS')


# In[165]:


plotting('CODE_GENDER', 'TARGET')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




