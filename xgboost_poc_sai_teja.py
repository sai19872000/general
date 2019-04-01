
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("/home/sai/tejasa/bs140513_032310.csv")


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


len(df.category.unique())


# In[6]:


len(df.merchant.unique())


# In[7]:


len(df.customer.unique())


# In[8]:


import matplotlib.pyplot as plt
df.step.hist()


# In[9]:


len(df.age.unique())


# In[10]:


df.gender.unique()


# In[11]:


df.amount.hist()


# In[12]:


df[df.amount<300].amount.hist(bins=10)


# In[13]:


df.fraud.hist()


# In[14]:


df_fraud_only = df[df["fraud"]==1]


# In[15]:


df_fraud_only.head()


# In[16]:


len(df_fraud_only.merchant.unique())


# In[17]:


len(df_fraud_only.customer.unique())


# In[18]:


len(df_fraud_only.category.unique())


# In[19]:


df_fraud_only.head(20)


# In[20]:


df_fraud_only[df_fraud_only.amount<2000].amount.hist(bins=10)


# In[21]:


df.customer.unique()


# In[22]:


df.columns


# In[23]:


for col in ['customer', 'age', 'gender', 'zipcodeOri', 'merchant', 'zipMerchant', 'category']:
    df[col] = df[col].astype('object')


# In[24]:


df.dtypes


# In[25]:


df = df.replace({"""'""":""""""}, regex=True)


# In[26]:


df.head()


# In[27]:


len(df[(df["customer"]=="C1093826151") & (df["merchant"]=="M348934600")])


# In[28]:


len(df[(df["customer"]=="C1093826151") & (df["merchant"]=="M348934600") & (df["step"]<45)])


# In[29]:


#df.loc[(df["customer"]=="C1093826151") & (df["merchant"]=="M348934600") & (df["step"]<45),'cust_merch_45']=1 


# In[30]:


cols1 = [180,45,90,135]
cols2 = [150,1500,4000]
import itertools
for combination in itertools.product(cols1,cols2):
    print(combination)


# In[31]:


df.step.max()


# In[32]:


cols = ['customer','age','gender','merchant','category','amount']


# In[33]:


import itertools
for combination in itertools.combinations(cols,2):
    print(combination)


# In[34]:


cols_cat = ['customer','age','gender','merchant','category']
for combination in itertools.combinations(cols_cat,2):

    cust_merch_all = df.groupby([combination[0],combination[1]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_all"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1]], how='left')
    
    cust_merch_all = df[df.step < 45].groupby([combination[0],combination[1]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_step45"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1]], how='left')
    
    cust_merch_all = df[df.step < 90].groupby([combination[0],combination[1]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_step90"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1]], how='left')

    cust_merch_all = df[df.step < 135].groupby([combination[0],combination[1]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_step135"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1]], how='left')

    cust_merch_all = df[df.amount <= 150].groupby([combination[0],combination[1]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_amount150"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1]], how='left')

    
    cust_merch_all = df[df.amount <= 1500].groupby([combination[0],combination[1]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_amount1500"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1]], how='left')

    
    cust_merch_all = df[df.amount <= 4000].groupby([combination[0],combination[1]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_amount4000"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1]], how='left')

    
    cols1 = [180,45,90,135]
    cols2 = [150,1500,4000]

    for combi2 in itertools.product(cols1,cols2):
        cust_merch_all = df[(df.amount < combi2[0])&(df.amount <= combi2[1])].groupby([combination[0],combination[1]]).size()
        df_cust_merch_all = cust_merch_all.to_frame()
        df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"step"+str(combi2[0])+"amount"+str(combi2[1])})
        df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1]], how='left')

    print(combination[0],combination[1])


# In[35]:


cols_cat = ['customer','age','gender','merchant','category']
for combination in itertools.combinations(cols_cat,3):

    cust_merch_all = df.groupby([combination[0],combination[1],combination[2]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_"+combination[2]+"_all"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1],combination[2]], how='left')
    
    cust_merch_all = df[df.step < 45].groupby([combination[0],combination[1],combination[2]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_"+combination[2]+"_step45"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1],combination[2]], how='left')
    
    cust_merch_all = df[df.step < 90].groupby([combination[0],combination[1],combination[2]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_"+combination[2]+"_step90"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1],combination[2]], how='left')

    cust_merch_all = df[df.step < 135].groupby([combination[0],combination[1],combination[2]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_"+combination[2]+"_step135"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1],combination[2]], how='left')

    cust_merch_all = df[df.amount <= 150].groupby([combination[0],combination[1],combination[2]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_"+combination[2]+"_amount150"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1],combination[2]], how='left')

    
    cust_merch_all = df[df.amount <= 1500].groupby([combination[0],combination[1],combination[2]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_"+combination[2]+"_amount1500"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1],combination[2]], how='left')

    
    cust_merch_all = df[df.amount <= 4000].groupby([combination[0],combination[1],combination[2]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_"+combination[2]+"_amount4000"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1],combination[2]], how='left')

    
    cols1 = [180,45,90,135]
    cols2 = [150,1500,4000]

    for combi2 in itertools.product(cols1,cols2):
        cust_merch_all = df[(df.amount < combi2[0])&(df.amount <= combi2[1])].groupby([combination[0],combination[1],combination[2]]).size()
        df_cust_merch_all = cust_merch_all.to_frame()
        df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_"+combination[2]+"step"+str(combi2[0])+"amount"+str(combi2[1])})
        df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1],combination[2]], how='left')

    print(combination[0],combination[1],combination[2])


# In[36]:


cols_cat = ['customer','age','gender','merchant','category']
for combination in itertools.combinations(cols_cat,4):

    cust_merch_all = df.groupby([combination[0],combination[1],combination[2],combination[3]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_"+combination[2]+"_"+combination[3]+"_all"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1],combination[2],combination[3]], how='left')
    
    cust_merch_all = df[df.step < 45].groupby([combination[0],combination[1],combination[2],combination[3]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_"+combination[2]+"_"+combination[3]+"_step45"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1],combination[2],combination[3]], how='left')
    
    cust_merch_all = df[df.step < 90].groupby([combination[0],combination[1],combination[2],combination[3]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_"+combination[2]+"_"+combination[3]+"_step90"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1],combination[2],combination[3]], how='left')

    cust_merch_all = df[df.step < 135].groupby([combination[0],combination[1],combination[2],combination[3]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_"+combination[2]+"_"+combination[3]+"_step135"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1],combination[2],combination[3]], how='left')

    cust_merch_all = df[df.amount <= 150].groupby([combination[0],combination[1],combination[2],combination[3]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_"+combination[2]+"_"+combination[3]+"_amount150"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1],combination[2],combination[3]], how='left')

    
    cust_merch_all = df[df.amount <= 1500].groupby([combination[0],combination[1],combination[2],combination[3]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_"+combination[2]+"_"+combination[3]+"_amount1500"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1],combination[2],combination[3]], how='left')

    
    cust_merch_all = df[df.amount <= 4000].groupby([combination[0],combination[1],combination[2],combination[3]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_"+combination[2]+"_"+combination[3]+"_amount4000"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1],combination[2],combination[3]], how='left')

    
    cols1 = [180,45,90,135]
    cols2 = [150,1500,4000]

    for combi2 in itertools.product(cols1,cols2):
        cust_merch_all = df[(df.amount < combi2[0])&(df.amount <= combi2[1])].groupby([combination[0],combination[1],combination[2],combination[3]]).size()
        df_cust_merch_all = cust_merch_all.to_frame()
        df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_"+combination[2]+"_"+combination[3]+"step"+str(combi2[0])+"amount"+str(combi2[1])})
        df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1],combination[2],combination[3]], how='left')

    print(combination[0],combination[1],combination[2],combination[3])


# In[37]:


cols_cat = ['customer','age','gender','merchant','category']
for combination in itertools.combinations(cols_cat,5):

    cust_merch_all = df.groupby([combination[0],combination[1],combination[2],combination[3],combination[4]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_"+combination[2]+"_"+combination[3]+"_"+combination[4]+"_all"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1],combination[2],combination[3],combination[4]], how='left')
    
    cust_merch_all = df[df.step < 45].groupby([combination[0],combination[1],combination[2],combination[3],combination[4]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_"+combination[2]+"_"+combination[3]+"_"+combination[4]+"_step45"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1],combination[2],combination[3],combination[4]], how='left')
    
    cust_merch_all = df[df.step < 90].groupby([combination[0],combination[1],combination[2],combination[3],combination[4]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_"+combination[2]+"_"+combination[3]+"_"+combination[4]+"_step90"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1],combination[2],combination[3],combination[4]], how='left')

    cust_merch_all = df[df.step < 135].groupby([combination[0],combination[1],combination[2],combination[3],combination[4]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_"+combination[2]+"_"+combination[3]+"_"+combination[4]+"_step135"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1],combination[2],combination[3],combination[4]], how='left')

    cust_merch_all = df[df.amount <= 150].groupby([combination[0],combination[1],combination[2],combination[3],combination[4]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_"+combination[2]+"_"+combination[3]+"_"+combination[4]+"_amount150"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1],combination[2],combination[3],combination[4]], how='left')

    
    cust_merch_all = df[df.amount <= 1500].groupby([combination[0],combination[1],combination[2],combination[3],combination[4]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_"+combination[2]+"_"+combination[3]+"_"+combination[4]+"_amount1500"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1],combination[2],combination[3],combination[4]], how='left')

    
    cust_merch_all = df[df.amount <= 4000].groupby([combination[0],combination[1],combination[2],combination[3],combination[4]]).size()
    df_cust_merch_all = cust_merch_all.to_frame()
    df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_"+combination[2]+"_"+combination[3]+"_"+combination[4]+"_amount4000"})
    df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1],combination[2],combination[3],combination[4]], how='left')

    
    cols1 = [180,45,90,135]
    cols2 = [150,1500,4000]

    for combi2 in itertools.product(cols1,cols2):
        cust_merch_all = df[(df.amount < combi2[0])&(df.amount <= combi2[1])].groupby([combination[0],combination[1],combination[2],combination[3],combination[4]]).size()
        df_cust_merch_all = cust_merch_all.to_frame()
        df_cust_merch_all = df_cust_merch_all.rename(index=str, columns={0: combination[0]+"_"+combination[1]+"_"+combination[2]+"_"+combination[3]+"_"+combination[4]+"step"+str(combi2[0])+"amount"+str(combi2[1])})
        df = pd.merge(df, df_cust_merch_all, on=[combination[0],combination[1],combination[2],combination[3],combination[4]], how='left')

    print(combination[0],combination[1],combination[2],combination[3],combination[4])


# In[38]:


df.to_pickle("df_after_variable_creation.pkl")

