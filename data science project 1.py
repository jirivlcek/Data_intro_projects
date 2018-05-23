
# coding: utf-8

# In[39]:


from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score


# In[35]:


# [height, weight, shoe_size]   learning data
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# In[36]:


# prediction data
_X=[[184,84,44],[198,92,48],[183,83,44],[166,47,36],[170,60,38],[172,64,39],[182,80,42],[180,80,43]]
_Y=['male','male','male','female','female','female','male','male']


# In[28]:


#classifiers
clf = tree.DecisionTreeClassifier()
forrest = RandomForestClassifier()
ada = AdaBoostClassifier()
svm = svm.SVC()


# In[33]:


#training models
clf = clf.fit(X, Y)
forrest = forrest.fit(X, Y)
ada = ada.fit(X, Y)
svm = svm.fit(X,Y)


# In[37]:


#prediction on prediction data
prediction1 = clf.predict(_X)
prediction2 = forrest.predict(_X)
prediction3 = ada.predict(_X)
prediction4 = svm.predict(_X)


# In[38]:


print(prediction1, prediction2, prediction3, prediction4)


# In[47]:


#accuracy score
r1 = accuracy_score(_Y, prediction1)
r2 = accuracy_score(_Y, prediction2)
r3 = accuracy_score(_Y, prediction3)
#r4 = accuracy_score(_Y, prediction4)

print(r1, r2, r2)


# In[46]:


#best result
if r1 > r2 and r1 > r3:
    print('tree : ',  r1)
elif r2 > r3 and r2 > r1:
    print('forrest:', r2)
else:
    print('ada :', r3)
    


# In[34]:


#test predikci na dvou cislech

#prediction1 = clf.predict([[185,75,43],[160,60,39]])
#prediction2 = forrest.predict([[185,75,43],[160,60,39]])
#prediction3 = ada.predict([[185,75,43],[160,60,39]])
#prediction4 = svm.predict([[185,75,43],[160,60,39]])


# In[32]:


#print(prediction1, prediction2, prediction3, prediction4)

