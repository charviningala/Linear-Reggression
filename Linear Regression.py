#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importss
import numpy as np
import matplotlib.pyplot as plt


# In[15]:


#inputs
x=np.array([1200,1500,2000,1800,2500,3000,2200,2700],dtype=np.float64)
y=np.array([300000,320000,450000,400000,550000,600000,480000,700000],dtype=np.float64)
plt.figure(figsize=(8,6))
plt.title("Data Points")
plt.scatter(x,y)
plt.grid()
plt.show()


# In[16]:


n = x.shape[0]
sigma_x = np.sum(x)
sigma_y = np.sum(y)
sigma_x_y = np.sum(x * y)
sigma_x_square_element_wise = np.sum(x ** 2)

# Compute slope (m) and intercept (c)
m = ((n * sigma_x_y) - (sigma_x * sigma_y)) / (n * sigma_x_square_element_wise - sigma_x ** 2)
c = (sigma_y - m * sigma_x) / n

print(f"m: {m}, c: {c}")


# In[17]:


y=m*x+c
plt.plot(x,y,color="green")
plt.scatter(x,y)
plt.scatter(m,c,color='red',label="intersection point")



# In[18]:


x=x.reshape(-1,1)
y=y.reshape(-1,1)
theta=np.linalg.inv(x.T@x)@x.T@y
print(f"Closed form theta :{theta}")


# In[19]:


#linear regression fit
from sklearn.linear_model import LinearRegression
reg=LinearRegression().fit(x,y)
y_pred=reg.predict(x)
print(y_pred)


# In[20]:



y=np.array([300000,320000,450000,400000,550000,600000,480000,700000])

plt.figure(figsize=(8,6))
plt.title("Linear Regression Fit")
plt.scatter(x,y,color='blue',label='Original Data')
plt.plot(x,y_pred,color='red',label='Predicted Output')
plt.legend()
plt.grid()
plt.show()


# In[21]:


# predicting output of point 1940
x_point=np.array([[1940]])
y_output=reg.predict(x_point)
print("Cost of 1940 sq ft. = ",y_output)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




