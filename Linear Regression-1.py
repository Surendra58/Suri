#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import torch


# In[6]:


# Training Data
# Input (temp, rainfall, humidity)
inputs=np.array([[73,67,43], 
                   [91,88,64], 
                   [87,134,58], 
                   [102,43,37], 
                   [69,96,70]],dtype='float32')

# Targets (apples, oranges)
targets=np.array([[56,70], 
                    [81,101], 
                    [119,133], 
                    [22,37], 
                    [103,119]],dtype='float32')
# Convert inputs and targets to tensors
inputs=torch.from_numpy(inputs)
targets=torch.from_numpy(targets)
print(inputs)
print(targets)


# In[9]:


# Linear Regression Model from Scratch
# Convert inputs and targets to tensors
inputs=torch.from_np(inputs)
targets=torch.from_np(targets)
print(inputs)
print(targets)


# In[10]:


# Linear regression model from scratch
# Weights and biases
w=torch.randn(2,3,requires_grad=True)
b=torch.randn(2,requires_grad=True)
print(w)
print(b)


# In[11]:


def model(x):
    return x @ w.t()+b


# In[12]:


# Generate predictions
preds=model(inputs)
print(preds)


# In[13]:


# Compare with targets
print(targets)


# In[14]:


# Loss function
# Mean Square Error(MSE) loss
def mse(t1,t2):
    diff=t1-t2
    return torch.sum(diff*diff)/diff.numel()


# In[15]:


# Compute loss
loss=mse(preds,targets)
print(loss)


# In[16]:


# Compute gradients
# Compute gradients
loss.backward()


# In[17]:


# Gradients for weights
print(w)
print(w.grad)


# In[18]:


w.grad.zero_()
b.grad.zero_()
print(w.grad)
print(b.grad)


# In[19]:


# Adjust weights and biases using gradient descent
# Generate predictions
preds=model(inputs)
print(preds)


# In[20]:


# Calculate the loss
loss=mse(preds,targets)
print(loss)


# In[21]:


# Compute gradients
loss.backward()
print(w.grad)
print(b.grad)


# In[23]:


# Adjust weights & reset gradients
with torch.no_grad():
    w-=w.grad*1e-5
    b-=b.grad*1e-5
    w.grad.zero_()
    b.grad.zero_()


# In[ ]:


print(w)
print(b)


# In[24]:


# Calculate loss
preds=model(inputs)
loss=mse(preds,targets)
print(loss)


# In[25]:


# Train for multiple epochs
# Train for 100 epochs
for i in range(100):
    preds=model(inputs)
    loss=mse(preds,targets)
    loss.backward()
    with torch.no_grad():
        w-= w.grad*1e-5
        b-= b.grad*1e-5
        w.grad.zero_()
        b.grad.zero_()


# In[26]:


# Calculate loss
preds=model(inputs)
loss=mse(preds,targets)
print(loss)


# In[27]:


# Predictions
preds


# In[28]:


# Targets
targets


# In[29]:


import jovian


# In[ ]:


jovian.commit()

