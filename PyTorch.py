#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
torch.tensor([[1,2],[3,4]])


# In[ ]:


torch.tensor([[1,2,3],[4,5,6]])


# In[ ]:


torch.tensor([[1.,-1.],[1.,-1.]])


# In[ ]:


import numpy as np
torch.tensor(np.array([[1,2,3],[4,5,6]]))


# In[ ]:


torch.zeros([2,4],dtype=torch.int32)
torch.zeros([2,4],dtype=torch.int64)


# In[14]:


x=torch.tensor([[1,2,3],[4,5,6]])
print(x[1][2])

x[0][1]=8
print(x)


# In[27]:


# Use torch.Tensor.item() to get a Python number from a tensor containing a single value:
x=torch.tensor([[1]])
print(x)
tensor([[1]])
print(x.item())


# In[25]:


x=torch.tensor(2.5)
print(x)
print(x.item())


# In[29]:


x=torch.tensor([[1.,-1.],[1.,1.]],requires_grad=True)
out=x.pow(2).sum()
out.backward()
x.grad


# In[30]:


tensor=torch.ones((2,),dtype=torch.int8)
data=[[0,1],[2,3]]
tensor.new_tensor(data)


# In[ ]:


tensor=torch.ones((2,),dtype=torch.float64)
tensor.new_full((3,4),3.141592)


# In[9]:


import torch
# Tensors:PyTorch is a library for processing tensors. A tensor is a number, vector, matrix or any n-dimensional array. Let's create a tensor with a single number:
# Number
t1=torch.tensor(4.)
print(t1)
print(t1.dtype)


# In[10]:


# Vector
t2=torch.tensor([1.,2,3,4])
print(t2)


# In[11]:


# Matrix
t3=torch.tensor([[5.,6], 
                [7,8], 
                [9,10]])
print(t3)


# In[12]:


# 3-dimensional array
t4 = torch.tensor([
    [[11,12,13], 
     [13,14,15]], 
    [[15,16,17], 
     [17,18,19.]]])
print(t4)


# In[13]:


print(t1)
t1.shape
print(t2)
t2.shape
print(t3)
t3.shape
print(t4)
t4.shape


# In[18]:


# Tensor operations and gradients
# Create tensors.
x=torch.tensor(3.)
w=torch.tensor(4.,requires_grad=True)
b=torch.tensor(5.,requires_grad=True)
print(x,w,b)
# Arithmetic operations
y=w*x+b
print(y)
# Compute derivatives
y.backward()
# Display gradients
print('dy/dx:', x.grad)
print('dy/dw:', w.grad)
print('dy/db:', b.grad)


# In[22]:


# Interoperability with Numpy
import numpy as np
x=np.array([[1,2],[3,4.]])
print(x)

# Convert the numpy array to a torch tensor.
y=torch.from_numpy(x)
print(y)

x.dtype, y.dtype

# Convert a torch tensor to a numpy array
z=y.numpy()
print(z)


# In[ ]:


# Commit and upload the notebook
import jovian
jovian.commit()
# Tensors and Gradient:A Tensor is a number,vector,matrix or n-dimensional array.

