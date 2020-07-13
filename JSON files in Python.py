#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Saving JSON Data
import json
# example dictionary to save as JSON
data = {
    "first_name":"John",
    "last_name":"Doe",
    "email":"john@doe.com",
    "salary":1499.9, # just to demonstrate we can use floats as well
    "age":17,
    "is_real":False, # also booleans!
    "titles":["The Unknown","Anonymous"] # also lists!
}
# save JSON file
# 1st option
with open("data1.json","w") as f:
    json.dump(data,f)
    
# 2nd option
with open("data2.json","w") as f:
    f.write(json.dumps(data,indent=4))   
# json.dumps() function returns the dictionary as a JSON parsed string    


# In[ ]:


# Notice I added indent=4 this time as a parameter to json.dumps() function,this will pretty-print JSON array elements and object members,if you use indent=0,it'll only print new lines,and if it's None (default),then it's dumped in a single line (not human readable).The indent keyword exist both in dump() and dumps() functions.


# In[3]:


# Handling Non-ASCII Characters
# If your data contains non ASCII characters, and you don't want unicode instances on your JSON file (such as \u0623), then you should pass ensure_ascii=False to json.dump() function:

unicode_data = {
    "first_name": "أحمد",
    "last_name": "علي"
}

with open("data_unicode.json","w",encoding="utf-8") as f:
    json.dump(unicode_data,f,ensure_ascii=False)


# In[4]:


# Loading JSON Data
# read a JSON file
# 1st option
file_name = "data1.json"
with open(file_name) as f:
    data = json.load(f)
    
print(data)


# In[ ]:


# json.load() function will automatically return a Python dictionary, which ease our work with JSON files, here is the output:


# In[6]:


# Similarly, you can also use json.loads() function to read a string instead:

# 2nd option
file_name="data1.json"
with open(file_name) as f:
    data = json.loads(f.read())
print(data)


# In[ ]:




