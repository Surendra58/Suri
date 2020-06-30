#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install PyPDF2')


# In[3]:


import PyPDF2  
# creating a pdf file object 
pdfFileObj=open('C:\\Users\\SurinderKumar\\Desktop\\SURENDRA KUMAR.pdf','rb') 
# creating a pdf reader object 
pdfReader=PyPDF2.PdfFileReader(pdfFileObj) 
# printing number of pages in pdf file 
print(pdfReader.numPages) 
# creating a page object 
pageObj=pdfReader.getPage(0) 
# extracting text from page 
print(pageObj.extractText()) 
# closing the pdf file object 
pdfFileObj.close() 


# In[5]:


get_ipython().system('pip install camelot-py[cv]')


# In[15]:


get_ipython().system('pip install Ghostscript')


# In[22]:


import camelot
# PDF file to extract tables from
file="C:\\Users\\SurinderKumar\\Desktop\\SURENDRA KUMAR.pdf"
# extract all the tables in the PDF file
tables=camelot.read_pdf(file)
# number of tables extracted
#print("Total tables extracted:",tables.n)
# print the first table as Pandas DataFrame
#print(tables[0].df)

