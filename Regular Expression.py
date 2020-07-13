#!/usr/bin/env python
# coding: utf-8

# How to Use Regular Expressions in Python
# # https://www.thepythoncode.com/article/work-with-regular-expressions-in-python?utm_source=newsletter&utm_medium=email&utm_campaign=newsletter

# In[1]:


import re # stands for regular expression 
# a regular expression for validating a password
match_regex=r"^(?=.*[0-9]).{8,}$"
# a list of example passwords
passwords=["pwd","password","password1"]
for pwd in passwords:
    m=re.match(match_regex,pwd)
    print(f"Password: {pwd},validate password strength:{bool(m)}")


# In[ ]:


# match_regex is the regular expression responsible for validating the password criteria we mentioned earlier:
# ^: Start character.
# (?=.*[0-9]): Ensure string has at least a digit.
# .{8,}: Ensure string has at least 8 characters.
# $: End character.


# Search Method : re.search() method is to search for a specific pattern in a string

# In[2]:


import re
# part of ipconfig output
example_text = """
Wireless LAN adapter Wi-Fi:
   Connection-specific DNS Suffix  . :
   Link-local IPv6 Address . . . . . : fe80::380e:9710:5172:caee%2
   IPv4 Address. . . . . . . . . . . : 192.168.1.100
   Subnet Mask . . . . . . . . . . . : 255.255.255.0
   Default Gateway . . . . . . . . . : 192.168.1.1
"""
# regex for IPv4 address
ip_address_regex = r"((25[0-5]|(2[0-4]|1[0-9]|[1-9]|)[0-9])(\.(?!$)|$)){4}"
# use re.search() method to get the match object
match = re.search(ip_address_regex, example_text)
print(match)


# match.start() to get the index of the first character of the found pattern.
# match.end() to get the index of the last character fo the found pattern.
# match.span() to get both start and end as a tuple (start, end).
# match.group() to get the actual string found.

# Finding Multiple Matches

# In[3]:


import re
# fake ipconfig output
example_text = """
Ethernet adapter Ethernet:
   Media State . . . . . . . . . . . : Media disconnected
   Physical Address. . . . . . . . . : 88-90-E6-28-35-FA
Ethernet adapter Ethernet 2:
   Physical Address. . . . . . . . . : 04-00-4C-4F-4F-60
   Autoconfiguration IPv4 Address. . : 169.254.204.56(Preferred)
Wireless LAN adapter Local Area Connection* 2:
   Media State . . . . . . . . . . . : Media disconnected
   Physical Address. . . . . . . . . : B8-21-5E-D3-66-98
Wireless LAN adapter Wi-Fi:
   Physical Address. . . . . . . . . : A0-00-79-AA-62-74
   IPv4 Address. . . . . . . . . . . : 192.168.1.101(Preferred)
   Default Gateway . . . . . . . . . : 192.168.1.1
"""
# regex for MAC address
mac_address_regex=r"([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})"
# iterate over matches and extract MAC addresses
extracted_mac_addresses=[ m.group(0) for m in re.finditer(mac_address_regex,example_text)]
print(extracted_mac_addresses)


# Replacing Matches

# If you have experience on web scraping, you may be encoutered with a website that uses a service like CloudFlare to hide email addresses from email harvester tools. In this section, we will do exactly that, given a string that has email addresses, we will replace each one of the addresses by a '[email protected]' token:

# In[4]:


import re
# a basic regular expression for email matching
email_regex=r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
# example text to test with
example_text="""
Subject: This is a text email!
From: John Doe <john@doe.com>
Some text here!
===============================
Subject: This is another email!
From: Abdou Rockikz <example@domain.com>
Some other text!
"""
# substitute any email found with [email protected]
print(re.sub(email_regex,"[email protected]",example_text))


# In[ ]:




