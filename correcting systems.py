import streamlit as st
import numpy as np
import pandas as pd
import string
data=pd.read_csv(r'./Book1.csv')
data['answers'] = data['answers'].apply(lambda x:''.join([i for i in x if i not in string.punctuation]))
data['answers']=data['answers'].apply(lambda x:x.lower())
# ind=int(input('enter the question number'))
ind = st.number_input('Insert a number min is 0 and max is 37792',min_value=1,max_value=10)
# y=input('enter the ans')
y=st.text_area(label='enter the answer')
answers=data['answers']
# answers.iloc[-1]
if st.button('submit'):
    answers.loc[len(answers.index)]=y
    from sklearn.feature_extraction.text import TfidfVectorizer
    v=TfidfVectorizer()
    x=v.fit_transform(data['answers'])
    x1=x.todense()
    from sklearn.metrics.pairwise import cosine_similarity,linear_kernel
    cosine=cosine_similarity(x[-1],x[ind-1])
    st.write(cosine)


