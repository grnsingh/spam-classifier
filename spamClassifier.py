import streamlit as st
import pickle 
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords

tf = pickle.load(open('spam-ham/vectorizer.pkl','rb'))
model = pickle.load(open('spam-ham/bnbModel.pkl','rb'))


punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'


ps = PorterStemmer()
def transformText(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    rT = []
    for i in text:
        if i.isalnum():
            rT.append(ps.stem(i))

    text =rT[:]
    rT.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in punctuation:
            rT.append(i)

    return " ".join(rT)


st.title(" SPAM CLASSIFIER")
input= st.text_input("ENTER MESSAGE")

processedText= transformText(input)

if st.button("PREDICT"):
    newText= tf.transform([processedText])
    result = model.predict(newText)

    if(result==1):

        st.header("SPAM")
    else:
        
        st.header("NOT SPAM")


