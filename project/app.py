# -*- coding: utf-8 -*-

import streamlit as st 
from transformers import BertTokenizerFast, BertForSequenceClassification
from PIL import Image

#LOADING THE MODEL
num_labels = 7 #total labels of our dataset
max_length = 512 #maximum lenghth we used in tokenizer and model arguments

target_names = ['angry', 'depression', 'fear', 'happy', 'neutral', 'sad','severe_depression']

# reload our model/tokenizer. Optional, only usable when in Python files instead of notebooks
model_path = 'D:\Depression Analysis\TransformersBertNemesisv2.4.3'

model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
tokenizer = BertTokenizerFast.from_pretrained(model_path)

#module status holders
is_contraction_modules_imported = False
is_lemmatizing_modules_imported = False
is_stopword_modules_imported = False
is_spelling_corrections_modules_imported = False

def clean_data(updated_list, expand_contractions=True, remove_punctuations=True, use_custom_punctuation_list=False, remove_digits=True, remove_tagsURLs=True, remove_unicodes=True, remove_stopwords=False, remove_extraspaces=False, lemmatizing=False,  spelling_corrections=True):
    
    global is_contraction_modules_imported
    global is_lemmatizing_modules_imported
    global is_stopword_modules_imported
    global is_spelling_corrections_modules_imported
    

    #Removing tags and URLs
    if remove_tagsURLs==True:
        import re
        list_holder = []

        for i in range(0,len(updated_list)):
            list_holder.append(re.sub(r'(?:\@|[PAD]|[CLS]|[UNK]|[SEP]|http?\://|https?\://|www)\S+', '', updated_list[i]))
        updated_list = list_holder
    
    
    # Removing punctuation marks
        punctuation_list = ''
        list_holder = updated_list

    if remove_punctuations==True and use_custom_punctuation_list==False:
        punctuation_list = '''!`,?.()-[]{};:"\<>/@#$%^&*_~'''
       
    elif remove_punctuations==True and use_custom_punctuation_list==True:
        punctuation_list = '''!`()-[]{};:"\<>/@#$%^&*_~'''
        

    def remove_punctuations(string,punctuation_list):
        for elements in string:
            if elements in punctuation_list:
                string = string.replace(elements, "")
        return string
    updated_list = [remove_punctuations(i,punctuation_list) for i in list_holder]

    
    # Expanding contractions
    if expand_contractions==True:

        if  is_contraction_modules_imported == False:
            is_contraction_modules_imported = True

        list_holder = []
        import contractions

        for word in updated_list:
            list_holder.append(contractions.fix(word))
        updated_list = list_holder
       

    # Correcting spelling mistakes
    if spelling_corrections==True:
        if is_spelling_corrections_modules_imported == False:
            is_spelling_corrections_modules_imported = True

        from textblob import TextBlob

        def correct_spellings(sentence):
            return str(TextBlob(sentence).correct())
        
        updated_list = [correct_spellings(i) for i in updated_list]


    # Removing digits
    if remove_digits==True:

        list_holder = []

        for i in range(0, len(updated_list)):
            list_holder.append(re.sub(r'\d', '', updated_list[i]))
        updated_list = list_holder


    # Removing unicode characters
    if remove_unicodes==True:
        list_holder = []

        for i in range(0, len(updated_list)):
            list_holder.append(updated_list[i].encode('ascii', 'ignore').decode())
        updated_list = list_holder
       

    # Removing stopwords
    if remove_stopwords==True:

        if  is_stopword_modules_imported == False:
            from gensim.parsing.preprocessing import remove_stopwords,STOPWORDS
            is_stopword_modules_imported = True

        list_holder = []

        for i in range(0, len(updated_list)):
            result_remove_stopwords = remove_stopwords(updated_list[i])
            list_holder.append(result_remove_stopwords)
        updated_list = list_holder


    # remove extra spaces
    if remove_extraspaces==True:

        for i in range(0, len(updated_list)):
            updated_list[i] = re.sub("\s\s+" , " ", updated_list[i]) # removes whitespaces longer than one
            updated_list[i]=updated_list[i].strip() # removes leading and trailing whitespaces.
       

    # lemmatizer
    if lemmatizing==True:

        if is_lemmatizing_modules_imported == False:
            import nltk # importing NLTK library
            nltk.download('punkt') # needs to divides a text into a list of sentences. Required in tokenization process
            nltk.download('wordnet') # WordNet is a database for the English language. Contains meanings of words, synonyms, antonyms and used in adding POS Tag process, lemmatization
            nltk.download('averaged_perceptron_tagger') # used for tagging words with their parts of speech (POS)
            is_lemmatizing_modules_imported = True

        
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import wordnet  # Lemmatize with POS Tag

        list_holder = updated_list

        # Init the Wordnet Lemmatizer
        lemmatizer = WordNetLemmatizer()

        def get_wordnet_pos(word):
            """Map POS tag to first character lemmatize() accepts"""
            tag = nltk.pos_tag([word])[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ,
                        "N": wordnet.NOUN,
                        "V": wordnet.VERB,
                        "R": wordnet.ADV}
            return tag_dict.get(tag, wordnet.NOUN)

        def lemmatizing_func(sentence):
            # Tokenize: Split the sentence into words
            # word_list = nltk.word_tokenize(sentence)
            # print(word_list)

            # Lemmatize list of words and join
            lemmatized_output = ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)])
            return lemmatized_output

        updated_list = [lemmatizing_func(i) for i in list_holder]

    return updated_list

def get_prediction(text):

    #cleaning inputs
    cleaned_text = clean_data(text, 
                        remove_tagsURLs=True,
                        remove_punctuations=True,
                        use_custom_punctuation_list=True,                            
                        expand_contractions=True,                            
                        spelling_corrections=True,                            
                        remove_digits=True, 
                        remove_unicodes=True,
                        remove_stopwords=False,                            
                        remove_extraspaces=False,                            
                        lemmatizing=False)

    # prepare our text into tokenized sequence
    inputs = tokenizer(cleaned_text, 
                    padding=True, 
                    truncation=True, 
                    max_length=max_length, 
                    return_tensors="pt")#.to("cuda")
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    return target_names[probs.argmax()]

#text = "I am depressed"
#w = (f'\nPrediction :{get_prediction(text)}')
#print(w)
#print(type(w))

def main():
    st.title("Depression Analysis")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Analyze my depression level</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    text1 = st.text_input("Input"," ")
    text = [text1]
    result =" "
    

    if st.button("Predict"):
        result = get_prediction(text)
        st.success('The output is {}'.format(result))
       
       
if __name__=='__main__':
    main()
    