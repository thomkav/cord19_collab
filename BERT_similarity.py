import warnings
warnings.filterwarnings("ignore")
import torch
from pymongo import MongoClient
import mongo_utils
import spacy
from spacy.lang.en import English
from scipy.spatial import distance
import numpy as np
from transformers import *
import transformers


assert transformers.__version__ == '2.6.0', print('need transformers version 2.6. Conda installs an old version, use pip!')
                                                  
#instantiate db connection
client = MongoClient('localhost', 56789)
db = client['CORD-19']

# BERT needs specially tokenized inputs, so we need to use HuggingFace's tokenizer object                          
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

# Only using spacy as a sentencizer here.
# The dependency parser component might give us better sentence boundaries, but for simplicity's sake
# I just used the rule-based sentencizer.
nlp = English()
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

doc_gen = mongo_utils.make_doc_gen(db)
spacy_docs = nlp.pipe(doc_gen, as_tuples=True)
with torch.no_grad():
    
    """
    First use transformer's tokenizer to get our encode our query, then run it through the model to 
    get the query_embedding. 
    
    Model output is a tuple of (last_hidden_state, pooled_output), where pooled output is the last hidden 
    state passed through another linear layer and activation function. According to the transformers docs, 
    'This output is usually not a good summary of the semantic content of the input, you’re often better 
    with averaging or pooling the sequence of hidden-states for the whole input sequence.'
    
    Note that last_hidden_state.shape == (batch_size, num_input_tokens, vocab_size). Since our batch_size = 1, we 
    "squeeze" (chop off) that dimension and then sum each token vector (i.e., along dim=0) to get a sentence-level embedding.
    """
    
    query= "Coronavirus cases exploded in March of 2020"
    query_encoding = tokenizer.encode(query, return_tensors='pt')
    query_embedding = model(query_encoding)[0].squeeze().sum(dim=0)

    for doc, context in spacy_docs:
        for sentence in doc.sents:
            input = tokenizer.encode(sentence.text, return_tensors='pt')
            output = model(input)[0]
            sentence_embedding =  output.squeeze().sum(dim=0)
            # calculate cosine distance between sentence and query embeddings. If small, print sentence. 
            if (1-distance.cosine(query_embedding, sentence_embedding))>0.75:
                print(sentence.text)



# not sure if we need to make a pytorch dataset object...

# class MongoDataset(IterableDataset):

#     def __init__(self, cursor: "pymongo cursor. Returned documents should have 'text' key.", nlp: "spacy language object",  
#                  query: "mongo-style query" = mongo_utils.TEXT_QUERY):
#         self.db = db
#         self.texts = (doc['text'] for doc in cursor)
#         self.pipe = nlp.pipe(self.texts)        
        
#     def sentence_iter(self):
#         for text in self.pipe:
#             yield from (sentence.text for sentence in text.sents)
    
#     def __iter__(self):
#         return sentence_iter
    

