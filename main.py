from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import LlamaCpp

from langchain.llms import CTransformers
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# # define what documents to load
loader = DirectoryLoader("./", glob="*.txt", loader_cls=TextLoader)

# interpret information in the documents
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                          chunk_overlap=50)
texts = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'})

# create and save the local database
db = FAISS.from_documents(texts, embeddings)
db.save_local("faiss")


# prepare the template we will use when prompting the AI
template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful and full answer below and nothing else.
Helpful answer:
"""
# Only return the helpful answer below and nothing else.
# Helpful answer:

# load the language model
n_gpu_layers = 1  # Metal set to 1 is enough.
# # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
n_batch = 512
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# llm = CTransformers(model='./models/llama-2-7b-chat.ggmlv3.q4_0.bin',
#                     model_type='llama',
#                     config={'max_new_tokens': 256, 'temperature': 0.01},
#                     n_gpu_layers=n_gpu_layers,
#                     n_batch=n_batch,
#                     # callback_manager=callback_manager
#                     )
#  ctransofrmer

 


 



# (((((((((((*************************************)))))))))
llm = LlamaCpp(model_path="./models/llama-2-7b-chat.ggmlv3.q4_0.bin",
            #    model_type='llama',
            #    config={'max_new_tokens': 256, 'temperature': 0.01},
               n_gpu_layers=n_gpu_layers,
               n_batch=n_batch,
               f16_kv=True,
               callback_manager=callback_manager)
 # (((((((((((*************************************)))))))))

# load the interpreted information from the local database
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'})
db = FAISS.load_local("faiss", embeddings)

# prepare a version of the llm pre-loaded with the local content
retriever = db.as_retriever(search_kwargs={'k': 2})
prompt = PromptTemplate(
    template=template,
    input_variables=['context', 'question'])






qa_llm = RetrievalQA.from_chain_type(llm=llm,
                                    #  chain_type='stuff',
                                     retriever=retriever,
                                     return_source_documents=True,
                                     chain_type_kwargs={'prompt': prompt},
                                    #  verbose=True
                                    #n_gpu_layers=n_gpu_layers,
                                    #  n_batch=n_batch
                                     )

# ask the AI chat about information in our local files

prompt = "What do you think about this text?"
output = qa_llm({'query': prompt})
f = open("demofile2.txt", "a",encoding="utf-8")
f.write(output["result"])
f.close()
# output = qa_llm({'query': prompt})
# print(output["result"])