from dotenv import load_dotenv
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from redundant_filter_retriever import RedundantFilterRetriever
import langchain

langchain.debug = True

load_dotenv()

chat = ChatOpenAI()

embeddings = OpenAIEmbeddings()

db = Chroma(
  embedding_function=embeddings,
  persist_directory="emb"
)
retriever = RedundantFilterRetriever(
  embeddings=embeddings,
  chroma=db
)
# retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
  llm=chat,
  retriever=retriever,
  chain_type="stuff"
)

result = chain.run("What is an interesting fact about the English language?")

print(result)