from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.vectorstores import FAISS
# from llama_index.embeddings import LangchainEmbedding
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_transformers import LongContextReorder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore

import gradio as gr
from functools import partial
from operator import itemgetter
import os

from tqdm import tqdm
from pathlib import Path

import os

nvidia_api_key = "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"
assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
os.environ["NVIDIA_API_KEY"] = nvidia_api_key


llm = ChatNVIDIA(model="ai-nemotron-4-340b-instruct")
embedder = NVIDIAEmbeddings(model="ai-embed-qa-4")


# 在这里我们读入文本数据并将它们准备到 vectorstore 中
ps = os.listdir("zh_data/")
data = []
sources = []
docs_name = []
for p in ps:
    if p.endswith('.txt'):
        path2file="zh_data/"+p
        docs_name.append(path2file)
        with open(path2file,encoding="utf-8") as f:
            lines=f.readlines()
            for line in lines:
                if len(line)>=1:
                    data.append(line)
                    sources.append(path2file)

documents=[d for d in data if d != '\n']
len(data), len(documents), data[0]

from operator import itemgetter
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import faiss
# create my own uuid
text_splitter = CharacterTextSplitter(chunk_size=400, separator=" ")
docs = []
metadatas = []

for i, d in enumerate(documents):
    splits = text_splitter.split_text(d)
    #print(len(splits))
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))
### 将创建好的embed存储到本地
store = FAISS.from_texts(docs, embedder , metadatas=metadatas)
store.save_local('./embed')

### 从本地读取已经创建好的embed
vecstores = [FAISS.load_local(folder_path="embed/", embeddings=embedder, allow_dangerous_deserialization=True)]

embed_dims = len(embedder.embed_query("test"))
def default_FAISS():
    '''Useful utility for making an empty FAISS vectorstore'''
    return FAISS(
        embedding_function=embedder,
        index=IndexFlatL2(embed_dims),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False
    )

def aggregate_vstores(vectorstores):
    ## 初始化一个空的 FAISS 索引并将其他索引合并到其中
    agg_vstore = default_FAISS()
    for vstore in vectorstores:
        agg_vstore.merge_from(vstore)
    return agg_vstore

if 'docstore' not in globals():
    docstore = aggregate_vstores(vecstores)

print(f"Constructed aggregate docstore with {len(docstore.docstore._dict)} chunks")

llm = ChatNVIDIA(model="ai-nemotron-4-340b-instruct") | StrOutputParser()
convstore = default_FAISS()

doc_names_string = "\n"
for doc_name in docs_name:
    doc_names_string += doc_name+"\n"
    
def save_memory_and_get_output(d, vstore):
    """Accepts 'input'/'output' dictionary and saves to convstore"""
    vstore.add_texts([
        f"User previously responded with {d.get('input')}",
        f"Agent previously responded with {d.get('output')}"
    ])
    return d.get('output')

initial_msg = (
    "Hello! I am a document chat agent here to help the user!"
    f" I have access to the following documents: {doc_names_string}\n\nHow can I help you?"
)

chat_prompt = ChatPromptTemplate.from_messages([("system",
    "You are a document chatbot. Help the user as they ask questions about documents."
    " User messaged just asked: {input}\n\n"
    " From this, we have retrieved the following potentially-useful info: "
    " Conversation History Retrieval:\n{history}\n\n"
    " Document Retrieval:\n{context}\n\n"
    " (Answer only from retrieval. Only cite sources that are used. Make your response conversational.Reply must more than 100 words)"
), ('user', '{input}')])

## Utility Runnables/Methods
def RPrint(preface=""):
    """Simple passthrough "prints, then returns" chain"""
    def print_and_return(x, preface):
        print(f"{preface}{x}")
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

def docs2str(docs, title="Document"):
    """Useful utility for making chunks into context string. Optional, but useful"""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name:
            out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str

## 将较长的文档重新排序到输出文本的中心， RunnableLambda在链中运行无参自定义函数 ，长上下文重排序（LongContextReorder）
long_reorder = RunnableLambda(LongContextReorder().transform_documents)

retrieval_chain = (
    {'input' : (lambda x: x)}
    | RunnableAssign({'history' : itemgetter('input') | convstore.as_retriever() | long_reorder | docs2str})
    | RunnableAssign({'context' : itemgetter('input') | docstore.as_retriever()  | long_reorder | docs2str})
    | RPrint()
)
stream_chain = chat_prompt | llm

def chat_gen(message, history=[], return_buffer=True):
    buffer = ""
    ##首先根据输入的消息进行检索
    retrieval = retrieval_chain.invoke(message)
    line_buffer = ""

    ## 然后流式传输stream_chain的结果
    for token in stream_chain.stream(retrieval):
        buffer += token
        ## 优化信息打印的格式
        if not return_buffer:
            line_buffer += token
            if "\n" in line_buffer:
                line_buffer = ""
            if ((len(line_buffer)>84 and token and token[0] == " ") or len(line_buffer)>100):
                line_buffer = ""
                yield "\n"
                token = "  " + token.lstrip()
        yield buffer if return_buffer else token

    ##最后将聊天内容保存到对话内存缓冲区中
    save_memory_and_get_output({'input':  message, 'output': buffer}, convstore)

chatbot = gr.Chatbot(value = [[None, initial_msg]])
demo = gr.ChatInterface(chat_gen, chatbot=chatbot).queue()

try:
    demo.launch(debug=True, share=False, show_api=False, server_port=8567, server_name="0.0.0.0")
    demo.close()
except Exception as e:
    demo.close()
    print(e)
    raise e