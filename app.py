import base64

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
from openai import OpenAI

import os

nvidia_api_key = "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"
assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
os.environ["NVIDIA_API_KEY"] = nvidia_api_key


llm = ChatNVIDIA(model="ai-nemotron-4-340b-instruct")
embedder = NVIDIAEmbeddings(model="ai-embed-qa-4")


# # 在这里我们读入文本数据并将它们准备到 vectorstore 中
# ps = os.listdir("data_cq/")
# data = []
# sources = []
# docs_name = []
# for p in ps:
#     if p.endswith('.txt'):
#         path2file="our_data/"+p
#         docs_name.append(path2file)
#         with open(path2file,encoding="utf-8") as f:
#             lines=f.readlines()
#             for line in lines:
#                 if len(line)>=1:
#                     data.append(line)
#                     sources.append(path2file)

# documents=[d for d in data if d != '\n']
# len(data), len(documents), data[0]

# 在这里我们读入文本数据并将它们准备到 vectorstore 中
ps_cq = os.listdir("data_cq/")
data_cq = []
sources_cq = []
docs_name_cq = []
for p in ps_cq:
    if p.endswith('.txt'):
        path2file="data_cq/"+p
        docs_name_cq.append(path2file)
        with open(path2file,encoding="utf-8") as f:
            lines=f.readlines()
            for line in lines:
                if len(line)>=1:
                    data_cq.append(line)
                    sources_cq.append(path2file)

documents_cq=[d for d in data_cq if d != '\n']
len(data_cq), len(documents_cq), data_cq[0]

# 在这里我们读入文本数据并将它们准备到 vectorstore 中
ps_gk = os.listdir("data_gk/")
data_gk = []
sources_gk = []
docs_name_gk = []
for p in ps_gk:
    if p.endswith('.txt'):
        path2file="data_gk/"+p
        docs_name_gk.append(path2file)
        with open(path2file,encoding="utf-8") as f:
            lines=f.readlines()
            for line in lines:
                if len(line)>=1:
                    data_gk.append(line)
                    sources_gk.append(path2file)

documents_gk=[d for d in data_gk if d != '\n']
len(data_gk), len(documents_gk), data_gk[0]

# 在这里我们读入文本数据并将它们准备到 vectorstore 中
ps_lp = os.listdir("data_lp/")
data_lp = []
sources_lp = []
docs_name_lp = []
for p in ps_lp:
    if p.endswith('.txt'):
        path2file="data_lp/"+p
        docs_name_lp.append(path2file)
        with open(path2file,encoding="utf-8") as f:
            lines=f.readlines()
            for line in lines:
                if len(line)>=1:
                    data_lp.append(line)
                    sources_lp.append(path2file)

documents_lp=[d for d in data_lp if d != '\n']
len(data_lp), len(documents_lp), data_lp[0]

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
docs_cq = []
metadatas_cq = []


# # 仅在重构embedding时需要重新运行
# for i, d in enumerate(documents_cq):
#     splits = text_splitter.split_text(d)
#     #print(len(splits))
#     docs_cq.extend(splits)
#     metadatas_cq.extend([{"source": sources_cq[i]}] * len(splits))
# ### 将创建好的embed存储到本地
# store_cq = FAISS.from_texts(docs_cq, embedder , metadatas=metadatas_cq)
# store_cq.save_local('./embed_cq')

print("successfully build embed_cq")

# create my own uuid
text_splitter = CharacterTextSplitter(chunk_size=400, separator=" ")
docs_gk = []
metadatas_gk = []

# # 仅在重构embedding时需要重新运行
# for i, d in enumerate(documents_gk):
#     splits = text_splitter.split_text(d)
#     #print(len(splits))
#     docs_gk.extend(splits)
#     metadatas_gk.extend([{"source": sources_gk[i]}] * len(splits))
# ### 将创建好的embed存储到本地
# store_gk = FAISS.from_texts(docs_gk, embedder , metadatas=metadatas_gk)
# store_gk.save_local('./embed_gk')

print("successfully build embed_gk")

# create my own uuid
text_splitter = CharacterTextSplitter(chunk_size=400, separator=" ")
docs_lp = []
metadatas_lp = []

# # 仅在重构embedding时需要重新运行
# for i, d in enumerate(documents_lp):
#     splits = text_splitter.split_text(d)
#     #print(len(splits))
#     docs_lp.extend(splits)
#     metadatas_lp.extend([{"source": sources_lp[i]}] * len(splits))
# ### 将创建好的embed存储到本地
# store_gk = FAISS.from_texts(docs_lp, embedder , metadatas=metadatas_lp)
# store_gk.save_local('./embed_lp')

print("successfully build embed_lp")

### 从本地读取已经创建好的embed
vecstores_cq = [FAISS.load_local(folder_path="embed_cq/", embeddings=embedder, allow_dangerous_deserialization=True)]
vecstores_gk = [FAISS.load_local(folder_path="embed_gk/", embeddings=embedder, allow_dangerous_deserialization=True)]
vecstores_lp = [FAISS.load_local(folder_path="embed_lp/", embeddings=embedder, allow_dangerous_deserialization=True)]

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

if 'docstore_cq' not in globals():
    docstore_cq = aggregate_vstores(vecstores_cq)

if 'docstore_gk' not in globals():
    docstore_gk = aggregate_vstores(vecstores_gk)

if 'docstore_lp' not in globals():
    docstore_lp = aggregate_vstores(vecstores_lp)

print(f"Constructed aggregate docstore with {len(docstore_cq.docstore._dict)} chunks")
print(f"Constructed aggregate docstore with {len(docstore_gk.docstore._dict)} chunks")
print(f"Constructed aggregate docstore with {len(docstore_lp.docstore._dict)} chunks")

llm = ChatNVIDIA(model="ai-nemotron-4-340b-instruct") | StrOutputParser()
convstore_cq = default_FAISS()
convstore_gk = default_FAISS()
convstore_lp = default_FAISS()

doc_names_string_cq = "\n"
for doc_name in docs_name_cq:
    doc_names_string_cq += doc_name+"\n"

doc_names_string_gk = "\n"
for doc_name in docs_name_gk:
    doc_names_string_gk += doc_name+"\n"

doc_names_string_lp = "\n"
for doc_name in docs_name_lp:
    doc_names_string_lp += doc_name+"\n"
    
def save_memory_and_get_output(d, vstore):
    """Accepts 'input'/'output' dictionary and saves to convstore"""
    vstore.add_texts([
        f"User previously responded with {d.get('input')}",
        f"Agent previously responded with {d.get('output')}"
    ])
    return d.get('output')

initial_msg_cq = (
    "Hello! I am a document chat agent here to help the user!"
    f" I have access to the following documents: {doc_names_string_cq}\n\nHow can I help you?"
)

initial_msg_gk = (
    "Hello! I am a document chat agent here to help the user!"
    f" I have access to the following documents: {doc_names_string_gk}\n\nHow can I help you?"
)

initial_msg_lp = (
    "Hello! I am a document chat agent here to help the user!"
    f" I have access to the following documents: {doc_names_string_lp}\n\nHow can I help you?"
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

retrieval_chain_cq = (
    {'input' : (lambda x: x)}
    | RunnableAssign({'history' : itemgetter('input') | convstore_cq.as_retriever() | long_reorder | docs2str})
    | RunnableAssign({'context' : itemgetter('input') | docstore_cq.as_retriever()  | long_reorder | docs2str})
    | RPrint()
)
retrieval_chain_gk = (
    {'input' : (lambda x: x)}
    | RunnableAssign({'history' : itemgetter('input') | convstore_gk.as_retriever() | long_reorder | docs2str})
    | RunnableAssign({'context' : itemgetter('input') | docstore_gk.as_retriever()  | long_reorder | docs2str})
    | RPrint()
)
retrieval_chain_lp = (
    {'input' : (lambda x: x)}
    | RunnableAssign({'history' : itemgetter('input') | convstore_lp.as_retriever() | long_reorder | docs2str})
    | RunnableAssign({'context' : itemgetter('input') | docstore_lp.as_retriever()  | long_reorder | docs2str})
    | RPrint()
)
stream_chain = chat_prompt | llm

def chat_gen_cq(message, history=[], return_buffer=True):
    buffer = ""
    ##首先根据输入的消息进行检索
    retrieval = retrieval_chain_cq.invoke(message)
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
    save_memory_and_get_output({'input':  message, 'output': buffer}, convstore_cq)

def chat_gen_gk(message, history=[], return_buffer=True):
    buffer = ""
    ##首先根据输入的消息进行检索
    retrieval = retrieval_chain_gk.invoke(message)
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
    save_memory_and_get_output({'input':  message, 'output': buffer}, convstore_gk)

def chat_gen_lp(message, history=[], return_buffer=True):
    buffer = ""
    ##首先根据输入的消息进行检索
    retrieval = retrieval_chain_lp.invoke(message)
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
    save_memory_and_get_output({'input':  message, 'output': buffer}, convstore_lp)

#李
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)


def chat_gen_code(message, history=[], return_buffer=True):
    completion = client.chat.completions.create(
        model="ibm/granite-34b-code-instruct",
        messages=[{"role": "user", "content": message}],
        temperature=0.5,
        top_p=1,
        max_tokens=1024,
        stream=True
    )

    buffer = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content
            buffer += token
            yield buffer if return_buffer else token

#陈
def generate_image_from_text(text, model="sdxl-turbo"):
    models = {
        "sdxl-turbo": {
            "url": "https://ai.api.nvidia.com/v1/genai/stabilityai/sdxl-turbo",
            "payload": {
                "text_prompts": [{"text": text}],
                "seed": 0,
                "sampler": "K_EULER_ANCESTRAL",
                "steps": 2
            }
        },
        "stable-diffusion-xl": {
            "url": "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-xl",
            "payload": {
                "text_prompts": [
                    {"text": text, "weight": 1},
                    {"text": "", "weight": -1}
                ],
                "cfg_scale": 5,
                "sampler": "K_DPM_2_ANCESTRAL",
                "seed": 0,
                "steps": 25
            }
        },
        "sdxl-lightning": {
            "url": "https://ai.api.nvidia.com/v1/genai/bytedance/sdxl-lightning",
            "payload": {
                "text_prompts": [{"text": text}],
                "seed": 0,
                "steps": 4
            }
        },
        "stable-diffusion-3-medium": {
            "url": "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium",
            "payload": {
                "prompt": text,
                "cfg_scale": 5,
                "aspect_ratio": "16:9",
                "seed": 0,
                "steps": 50,
                "negative_prompt": ""
            }
        }
    }

    if model not in models:
        raise ValueError(f"Unsupported model: {model}")

    headers = {
        "Authorization": "Bearer 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",
        "Accept": "application/json",
    }

    response = requests.post(models[model]["url"], headers=headers, json=models[model]["payload"])
    response.raise_for_status()
    response_body = response.json()

    # 解码Base64数据
    if model == "stable-diffusion-3-medium":
        image_data = base64.b64decode(response_body['image'])
    else:
        image_data = base64.b64decode(response_body['artifacts'][0]['base64'])

    # 将解码后的数据保存为图像文件
    image_filename = f'generated_image_{model}_{int(time.time())}.png'
    full_path = os.path.join('image_cache', image_filename)
    with open(full_path, 'wb') as image_file:
        image_file.write(image_data)

    print(f"图像已成功解码并保存为 {full_path}")
    return full_path

import os
import tempfile
import time
import gradio as gr
import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# 设置Gradio缓存目录
os.environ['GRADIO_TEMP_DIR'] = os.path.join(os.getcwd(), 'gradio_cache')
tempfile.tempdir = os.environ['GRADIO_TEMP_DIR']

# 确保缓存目录存在
os.makedirs(os.environ['GRADIO_TEMP_DIR'], exist_ok=True)

# 创建图片缓存目录
CACHE_DIR = "image_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# 自定义CSS样式
custom_css = """
.chat-box {
    height: 60vh !important;
    overflow-y: auto;
}
body, .gradio-container {
    width: 99% !important;
    max-width: 99% !important;
    padding: 5px !important;
    margin: 0 auto !important;
}
#fullscreen-iframe {
    width: 100%;
    height: 100vh;
    border: none;
}
#chat-interface {
    height: calc(100vh - 200px);
}
::-webkit-scrollbar {
    display: none;  
}
* {
    -ms-overflow-style: none;
    scrollbar-width: none;
}
.gradio-dropdown {
    transition: opacity 0.3s ease-in-out, max-height 0.3s ease-in-out;
    max-height: 0;
    opacity: 0;
    overflow: hidden;
}
.gradio-dropdown.visible {
    max-height: 50px;
    opacity: 1;
}
"""

# 设置默认图片
DEFAULT_IMAGE_URL = "https://gradio-static-files.s3.us-west-2.amazonaws.com/header-image.jpg"
DEFAULT_IMAGE_PATH = os.path.join(CACHE_DIR, "default_image.jpg")
user_avatar = "image/user.png"

# 下载并保存默认图片
if not os.path.exists(DEFAULT_IMAGE_PATH):
    response = requests.get(DEFAULT_IMAGE_URL)
    with open(DEFAULT_IMAGE_PATH, "wb") as f:
        f.write(response.content)

# 定义聊天配置类
class ChatConfig:
    def __init__(self):
        self.temperature = 0.5
        self.model = "重庆旅游指南"
        self.drop = "sdxl-turbo"
        self.image = "image/重庆夜景.jpg"
        _, _, self.ai_avatar = update_mode_image(self.model)

# 定义聊天状态类
class ChatState:
    def __init__(self):
        self.histories = [[]]
        self.configs = [ChatConfig()]
        self.chat_names = ["新对话"]
        self.current_index = 0
        self.max_history = 5
        self.user_avatar = "image/user.png"

    def current_chat(self):
        return self.histories[self.current_index]

    def current_config(self):
        return self.configs[self.current_index]

    def update_current_chat(self, new_history):
        self.histories[self.current_index] = new_history
        print(f"Updated chat history for index {self.current_index}")

    def update_current_config(self, temperature, model, drop, image):
        config = self.configs[self.current_index]
        config.temperature = temperature
        config.model = model
        config.drop = drop
        config.image = image
        print(f"Updated config for index {self.current_index}: temp={temperature}, model={model}, drop={drop}")

    def update_current_model(self, model):
        if len(self.current_chat()) == 0:  # 只有在对话为空时才更新AI头像
            self.configs[self.current_index].model = model
            _, _, ai_avatar = update_mode_image(model)
            self.configs[self.current_index].ai_avatar = ai_avatar
        print(f"Updated model for chat {self.current_index}: {model}")

    def new_chat(self):
        if len(self.histories) >= self.max_history:
            self.histories.pop(0)
            self.configs.pop(0)
            self.chat_names.pop(0)
        else:
            self.current_index = len(self.histories)
        new_config = ChatConfig()
        _, _, new_ai_avatar = update_mode_image(new_config.model)
        new_config.ai_avatar = new_ai_avatar
        self.histories.append([])
        self.configs.append(new_config)
        self.chat_names.append("新对话")
        while len(self.chat_names) < self.max_history:
            self.chat_names.append("新对话")
        print(f"Created new chat at index {self.current_index}")

    def load_chat(self, index):
        if 0 <= index < len(self.histories):
            self.current_index = index
            config = self.configs[index]
            if not config.ai_avatar:
                _, _, config.ai_avatar = update_mode_image(config.model)
            print(f"Loaded chat at index {index}")
            return self.histories[index], config
        print(f"Failed to load chat at index {index}")
        return [], ChatConfig()

    def get_current_state(self):
        config = self.current_config()
        return (
            self.current_chat(),
            config.temperature,
            config.model,
            config.drop,
            config.image
        )

    def update_chat_name(self, new_name):
        self.chat_names[self.current_index] = new_name
        print(f"Updated chat name for index {self.current_index}: {new_name}")

    def get_current_avatars(self):
        return (self.user_avatar, self.configs[self.current_index].ai_avatar)

# 创建iframe HTML
def create_iframe_html():
    return """
    <iframe src="/app.html" style="width:100%; height:100vh; border:none;"></iframe>
    """

def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def update_drop_visibility(model):
    return gr.update(visible=(model == "文生图"), label="选择图像生成模型" if model == "文生图" else "选择")

def update_interface_for_model(model):
    if model == "文生图":
        return gr.update(visible=True, label="选择图像生成模型")
    else:
        return gr.update(visible=False)
# 更新模式图片和标签
def update_mode_image(model):
    if model == "高考志愿填报":
        return "image/高考志愿.jpg", "高考志愿助手", "image/student.png"
    elif model == "重庆旅游指南":
        return "image/来福士2.jpg", "重庆旅游助手", "image/travel.png"
    elif model == "笔记本选购":
        return "image/笔记本5.jpg", "电脑推荐助手", "image/laptop.png"
    elif model == "代码助手":
        return "image/代码.jpg", "代码助手", "image/code.png"
    elif model == "文生图":
        return "image/文生图.jpg", "图像生成助手", "image/multimodal.png"
    else:
        return "image/重庆夜景.jpg", "重庆旅游助手", "image/travel.png"

# 聊天函数
def chat(message, history, state):
    if not message:
        return {"text": "", "files": []}, history, state

    current_config = state.current_config()
    model = current_config.model
    print(f"Current model in chat function: {model}")

    text_input = message.get("text", "")
    file_paths = message.get("files", [])

    user_message = text_input if not file_paths else [text_input] + file_paths

    if len(history) == 0:
        new_chat_name = text_input[:15] + "..." if len(text_input) > 15 else text_input
        state.update_chat_name(new_chat_name)
        _, _, ai_avatar = update_mode_image(model)
        state.configs[state.current_index].ai_avatar = ai_avatar

    # 显示思考动画
    thinking_dots = [".", "..", "..."]
    for _ in range(6):
        for dots in thinking_dots:
            yield {"text": "", "files": []}, history + [(user_message, f"思考中{dots}")], state

    if model == "重庆旅游指南":
        gen_func = chat_gen_cq
    elif model == "高考志愿填报":
        gen_func = chat_gen_gk
    elif model == "笔记本选购":
        gen_func = chat_gen_lp
    elif model == "代码助手":
        gen_func = chat_gen_code
    elif model == "文生图":
        yield {"text": "", "files": []}, history + [(user_message, "正在生成图片，请稍候...")], state
        image_path = generate_image_from_text(text_input, current_config.drop)
        image_base64 = get_image_base64(image_path)
        full_response = f"根据您的描述，我使用{current_config.drop}模型生成了以下图片：<img src='data:image/png;base64,{image_base64}'/>"
        new_history = history + [(user_message, full_response)]
        yield {"text": "", "files": []}, new_history, state
        state.update_current_chat(new_history)
        return
    else:
        print(f"Unknown model: {model}")
        gen_func = chat_gen_cq  # 默认使用重庆旅游指南

    full_response = ""
    last_response = ""
    for chunk in gen_func(text_input, history):
        if chunk != last_response:
            full_response = chunk
            yield {"text": "", "files": []}, history + [(user_message, full_response)], state
            last_response = chunk

    state.update_current_chat(history + [(user_message, full_response)])
    print(f"Chat completed. Updated state for model: {model}")
    return {"text": "", "files": []}, history + [(user_message, full_response)], state

def retry(history, state):
    if len(history) > 0:
        last_user_message = history[-1][0]
        new_history = history[:-1]
        if isinstance(last_user_message, list):
            return chat({"text": last_user_message[0], "files": last_user_message[1:]}, new_history, state)
        else:
            return chat({"text": last_user_message, "files": []}, new_history, state)
    return {"text": "", "files": []}, history, state

def undo(history, state):
    if len(history) > 0:
        new_history = history[:-1]
        state.update_current_chat(new_history)
        return {"text": "", "files": []}, new_history, state
    return {"text": "", "files": []}, history, state

def clear():
    new_state = ChatState()
    return {"text": "", "files": []}, [], new_state

def new_chat(state):
    state.new_chat()
    current_config = state.current_config()
    return (
        state,
        [],
        current_config.temperature,
        current_config.model,
        current_config.drop,
        current_config.image,
        gr.update(avatar_images=state.get_current_avatars()),
        *update_history_buttons(state)
    )

def load_chat(state, index):
    history, config = state.load_chat(index)
    print(f"Loading chat {index}")
    print(f"Config: temp={config.temperature}, model={config.model}, drop={config.drop}")

    if not config.ai_avatar:
        _, _, config.ai_avatar = update_mode_image(config.model)

    return (
        state,
        history,
        config.temperature,
        config.model,
        config.drop,
        config.image,
        gr.update(avatar_images=state.get_current_avatars()),
        *update_history_buttons(state)
    )

def update_history_buttons(state):
    while len(state.chat_names) < state.max_history:
        state.chat_names.append("新对话")
    return [gr.update(visible=(i < len(state.histories)), value=state.chat_names[i]) for i in range(state.max_history)]

def update_config(state, model, temperature, drop, image):
    config = state.current_config()
    config.model = model
    config.temperature = temperature
    if model == "文生图":
        config.drop = drop
    config.image = image
    state.update_current_model(model)
    print(f"Config updated: model={model}, temp={temperature}, drop={drop}")
    return state, state.current_chat()

def update_interface(image, label, ai_avatar, chatbot, state):
    current_config = state.current_config()
    if not current_config.ai_avatar:
        _, label, current_config.ai_avatar = update_mode_image(current_config.model)
    current_avatars = state.get_current_avatars()
    return (
        image,
        gr.update(label=label),
        state,
        gr.update(avatar_images=current_avatars),
        state.current_chat()
    )

def update_interface_for_model(model):
    if model == "文生图":
        return gr.update(visible=True, label="选择图像生成模型")
    else:
        return gr.update(visible=False)

def initialize_chatbot_label(state):
    initial_config = state.current_config()
    _, label, ai_avatar = update_mode_image(initial_config.model)
    state.configs[state.current_index].ai_avatar = ai_avatar
    return gr.update(label=label), gr.update(avatar_images=state.get_current_avatars()), []

# 创建Gradio界面
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="CyberChatterAI", analytics_enabled=False) as demo:
    initial_state = ChatState()
    state = gr.State(initial_state)

    gr.HTML(create_iframe_html())

    with gr.Row(equal_height=True):
        with gr.Column(scale=10):
            gr.Markdown("# CyberChatter AI助手")
        with gr.Column(scale=2):
            toggle_dark = gr.Button(value="日间/夜间模式")

    toggle_dark.click(
        fn=None,
        inputs=None,
        outputs=None,
        js="() => { document.body.classList.toggle('dark'); }"
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            with gr.Tab("历史记录"):
                history_buttons = [gr.Button("新对话", visible=False) for _ in range(5)]
                new_chat_btn = gr.Button("新建对话", variant="primary")
        with gr.Column(scale=11):
            gr.Markdown("## 欢迎回来，用户USER")
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="AI助手",
                        elem_classes="chat-box",
                        show_copy_button=True,
                        avatar_images=initial_state.get_current_avatars(),
                        bubble_full_width=False,
                        sanitize_html=False
                    )
                    msg = gr.MultimodalTextbox(placeholder="在这里输入文字或上传图片...", lines=1, max_lines=10, show_label=False)
                    with gr.Row():
                        submit = gr.Button(value="发送", variant="primary")
                        retry_btn = gr.Button(value="重试")
                        undo_btn = gr.Button(value="撤销")
                        clear_btn = gr.Button(value="清除")
                with gr.Column(scale=1):
                    with gr.Column(scale=1):
                        model = gr.Radio(
                            ["重庆旅游指南", "高考志愿填报", "笔记本选购", "代码助手", "文生图"],
                            label="选择模型",
                            info="选择你想要的对话的模型",
                            value=initial_state.current_config().model
                        )
                        temperature = gr.Slider(
                            label="Temperature",
                            value=initial_state.current_config().temperature,
                            minimum=0, maximum=1,
                            visible=True
                        )
                        drop = gr.Dropdown(
                            ["sdxl-turbo", "stable-diffusion-xl", "sdxl-lightning", "stable-diffusion-3-medium"],
                            label="选择图像生成模型",
                            value=initial_state.current_config().drop,
                            visible=False
                        )
                    mode_img = gr.Image(label="模式图片", value=initial_state.current_config().image,
                                        elem_id="mode-image", type="filepath")

            msg.submit(chat, inputs=[msg, chatbot, state], outputs=[msg, chatbot, state])
            submit.click(chat, inputs=[msg, chatbot, state], outputs=[msg, chatbot, state])
            retry_btn.click(retry, inputs=[chatbot, state], outputs=[msg, chatbot, state])
            undo_btn.click(undo, inputs=[chatbot, state], outputs=[msg, chatbot, state])
            clear_btn.click(fn=clear, outputs=[msg, chatbot, state])

            temp_label = gr.Textbox(visible=False)
            model.change(
                update_mode_image,
                inputs=[model],
                outputs=[mode_img, temp_label, gr.State()]
            ).then(
                update_interface,
                inputs=[mode_img, temp_label, gr.State(), chatbot, state],
                outputs=[mode_img, chatbot, state, chatbot, chatbot]
            ).then(
                update_config,
                inputs=[state, model, temperature, drop, mode_img],
                outputs=[state, chatbot]
            ).then(
                update_interface_for_model,
                inputs=[model],
                outputs=[drop]
            )

            for i, btn in enumerate(history_buttons):
                btn.click(
                    load_chat,
                    inputs=[state, gr.State(i)],
                    outputs=[state, chatbot, temperature, model, drop, mode_img, chatbot, *history_buttons]
                )

            new_chat_btn.click(
                new_chat,
                inputs=[state],
                outputs=[state, chatbot, temperature, model, drop, mode_img, chatbot, *history_buttons]
            ).then(
                lambda state: gr.update(avatar_images=state.get_current_avatars()),
                inputs=[state],
                outputs=[chatbot]
            )

            temperature.change(update_config, inputs=[state, model, temperature, drop, mode_img],
                               outputs=[state, chatbot])
            model.change(update_config, inputs=[state, model, temperature, drop, mode_img], outputs=[state, chatbot])
            drop.change(update_config, inputs=[state, model, temperature, drop, mode_img], outputs=[state, chatbot])
            mode_img.change(update_config, inputs=[state, model, temperature, drop, mode_img], outputs=[state, chatbot])

            demo.load(
                fn=None,
                inputs=None,
                outputs=None,
                js="""
                    function() {
                        const isDarkMode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
                        if (isDarkMode && document.body.classList.contains('dark')) {
                            document.body.classList.toggle('dark');
                        }
                        window.scrollTo(0, 0);
                    }
                """
            )

            demo.load(
                initialize_chatbot_label,
                inputs=[state],
                outputs=[chatbot, chatbot, chatbot]
            )

            demo.load(update_history_buttons, inputs=[state], outputs=history_buttons)

    app = FastAPI()

    app.mount("/static", StaticFiles(directory="static"), name="static")
    app.mount("/image_cache", StaticFiles(directory="image_cache"), name="image_cache")
    app.mount("/image", StaticFiles(directory="image"), name="image")

    @app.get("/app.html")
    async def serve_scroll_demo():
        return FileResponse("app.html")

    app = gr.mount_gradio_app(app, demo, path="/")

    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=7879, ssl_keyfile=None, ssl_certfile=None)