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
        path2file = "data_cq/" + p
        docs_name_cq.append(path2file)
        with open(path2file, encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                if len(line) >= 1:
                    data_cq.append(line)
                    sources_cq.append(path2file)

documents_cq = [d for d in data_cq if d != '\n']
len(data_cq), len(documents_cq), data_cq[0]

# 在这里我们读入文本数据并将它们准备到 vectorstore 中
ps_gk = os.listdir("data_gk/")
data_gk = []
sources_gk = []
docs_name_gk = []
for p in ps_gk:
    if p.endswith('.txt'):
        path2file = "data_gk/" + p
        docs_name_gk.append(path2file)
        with open(path2file, encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                if len(line) >= 1:
                    data_gk.append(line)
                    sources_gk.append(path2file)

documents_gk = [d for d in data_gk if d != '\n']
len(data_gk), len(documents_gk), data_gk[0]

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

### 从本地读取已经创建好的embed
vecstores_cq = [FAISS.load_local(folder_path="embed_cq/", embeddings=embedder, allow_dangerous_deserialization=True)]
vecstores_gk = [FAISS.load_local(folder_path="embed_gk/", embeddings=embedder, allow_dangerous_deserialization=True)]

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

print(f"Constructed aggregate docstore with {len(docstore_cq.docstore._dict)} chunks")
print(f"Constructed aggregate docstore with {len(docstore_gk.docstore._dict)} chunks")

llm = ChatNVIDIA(model="ai-nemotron-4-340b-instruct") | StrOutputParser()
convstore_cq = default_FAISS()
convstore_gk = default_FAISS()

doc_names_string_cq = "\n"
for doc_name in docs_name_cq:
    doc_names_string_cq += doc_name + "\n"

doc_names_string_gk = "\n"
for doc_name in docs_name_gk:
    doc_names_string_gk += doc_name + "\n"


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
        {'input': (lambda x: x)}
        | RunnableAssign({'history': itemgetter('input') | convstore_cq.as_retriever() | long_reorder | docs2str})
        | RunnableAssign({'context': itemgetter('input') | docstore_cq.as_retriever() | long_reorder | docs2str})
        | RPrint()
)
retrieval_chain_gk = (
        {'input': (lambda x: x)}
        | RunnableAssign({'history': itemgetter('input') | convstore_gk.as_retriever() | long_reorder | docs2str})
        | RunnableAssign({'context': itemgetter('input') | docstore_gk.as_retriever() | long_reorder | docs2str})
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
            if ((len(line_buffer) > 84 and token and token[0] == " ") or len(line_buffer) > 100):
                line_buffer = ""
                yield "\n"
                token = "  " + token.lstrip()
        yield buffer if return_buffer else token

    ##最后将聊天内容保存到对话内存缓冲区中
    save_memory_and_get_output({'input': message, 'output': buffer}, convstore_cq)


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
            if ((len(line_buffer) > 84 and token and token[0] == " ") or len(line_buffer) > 100):
                line_buffer = ""
                yield "\n"
                token = "  " + token.lstrip()
        yield buffer if return_buffer else token

    ##最后将聊天内容保存到对话内存缓冲区中
    save_memory_and_get_output({'input': message, 'output': buffer}, convstore_gk)

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
    over
::-webkit-scrollbar {
    display: none;  
}
* {
    -ms-overflow-style: none;
    scrollbar-width: none;
}
"""

# 设置默认图片
DEFAULT_IMAGE_URL = "https://gradio-static-files.s3.us-west-2.amazonaws.com/header-image.jpg"
DEFAULT_IMAGE_PATH = os.path.join(CACHE_DIR, "default_image.jpg")

# 下载并保存默认图片
if not os.path.exists(DEFAULT_IMAGE_PATH):
    response = requests.get(DEFAULT_IMAGE_URL)
    with open(DEFAULT_IMAGE_PATH, "wb") as f:
        f.write(response.content)


# 定义聊天配置类
class ChatConfig:
    def __init__(self):
        self.temperature = 50
        self.model = "重庆旅游指南"
        self.drop = "选项1"
        self.check = False
        self.image = "image/重庆夜景.jpg"


# 定义聊天状态类
class ChatState:
    def __init__(self):
        self.histories = [[]]
        self.configs = [ChatConfig()]
        self.chat_names = ["新对话"] * 5  # 初始化为5个"新对话"
        self.current_index = 0
        self.max_history = 5

    def current_chat(self):
        return self.histories[self.current_index]

    def current_config(self):
        return self.configs[self.current_index]

    def update_current_chat(self, new_history):
        self.histories[self.current_index] = new_history
        print(f"Updated chat history for index {self.current_index}")

    def update_current_config(self, temperature, model, drop, check, image):
        config = self.configs[self.current_index]
        config.temperature = temperature
        config.model = model
        config.drop = drop
        config.check = check
        config.image = image
        print(
            f"Updated config for index {self.current_index}: temp={temperature}, model={model}, drop={drop}, check={check}")

    def new_chat(self):
        if len(self.histories) >= self.max_history:
            self.histories.pop(0)
            self.configs.pop(0)
            self.chat_names.pop(0)
        else:
            self.current_index = len(self.histories)
        self.histories.append([])
        self.configs.append(ChatConfig())
        self.chat_names.append("新对话")
        # 确保 chat_names 列表长度始终为 max_history
        while len(self.chat_names) < self.max_history:
            self.chat_names.append("新对话")
        print(f"Created new chat at index {self.current_index}")

    def load_chat(self, index):
        if 0 <= index < len(self.histories):
            self.current_index = index
            print(f"Loaded chat at index {index}")
            return self.histories[index], self.configs[index]
        print(f"Failed to load chat at index {index}")
        return [], ChatConfig()

    def get_current_state(self):
        config = self.current_config()
        return (
            self.current_chat(),
            config.temperature,
            config.model,
            config.drop,
            config.check,
            config.image
        )

    def update_chat_name(self, new_name):
        self.chat_names[self.current_index] = new_name
        print(f"Updated chat name for index {self.current_index}: {new_name}")


# 创建iframe HTML
def create_iframe_html():
    return """
    <iframe src="/app.html" style="width:100%; height:100vh; border:none;"></iframe>
    """


class CustomChatInterface(gr.ChatInterface):
    def __init__(self, fn, state, initial_label="重庆旅游助手", **kwargs):
        self.chatbot = gr.Chatbot(label=initial_label, elem_classes="chat-box", show_copy_button=True)
        self.fn = fn
        self.state = state
        self.initial_label = initial_label
        self.kwargs = kwargs
        self.submit_btn = gr.Button(value="发送", variant="primary")
        self.retry_btn = gr.Button(value="重试")
        self.undo_btn = gr.Button(value="撤销")
        self.clear_btn = gr.Button(value="清除")
        self.textbox = gr.Textbox(placeholder="在这里输入...", lines=2, max_lines=10, show_label=False)

    def render(self):
        with gr.Group():
            chatbot = self.chatbot
            msg = self.textbox
            with gr.Row():
                submit = self.submit_btn
                retry = self.retry_btn
                undo = self.undo_btn
                clear = self.clear_btn

        super().__init__(
            fn=self.fn,
            chatbot=chatbot,
            textbox=msg,
            submit_btn=submit,
            retry_btn=retry,
            undo_btn=undo,
            clear_btn=clear,
            **self.kwargs
        )

        return self


# 修改聊天函数以适应新的结构
def chat(message, history, state):
    if not message:
        return "", history, state

    # 如果是第一次对话，更新对话名称
    if len(history) == 0:
        new_chat_name = message[:15] + "..." if len(message) > 15 else message
        state.update_chat_name(new_chat_name)

    # 显示思考动画
    thinking_dots = [".", "..", "..."]
    for _ in range(5):
        for dots in thinking_dots:
            yield "", history + [(message, f"思考中{dots}")], state
            time.sleep(0.3)

    # 使用适当的 chat_gen 函数
    model = state.current_config().model
    gen_func = chat_gen_cq if model == "重庆旅游指南" else chat_gen_gk

    full_response = ""
    for chunk in gen_func(message, history):
        if isinstance(chunk, str) and chunk != full_response:
            full_response = chunk
            yield "", history + [(message, full_response)], state

    state.update_current_chat(history + [(message, full_response)])
    print(f"Chat completed. Updated state for model: {model}")
    return "", history + [(message, full_response)], state

def retry(history, state):
    if len(history) > 0:
        last_user_message = history[-1][0]
        new_history = history[:-1]
        return chat(last_user_message, new_history, state)
    return "", history, state

def undo(history, state):
    if len(history) > 0:
        new_history = history[:-1]
        state.update_current_chat(new_history)
        return "", new_history, state
    return "", history, state


# 新建聊天
def new_chat(state):
    state.new_chat()
    current_config = state.current_config()
    return (
        state,
        [],  # 新对话的空历史记录
        current_config.temperature,
        current_config.model,
        current_config.drop,
        current_config.check,
        current_config.image
    )


# 加载聊天
def load_chat(state, index):
    history, config = state.load_chat(index)
    print(f"Loading chat {index}")
    print(f"Config: temp={config.temperature}, model={config.model}, drop={config.drop}, check={config.check}")
    return (
        state,
        history,  # 这里返回加载的历史记录
        config.temperature,
        config.model,
        config.drop,
        config.check,
        config.image
    )


# 更新历史按钮
def update_history_buttons(state):
    # 确保 chat_names 列表长度至少为 max_history
    while len(state.chat_names) < state.max_history:
        state.chat_names.append("新对话")

    return [gr.update(visible=(i < len(state.histories)), value=state.chat_names[i]) for i in range(state.max_history)]


# 更新模式图片和标签
def update_mode_image(model):
    if model == "高考志愿填报":
        return "image/高考志愿.jpg", "高考志愿助手"
    elif model == "重庆旅游指南":
        return "image/来福士2.jpg", "重庆旅游助手"
    elif model == "笔记本选购":
        return "image/笔记本5.jpg", "电脑推荐助手"
    else:
        return "image/重庆夜景.jpg", "重庆旅游助手"


# 更新配置
def update_config(state, temperature, model, drop, check, image):
    state.update_current_config(temperature, model, drop, check, image)
    return state


# 更新界面函数
def update_interface(state, history, temperature, model, drop, check, image):
    current_config = state.current_config()
    current_config.temperature = temperature
    current_config.model = model
    current_config.drop = drop
    current_config.check = check
    current_config.image = image
    return state, history, temperature, model, drop, check, image


# 更新chatbot标签
def update_chatbot_label(image, label, chatbot):
    return gr.update(value=image), gr.update(label=label), gr.update(label=label)


def initialize_chatbot_label(state):
    initial_config = state.current_config()
    _, label = update_mode_image(initial_config.model)
    return gr.update(label=label)
initial_state = ChatState()
state = gr.State(initial_state)
chat_interface = CustomChatInterface(fn=chat, state=state, initial_label="重庆旅游助手")
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

    # 切换深色/浅色模式
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
                    chatbot = gr.Chatbot(label="重庆旅游助手", elem_classes="chat-box", show_copy_button=True)
                    msg = gr.Textbox(placeholder="在这里输入...", lines=2, max_lines=10, show_label=False)
                    with gr.Row():
                        submit = gr.Button(value="发送", variant="primary")
                        retry_btn = gr.Button(value="重试")
                        undo_btn = gr.Button(value="撤销")
                        clear = gr.Button(value="清除")
                with gr.Column(scale=1):
                    with gr.Column(scale=1):
                        temperature = gr.Slider(label="Temperature", value=initial_state.current_config().temperature,
                                                minimum=0, maximum=100)
                        model = gr.Radio(
                            ["重庆旅游指南", "高考志愿填报", "笔记本选购" ,"文生图"],
                            label="选择模型",
                            info="选择你想要的对话的模型",
                            value=initial_state.current_config().model
                        )
                        drop = gr.Dropdown(["选项1", "选项2", "选项3"], label="选择",
                                           value=initial_state.current_config().drop)
                        check = gr.Checkbox(label="是否XX", value=initial_state.current_config().check)
                    mode_img = gr.Image(label="模式图片", value=initial_state.current_config().image,
                                        elem_id="mode-image", height=660, type="filepath")

    # 设置聊天功能
    msg.submit(chat, inputs=[msg, chatbot, state], outputs=[msg, chatbot, state])
    submit.click(chat, inputs=[msg, chatbot, state], outputs=[msg, chatbot, state])
    retry_btn.click(retry, inputs=[chatbot, state], outputs=[msg, chatbot, state])
    undo_btn.click(undo, inputs=[chatbot, state], outputs=[msg, chatbot, state])
    clear.click(lambda: (None, [], ChatState()), outputs=[msg, chatbot, state])

    # 更新模式图片和聊天机器人标签
    temp_label = gr.Textbox(visible=False)
    model.change(
        update_mode_image,
        inputs=[model],
        outputs=[mode_img, temp_label]
    ).then(
        update_chatbot_label,
        inputs=[mode_img, temp_label, chatbot],
        outputs=[mode_img, chatbot, chatbot]
    )

    # 设置历史记录按钮点击事件
    for i, btn in enumerate(history_buttons):
        btn.click(
            load_chat,
            inputs=[state, gr.State(i)],
            outputs=[state, chatbot, temperature, model, drop, check, mode_img]
        ).then(
            update_interface,
            inputs=[state, chatbot, temperature, model, drop, check, mode_img],
            outputs=[state, chatbot, temperature, model, drop, check, mode_img]
        ).then(
            update_history_buttons,
            inputs=[state],
            outputs=history_buttons
        )

    # 设置新建对话按钮点击事件
    new_chat_btn.click(new_chat, inputs=[state],
                       outputs=[state, chatbot, temperature, model, drop, check, mode_img]).then(
        update_history_buttons, inputs=[state], outputs=history_buttons
    )

    # 更新配置事件
    temperature.change(update_config, inputs=[state, temperature, model, drop, check, mode_img], outputs=[state])
    model.change(update_config, inputs=[state, temperature, model, drop, check, mode_img], outputs=[state])
    drop.change(update_config, inputs=[state, temperature, model, drop, check, mode_img], outputs=[state])
    check.change(update_config, inputs=[state, temperature, model, drop, check, mode_img], outputs=[state])
    mode_img.change(update_config, inputs=[state, temperature, model, drop, check, mode_img], outputs=[state])

    # 页面加载时更新历史按钮
    demo.load(update_history_buttons, inputs=[state], outputs=history_buttons)

    # 页面加载时检查并设置主题
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
        outputs=[chatbot]
    )

# 创建FastAPI应用
app = FastAPI()

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/image_cache", StaticFiles(directory="image_cache"), name="image_cache")


@app.get("/app.html")
async def serve_scroll_demo():
    return FileResponse("app.html")

# 将Gradio应用挂载到FastAPI应用
app = gr.mount_gradio_app(app, demo, path="/")

# 启动服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7878)