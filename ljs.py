from openai import OpenAI
import gradio as gr
import os

# 设置NVIDIA API客户端
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# 聊天生成函数
def chat_gen_code(message, history):
    completion = client.chat.completions.create(
        model="ibm/granite-34b-code-instruct",
        messages=[{"role": "user", "content": message}],
        temperature=0.5,
        top_p=1,
        max_tokens=1024,
        stream=True
    )
    
    response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content
    
    return response

# 创建Gradio界面
initial_msg_code = "Hello! I am a code chat agent here to help you with coding questions. How can I assist you today?"

chatbot_code = gr.Chatbot(value=[[None, initial_msg_code]])

# 使用 Gradio 的 ChatInterface 启动聊天机器人
chat_interface_code = gr.ChatInterface(fn=chat_gen_code, chatbot=chatbot_code)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            chat_interface_code.render()

if __name__ == "__main__":
    try:
        demo.launch(debug=True, share=False, show_api=False, server_port=8666, server_name="0.0.0.0")
    except Exception as e:
        print(e)
