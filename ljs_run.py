from openai import OpenAI
import gradio as gr

# 设置NVIDIA API客户端
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# 聊天生成函数
def chat_gen(message):
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
def gradio_interface(message):
    response = chat_gen(message)
    return response

initial_msg = "Hello! I am a chat agent here to help you. How can I assist you today?"

chatbot = gr.Chatbot(value=[[None, initial_msg]])
demo = gr.Interface(fn=gradio_interface, inputs="text", outputs="text")

# 启动Gradio界面
if __name__ == "__main__":
    demo.launch(debug=True, share=False, show_api=False, server_port=8567, server_name="0.0.0.0")
