import gradio as gr

# # 定义一个函数，用于增加计数
# def increment(state):
#     return state + 1, state + 1

# # 定义一个 Gradio 接口
# with gr.Blocks() as demo:
#     # 显示按钮点击次数的文本框
#     count_text = gr.Textbox(label="按钮点击次数", value="0")
#     # 计数按钮
#     increment_button = gr.Button("增加计数")
#     # 共享状态，用于存储计数值
#     state = gr.State(value=0)
    
#     # 当按钮被点击时，调用 increment 函数
#     increment_button.click(
#         fn=increment, 
#         inputs=state, 
#         outputs=[state, count_text] # 返回的值会更新共享状态和count_text
#     )


# 定义一个函数，用于添加消息到聊天记录
def add_message(message, chat_history):
    chat_history.append(message)
    return chat_history, chat_history

# 定义一个 Gradio 接口
with gr.Blocks() as demo:
    # 输入消息的文本框
    message_input = gr.Textbox(label="输入消息")
    # 显示聊天记录的文本框
    chat_display = gr.Textbox(label="聊天记录", interactive=False)
    # 发送按钮
    send_button = gr.Button("发送")
    # 共享状态，用于存储聊天记录
    chat_history = gr.State(value=[])

    # 当发送按钮被点击时，调用 add_message 函数
    send_button.click(
        fn=add_message, 
        inputs=[message_input, chat_history], 
        outputs=[chat_history, chat_display]
    )


# 启动 Gradio 接口
try:
    demo.launch(debug=True, share=False, show_api=False, server_port=8777, server_name="0.0.0.0")
    demo.close()
except Exception as e:
    demo.close()
    print(e)
    raise e
