import gradio as gr
from typing import TYPE_CHECKING, Dict, Optional, Tuple
import json
import random

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

task_list = load_json("./task_info_full_13000.json")
# selected_topic = ["seed_good"]
# task_list = [item for item in task_list if item["topic"] in selected_topic]


if TYPE_CHECKING:
    from gradio.blocks import Block
    from gradio.components import Component
    from llmtuner.webui.engine import Engine

def rand_task():
    task = random.choice(task_list)
    return task["context"], task["task"]

js_code = '''
function jump_to_query(){

}
'''
def create_chat_box(
    engine: "Engine",
    visible: Optional[bool] = False
) -> Tuple["Block", "Component", "Component", Dict[str, "Component"]]:
    # add the title
    gr.HTML("<h1>PersuGPT (ACL 2024)</h1>")
    gr.HTML("<p>PersuGPT is a dialogue model developed for persuasion tasks, which is beneficial to marketing, negotiation, debate, psychotherapy, education, law and many other fields.</p>")
    with gr.Box(visible=visible, js = '''window.scrollTo(0, document.documentElement.scrollHeight-document.documentElement.clientHeight);''') as chat_box:
        with gr.Row():
            with gr.Column(scale=1):
                # persuader = gr.Textbox(show_label=True)
                gr.HTML("<h3>Persuasion Task</h3>")
                context = gr.Textbox(show_label=True, lines=2)
                Task = gr.Textbox(show_label=True, lines=2)
                start_btn = gr.Button(variant="primary")
                rand_btn = gr.Button(variant="primary")
                clear_btn = gr.Button()
                gen_kwargs = engine.chatter.generating_args
                max_new_tokens = gr.Slider(10, 2048, value=gen_kwargs.max_new_tokens, step=1, visible=False)
                top_p = gr.Slider(0.01, 1, value=gen_kwargs.top_p, step=0.01, visible=False)
                temperature = gr.Slider(0.01, 1.5, value=gen_kwargs.temperature, step=0.01, visible=False)
                repetition_penalty = gr.Slider(1.0, 1.5, value=gen_kwargs.repetition_penalty, step=0.01, visible=False)
                gr.HTML("<script>window.scrollTo(0, document.documentElement.scrollHeight-document.documentElement.clientHeight);</script>")
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=400)
                history = gr.State([])
                query = gr.Textbox(show_label=False, lines=3)
                submit_btn = gr.Button(variant="primary", elem_id = "query")
    # gr.HTML('''<center><a href="https://beian.miit.gov.cn/" target="_blank">冀ICP备2023002163号</a></center>''')
    submit_btn.click(
        engine.chatter.predict,
        [chatbot, query, history, max_new_tokens, top_p, temperature, repetition_penalty],
        [chatbot, history],
        show_progress=True,
    ).then(
        lambda: gr.update(value=""), outputs=[query]
    )
    start_btn.click(
        engine.chatter.start,
        [chatbot, history, context, Task, max_new_tokens, top_p, temperature, repetition_penalty],
        [chatbot, history],
        show_progress=True
    ).then(
        lambda: gr.update(value=""), outputs=[query]
    )

    rand_btn.click(lambda: rand_task(), outputs=[ context, Task], show_progress=True)

    clear_btn.click(lambda: ([], [], "", "", ""), outputs=[chatbot, history, context, Task], show_progress=True)

    return chat_box, chatbot, history, dict(
        # persuader=persuader,
        context=context,
        Task=Task,
        query=query,
        submit_btn=submit_btn,
        start_btn=start_btn,
        rand_btn=rand_btn,
        clear_btn=clear_btn,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty
    )
