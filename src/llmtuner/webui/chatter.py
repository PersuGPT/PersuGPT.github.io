import gradio as gr
from gradio.components import Component # cannot use TYPE_CHECKING here
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Tuple

from llmtuner.chat import ChatModel
from llmtuner.extras.misc import torch_gc
from llmtuner.hparams import GeneratingArguments
from llmtuner.webui.common import get_save_dir
from llmtuner.webui.locales import ALERTS
import jsonlines
import time

if TYPE_CHECKING:
    from llmtuner.webui.manager import Manager


def get_now_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def save_new_content(chatbot, context, Task):
    with jsonlines.open('./web_history/history.jsonl', mode='a') as writer:
        writer.write({"time": get_now_time(), "persuader": "persuader", "context": context, "task": Task, "chatbot": chatbot})

class WebChatModel(ChatModel):

    def __init__(
        self,
        manager: "Manager",
        demo_mode: Optional[bool] = False,
        lazy_init: Optional[bool] = True
    ) -> None:
        self.manager = manager
        self.demo_mode = demo_mode
        self.model = None
        self.tokenizer = None
        self.generating_args = GeneratingArguments()

        if not lazy_init: # read arguments from command line
            super().__init__()

        if demo_mode: # load demo_config.json if exists
            import json
            try:
                with open("demo_config.json", "r", encoding="utf-8") as f:
                    args = json.load(f)
                assert args.get("model_name_or_path", None) and args.get("template", None)
                super().__init__(args)
            except AssertionError:
                print("Please provided model name and template in `demo_config.json`.")
            except:
                print("Cannot find `demo_config.json` at current directory.")

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load_model(self, data: Dict[Component, Any]) -> Generator[str, None, None]:
        get = lambda name: data[self.manager.get_elem_by_name(name)]
        lang = get("top.lang")
        error = ""
        if self.loaded:
            error = ALERTS["err_exists"][lang]
        elif not get("top.model_name"):
            error = ALERTS["err_no_model"][lang]
        elif not get("top.model_path"):
            error = ALERTS["err_no_path"][lang]
        elif self.demo_mode:
            error = ALERTS["err_demo"][lang]

        if error:
            gr.Warning(error)
            yield error
            return

        if get("top.adapter_path"):
            adapter_name_or_path = ",".join([
                get_save_dir(get("top.model_name"), get("top.finetuning_type"), adapter)
            for adapter in get("top.adapter_path")])
        else:
            adapter_name_or_path = None

        yield ALERTS["info_loading"][lang]
        args = dict(
            model_name_or_path=get("top.model_path"),
            adapter_name_or_path=adapter_name_or_path,
            finetuning_type=get("top.finetuning_type"),
            quantization_bit=int(get("top.quantization_bit")) if get("top.quantization_bit") in ["8", "4"] else None,
            template=get("top.template"),
            flash_attn=(get("top.booster") == "flash_attn"),
            use_unsloth=(get("top.booster") == "unsloth"),
            rope_scaling=get("top.rope_scaling") if get("top.rope_scaling") in ["linear", "dynamic"] else None
        )
        super().__init__(args)

        yield ALERTS["info_loaded"][lang]

    def unload_model(self, data: Dict[Component, Any]) -> Generator[str, None, None]:
        lang = data[self.manager.get_elem_by_name("top.lang")]

        if self.demo_mode:
            gr.Warning(ALERTS["err_demo"][lang])
            yield ALERTS["err_demo"][lang]
            return

        yield ALERTS["info_unloading"][lang]
        self.model = None
        self.tokenizer = None
        torch_gc()
        yield ALERTS["info_unloaded"][lang]

    def start(
        self,
        chatbot: List[Tuple[str, str]],
        history: List[Tuple[str, str]],
        # persuader: str,
        context: str,
        Task: str,
        max_new_tokens: int,
        top_p: float,
        temperature: float,
        repetition_penalty: float
    ) -> Generator[Tuple[List[Tuple[str, str]], List[Tuple[str, str]]], None, None]:
        # if len(persuader) == 0:
        #     raise gr.Error("说服者不能为空")
        if len(context) == 0:
            raise gr.Error("说服场景不能为空")
        if len(Task) == 0:
            raise gr.Error("说服任务不能为空")
        system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        query=f"你是一个专业的说服者。你的身份是：情绪激动的说服者\n说服场景为：{context}\n说服任务是：{Task}\n接下来，你需要先说明可以使用哪些说服策略，然后在之后的对话中，进行分析、选择策略和对话。"
        # query=f"你是一个专业的说服者。你的身份是：说服者\n说服场景为：{context}\n说服任务是：{Task}\n接下来，你以说服者的身份开始进行对话。"
        
        chatbot = []
        history = []
        chatbot.append([None, ""])
        response = ""
        idx = 0
        for new_text in self.stream_chat(
            query, history, system, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty
        ):
            idx += 0.2
            dot_num = int(idx % 3) + 1
            dot = "." * dot_num
            response += new_text
            new_history = history + [(query, response)]

            ## dialogue + strategy
            chatbot[-1] = [None, self.postprocess(response)]

            ## only dialogue
            # if "该策略对应回复：" in response:
            #     chatbot[-1] = [None, self.postprocess(response.split("该策略对应回复：")[-1])]
            # else:
            #     chatbot[-1] = [None, f"初始化中，请稍候{dot}"]
            yield chatbot, new_history
        save_new_content(new_history, context, Task)

    def predict(
        self,
        chatbot: List[Tuple[str, str]],
        query: str,
        history: List[Tuple[str, str]],
        max_new_tokens: int,
        top_p: float,
        temperature: float,
        repetition_penalty: float
    ) -> Generator[Tuple[List[Tuple[str, str]], List[Tuple[str, str]]], None, None]:
        if len(history) == 0:
            raise gr.Error("请先点击开始按钮")
        
        if len(query) == 0:
            raise gr.Error("回复不能为空")
        system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        chatbot.append([query, ""])
        response = ""
        for new_text in self.stream_chat(
            query, history, system, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty
        ):
            response += new_text
            new_history = history + [(query, response)]

            ## dialogue + strategy
            chatbot[-1] = [query, self.postprocess(response)]

            ## only dialogue
            # if "该策略对应回复：" in response:
            #     chatbot[-1] = [query, self.postprocess(response.split("该策略对应回复：")[-1])]
            yield chatbot, new_history
        save_new_content(new_history, context = "", Task = "")
    def postprocess(self, response: str) -> str:
        blocks = response.split("```")
        for i, block in enumerate(blocks):
            if i % 2 == 0:
                blocks[i] = block.replace("<", "&lt;").replace(">", "&gt;")
        return "```".join(blocks)
