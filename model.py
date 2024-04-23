# Copyright (c) 2024 yaqiang.sun.
# 
# This source code is licensed under the license found in the LICENSE file 
# in the root directory of this source tree.

"""TaibaiChat class"""

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from transformers import GenerationConfig

from transformers import InfNanRemoveLogitsProcessor, LogitsProcessorList
def get_logits_processor() -> "LogitsProcessorList":
    r"""
    Gets logits processor that removes NaN and Inf logits.
    """
    logits_processor = LogitsProcessorList()
    logits_processor.append(InfNanRemoveLogitsProcessor())
    return logits_processor


class TaibaiChat:
    def __init__(self,checkpoint_path:str=None,cpu_only:bool=False) -> None:
        if checkpoint_path:
            self._load_model_tokenizer(checkpoint_path,cpu_only)
        pass

    def _load_model_tokenizer(self,checkpoint_path:str,cpu_only:bool=False):
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path, resume_download=True,
        )
        if cpu_only:
            device_map = "cpu"
        else:
            device_map = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map=device_map,
            resume_download=True,
        ).eval()
        self.model.generation_config.max_new_tokens = 1024    # For chat.
        

    def chat_stream(self, query, history):
        model = self.model
        tokenizer = self.tokenizer
        conversation = [
            {'role': 'system', 'content': 
                        "You are a helpful assistant. 你是一个乐于助人的助手。"
                        # "You are a helpful, respectful and honest assistant. "
                        # "Always answer as helpfully as possible, while being safe. "
                        # "Your answers should not include any harmful, unethical, "
                        # "racist, sexist, toxic, dangerous, or illegal content. "
                        # "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
                        # "If a question does not make any sense, or is not factually coherent, "
                        # "explain why instead of answering something not correct. "
                        # "If you don't know the answer to a question, please don't share false information."
             },
        ]
        for query_h, response_h in history:
            conversation.append({'role': 'user', 'content': query_h})
            conversation.append({'role': 'assistant', 'content': response_h})
        conversation.append({'role': 'user', 'content': query})
        inputs = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors='pt',
        )
        # inputs_word = tokenizer.apply_chat_template(
        #     conversation,
        #     add_generation_prompt=True,
        #     tokenize=False
        # )
        # print("inputs_word:\n",inputs_word,"\n end.")
        inputs = inputs.to(model.device)
        streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True)
        generating_args = {}
        generating_args["do_sample"] = True
        generating_args["temperature"] = 0.95
        generating_args["top_p"] =  0.7
        generating_args["pad_token_id"] = 32000
        generating_args["eos_token_id"] = [2]
        generating_args["max_new_tokens"] = 512
        generation_kwargs = dict(
            input_ids=inputs,
            streamer=streamer,
            logits_processor=get_logits_processor(),
            generation_config=GenerationConfig(**generating_args),
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text