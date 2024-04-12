# Copyright (c) 2024 yaqiang.sun.
# 
# This source code is licensed under the license found in the LICENSE file 
# in the root directory of this source tree.

"""TaibaiChat class"""

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

class TaibaiChat:
    def __init__(self,checkpoint_path:str=None,cpu_only:bool=False) -> None:
        if checkpoint_path is None:
            self.model, self.tokenizer = self._load_model_tokenizer(checkpoint_path,cpu_only)
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
            {'role': 'system', 'content': 'You are a helpful assistant.'},
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
        inputs = inputs.to(model.device)
        streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True)
        generation_kwargs = dict(
            input_ids=inputs,
            streamer=streamer,
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text