# Copyright (c) 2024 yaqiang.sun.
# 
# This source code is licensed under the license found in the LICENSE file 
# in the root directory of this source tree.

"""A command-line interactive chat demo"""

import os
import argparse
import platform

from model import TaibaiChat


DEFAULT_CKPT_PATH = 'models_dir/Taibai/Taibai-Chat'

def _get_input() -> str:
    while True:
        try:
            message = input('User> ').strip()
        except UnicodeDecodeError:
            print('[ERROR] Encoding error in input!')
            continue
        except KeyboardInterrupt:
            exit(1)
        if message:
            return message
        print('[ERROR] Query is empty!')

def main():
    parser = argparse.ArgumentParser(
    description='Taibai-Chat command-line interactive chat demo.')
    parser.add_argument("-c", "--checkpoint", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint path, default to %(default)r")
    parser.add_argument("-s", "--seed", type=int, default=9989, help="Random seed")
    parser.add_argument("--cpu-only", action="store_true", help="Run model with CPU")
    args = parser.parse_args()

    history = []

    taibai_chat = TaibaiChat(args.checkpoint, args.cpu_only)

    while True:
        query = _get_input()
        # print(f"\nUser-Input: {query}")
        print(f"\nTaibai-Chat: ", end="")
        try:
            partial_text = ''
            for new_text in taibai_chat.chat_stream(query, history):
                print(new_text, end='', flush=True)
                partial_text += new_text
            response = partial_text
            history.append((query, response))
            print()
        except KeyboardInterrupt:
            print('[WARNING] Generation interrupted')
            continue

    pass


if __name__ == "__main__":
    main()