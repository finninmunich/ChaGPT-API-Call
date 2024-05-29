import time

from config.chatgpt_config import config_dict
from src.openai_request import OpenAI_Request
from tools.cfg_wrapper import load_config
from tools.context import ContextHandler
from tools.tokennizer import Tokennizer


def chat_test(keys, model_name, request_address, context_handler, tokenizer, log_time=False, context_max=3200):
    requestor = OpenAI_Request(keys, model_name, request_address)
    input_s = (
        "Hello, ChatGPT. I require 50 template to assist me in crafting descriptive sentences that detail road scenes captured by a moving vehicle's front-face"
        "These descriptions should be based on labels that include: 1) Scene type (e.g., urban street, highway, etc.), "
        "2) Time and weather conditions (e.g., rainy night, sunny day, etc.), "
        "3) Lighting condition (e.g., bright, dim), 4) Objects in the image and their quantity (e.g., 3 cars, 2 trucks, etc.). "
        ""
        "Template Example:"
        "The [scene type] is depicted under [time and weather conditions] with [lighting condition] highlighting [objects in the image and their quantity]."
        ""
        "Filled-in examples, with scene details:"
        "The urban street is depicted under a rainy night with dim lighting highlighting 2 cars and 1 truck in the scene."
        ""
        "Filled-in examples, without scene details:"
        "The highway is depicted under a sunny day with bright lighting highlighting 3 cars.")
    inputs_length = tokenizer.num_tokens_from_string(input_s)
    context_handler.append_cur_to_context(input_s, inputs_length)

    st_time = time.time()

    res = requestor.post_request(context_handler.context)
    ed_time = time.time()

    if res.status_code == 200:

        response = res.json()['choices'][0]['message']['content']
        # cut \n for show
        response = response.lstrip("\n")

        completion_length = res.json()['usage']['completion_tokens']
        total_length = res.json()['usage']['total_tokens']
        print(f"\nresponse : {response}")

        context_handler.append_cur_to_context(response, completion_length, tag=1)
        if total_length > context_max:
            context_handler.cut_context(total_length, tokenizer)

    else:
        status_code = res.status_code
        reason = res.reason
        des = res.text
        raise print(f'visit error :\n status code: {status_code}\n reason: {reason}\n err description: {des}\n '
                    f'please check whether your account  can access OpenAI API normally')

    if log_time:
        print(f'time cost : {ed_time - st_time}')
    while 1:
        input_s = input('\nuser input : ')

        if input_s == "clear":
            context_handler.clear()
            print('start a new session')
            continue
        else:
            inputs_length = tokenizer.num_tokens_from_string(input_s)
            context_handler.append_cur_to_context(input_s,inputs_length)

        st_time = time.time()

        res = requestor.post_request(context_handler.context)
        ed_time = time.time()

        if res.status_code == 200:

            response = res.json()['choices'][0]['message']['content']
            # cut \n for show
            response = response.lstrip("\n")

            completion_length = res.json()['usage']['completion_tokens']
            total_length = res.json()['usage']['total_tokens']
            print(f"\nresponse : {response}")

            context_handler.append_cur_to_context(response,completion_length,tag=1)
            if total_length > context_max:
                context_handler.cut_context(total_length,tokenizer)

        else:
            status_code = res.status_code
            reason = res.reason
            des = res.text
            raise print(f'visit error :\n status code: {status_code}\n reason: {reason}\n err description: {des}\n '
                        f'please check whether your account  can access OpenAI API normally')




        if log_time:
            print(f'time cost : {ed_time - st_time}')


if __name__ == '__main__':
    # load config
    config = load_config(config_dict)
    keys = config.Acess_config.authorization
    model_name = config.Model_config.model_name
    request_address = config.Model_config.request_address

    # load context
    context_manage_config = config.Context_manage_config
    del_config = context_manage_config.del_config
    max_context = context_manage_config.max_context
    context = ContextHandler(max_context=max_context, context_del_config=del_config)

    # load tokenizer
    tokenizer = Tokennizer(model_name)

    # for test
    chat_test(keys, model_name, request_address, context, tokenizer)
