import time

from config.chatgpt_config import config_dict
from src.openai_request import OpenAI_Request
from tools.cfg_wrapper import load_config
from tools.context import ContextHandler
from tools.tokennizer import Tokennizer

SYSTEM_PROMPT = ("You will create descriptive sentences for me using words from dashboard camera images,"
                 " categorized as follows: 1) Scene type (urban street, highway scene, etc. ), 2) Time and weather conditions (rainy night, sunny day, etc.),"
                 " 3) Lighting (bright, dim), 4) Objects in the image (cars, trucks, etc., in the order provided, without specifying quantities),"
                 " and 5) Specific scene details (road, parking lot, gas station, bridge, etc.). Note that if the detail is a parking lot,"
                 " gas station, or toll booth, it represents the main setting of the image,"
                 " whereas other details like roads or bridges should be treated as more vague and less defining of the setting."
                 " Each sentence should integrate all categories, not exceed 40 words, and vividly depict the scene.")


def chat_test(keys, model_name, request_address, context_handler, tokenizer, log_time=False, context_max=3200):
    requestor = OpenAI_Request(keys, model_name, request_address)

    while 1:
        input_s = input('\nuser input : ')

        if input_s == "clear":
            context_handler.clear()
            print('start a new session')
            continue
        else:
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
