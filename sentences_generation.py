import json
import time

from config.chatgpt_config import config_dict
from src.openai_request import OpenAI_Request
from tools.cfg_wrapper import load_config
from tools.context import ContextHandler
from tools.tokennizer import Tokennizer
import os

SCENE_TYPE = ["urban street", "suburb street", "urban scene", "highway scene"]
DAY_WEATHER = ["sunny day", "clear night", "cloudy day", "rainy night"]
TARGET_OBJ_CLASS_MAPPING = {'CAR': "car",
                            'VAN': "van",
                            'TRUCK': "truck",
                            'BUS': "bus",
                            'PEDESTRIAN': "pedestrian",
                            'CYCLIST': "cyclist",
                            'TRICYCLIST': "tricyclist"}
LIGHT_CONDITION = ["bright", "dim"]
DETAIL = ["road", "bridge", "tunnel", "parking lot", "building", "tunnel", "gas station","wall",
          "toll booth", "cityscape","construction site"]


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


def process_data(data):
    prompt = data["text"]
    obj_dict = data['obj']
    scene_type = ''
    for scene in SCENE_TYPE:
        if scene in prompt:
            scene_type = scene
            break
    weather_day = ''
    for weather in DAY_WEATHER:
        if weather in prompt:
            weather_day = weather
            break
    light_condition = ''
    for light in LIGHT_CONDITION:
        if light in prompt:
            light_condition = light
            break
    object = ''
    for obj in TARGET_OBJ_CLASS_MAPPING:
        if obj in obj_dict:
            object += str(obj_dict[obj]) + ' ' + TARGET_OBJ_CLASS_MAPPING[obj] + ','
    if object == '':
        object = 'empty'
    if object[-1] == ',':
        object = object[:-1]
    detail = ''
    for det in DETAIL:
        if det in prompt:
            detail += det + ','
    assert (scene_type != ''
            and weather_day != ''
            and light_condition != ''
            and object != ''), \
        f"scene_type:{scene_type}, weather_day:{weather_day}, light_condition:{light_condition}, object:{object}, detail:{detail}"
    return scene_type, weather_day, light_condition, object, detail
def sentences_generation(keys, json_file_path, system_prompt, model_name, request_address, context_handler, tokenizer,
                         log_time=False, context_max=3200,from_start_generation=True):
    requestor = OpenAI_Request(keys, model_name, request_address)
    #context_handler.clear()
    system_prompt_length = tokenizer.num_tokens_from_string(system_prompt)
    context_handler.append_cur_to_context(system_prompt, system_prompt_length)
    requestor.post_request(context_handler.context)
    processed_dict_list = []
    with open(json_file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            processed_dict_list.append(data)
    if os.path.exists(json_file_path[:-6] + '_updated.jsonl') and from_start_generation:
        os.remove(json_file_path[:-6] + '_updated.jsonl')
    with open(json_file_path[:-6] + '_updated.jsonl', 'a') as f:
        for i, data in enumerate(processed_dict_list):
            if (i+1)%200==0:
                context_handler.clear()
                system_prompt_length = tokenizer.num_tokens_from_string(system_prompt)
                context_handler.append_cur_to_context(system_prompt, system_prompt_length)
                requestor.post_request(context_handler.context)
            scene_type, weather_day, light_condition, object, detail = process_data(data)
            input_s = (f"Scene type: {scene_type}; "
                       f"Time and weather conditions: {weather_day}; "
                       f"Light Condition: {light_condition}; "
                       f"Objects in the image:{object}; "
                       f"Specific scene details:{detail}")
            inputs_length = tokenizer.num_tokens_from_string(input_s)
            context_handler.append_cur_to_context(input_s, inputs_length)

            st_time = time.time()
            while True:
                try:
                    res = requestor.post_request(context_handler.context)

                    if res.status_code == 200:

                        response = res.json()['choices'][0]['message']['content']
                        # cut \n for show
                        response = response.lstrip("\n")

                        completion_length = res.json()['usage']['completion_tokens']
                        total_length = res.json()['usage']['total_tokens']
                        data['text'] = response
                        print(f"\nresponse : {response}")
                        f.write(json.dumps(data) + "\n")
                        context_handler.append_cur_to_context(response, completion_length, tag=1)
                        if total_length > context_max:
                            context_handler.cut_context(total_length, tokenizer)
                        break

                    else:
                        status_code = res.status_code
                        reason = res.reason
                        des = res.text
                        print(f"error happened in {i}th data")
                        print(f'visit error :\n status code: {status_code}\n reason: {reason}\n err description: {des}\n '
                                    f'please check whether your account  can access OpenAI API normally')
                        print('sleep 60s')
                        time.sleep(60)
                except Exception as e:
                    # 处理其他类型的错误
                    print(f"An error occurred: {e}")
                    print("Waiting for 60 seconds before retrying.")
                    print(f"error happened in {i}th data")
                    time.sleep(60)

            ed_time = time.time()

            if log_time:
                print(f'time cost : {ed_time - st_time}')


if __name__ == '__main__':
    # load config
    config = load_config(config_dict)
    json_file_path = config.sentence_generation_config.json_path
    system_prompt = config.sentence_generation_config.system_prompt
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
    sentences_generation(keys, json_file_path, system_prompt, model_name, request_address, context, tokenizer)
    # chat_test(keys, model_name, request_address, context, tokenizer)
