""" This file contains the code for calling all LLM APIs. """

import os
import multiprocessing
import tiktoken
import time
import torch
import sys
import re
import base64
import json
import requests
from functools import partial
sys.path.append('.')

enc = tiktoken.get_encoding("cl100k_base")
try:   
    import anthropic
    # setup anthropic API key
    anthropic_client = anthropic.Anthropic(api_key=open("config/claude_api_key.txt").read().strip())
except Exception as e:
    print(e)
    print("Could not load anthropic API key config/claude_api_key.txt.")
    
try:
    import openai
    # setup OpenAI API key
    openai.organization, openai.api_key = open("config/openai_api_key.txt").read().strip().split(":")    
    os.environ["OPENAI_API_KEY"] = openai.api_key 
except Exception as e:
    print(e)
    print("Could not load OpenAI API key config/openai_api_key.txt.")

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

MAX_OPENAI_RETRY = 5
MAX_CLAUDE_RETRY= 30

def log_to_file(log_file, prompt, completion, model, max_tokens_to_sample):
    """ Log the prompt and completion to a file."""
    with open(log_file, "a") as f:
        f.write("\n===================prompt=====================\n")
        f.write(f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}")
        num_prompt_tokens = len(enc.encode(f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}"))
        f.write(f"\n==================={model} response ({max_tokens_to_sample})=====================\n")
        f.write(completion)
        num_sample_tokens = len(enc.encode(completion))
        f.write("\n===================tokens=====================\n")
        f.write(f"Number of prompt tokens: {num_prompt_tokens}\n")
        f.write(f"Number of sampled tokens: {num_sample_tokens}\n")
        f.write("\n\n")


def complete_text_claude_with_history(prompt, 
                                      tools,
                                      model="claude-2.1",
                                      history=None,
                                      json_object=False,
                                      max_tokens=2000, 
                                      temperature=1, **kwargs):
    """ Call the Claude API to complete a prompt."""

    if json_object:
        prompt = "You are a helpful assistant designed to output in JSON format." + prompt
    if history is None:
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = history + [{"role": "user", "content": prompt}]
    cnt = 0
    while True:
        try:
            import pdb; pdb.set_trace()
            message = anthropic_client.tools.messages.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                **kwargs
            )
            break
        except Exception as e:
            print(cnt, "=>", e)
            cnt += 1
            if cnt > MAX_CLAUDE_RETRY:
                raise e
    return message

def get_gpt_output_with_history(message, history=None, model="gpt-4-1106-preview", max_tokens=2048, temperature=1, json_object=False):
    assert isinstance(message, str)
    if json_object and history is None:
        history = [{"role": "system", "content": "You are a helpful assistant designed to output JSON."}]
    if not json_object and history is None:
        messages = [{"role": "user", "content": message}]
    else:
        messages = history + [{"role": "user", "content": message}]
    kwargs = {"response_format": { "type": "json_object" }} if json_object else {}
    try:
        chat = openai.OpenAI().chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            # max_tokens=max_tokens,
            **kwargs
            )
    except Exception as e:
        print(f'{e}\nTry after 1 min')
        time.sleep(61)
        # print(f'{messages=}')
        chat = openai.OpenAI().chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            # max_tokens=max_tokens,
            **kwargs
            )
    reply = chat.choices[0].message.content 

    return reply


def complete_text_claude(message, 
                         model="claude-2.1",
                         json_object=False,
                         max_tokens=2000, 
                         temperature=1, 
                         tools=[],
                         **kwargs):
    """ Call the Claude API to complete a prompt."""
    cnt = 0
    if isinstance(message, str):
        if json_object:
            message = "You are a helpful assistant designed to output in JSON format." + message
        messages = [{"role": "user", "content": message}] 
    else:
        messages = message
    while True:
        try:
            message = anthropic_client.beta.tools.messages.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                **kwargs
            )
            break
        except Exception as e:
            print(cnt, "=>", e)
            cnt += 1
            if cnt > 30:
                raise e
    completion = message.to_dict()
    contents = completion["content"][0]['text']
    return contents


def complete_texts_claude(texts: list, model="claude-2.1", temperature=0.5, n_max_nodes=5):
    """ 
    Get embeddings for a list of texts.
    """
    processes = min(len(texts), n_max_nodes)
    partial_func = partial(complete_text_claude, model=model, temperature=temperature)
    with multiprocessing.Pool(processes=processes) as pool:
        results = pool.map(partial_func, texts)
    return results


def get_ada_embedding(text, model="text-embedding-ada-002"):
    assert isinstance(text, str), f'text must be str, but got {type(text)}'
    assert len(text) > 0, f'text to be embedded should be non-empty'
    if openai.__version__.startswith('0.'):
        try:
            emb = openai.Embedding.create(input=[text], model=model)
        except (openai.error.RateLimitError, 
                openai.error.ServiceUnavailableError) as e:
            print(f'{e}, sleep for 1 min')
            time.sleep(100)
            emb =  openai.Embedding.create(input=[text], model=model)
        except openai.error.InvalidRequestError as e:
            e = str(e)
            ori_length = len(text.split(' '))
            match = re.search(r'maximum context length is (\d+) tokens, however you requested (\d+) tokens', e)
            if match is not None:
                max_length = int(match.group(1))
                cur_length = int(match.group(2))
                ratio = float(max_length) / cur_length
                for reduce_rate in range(9, 0, -1):
                    shorten_text = text.split(' ')
                    length = int(ratio * ori_length * (reduce_rate * 0.1))
                    shorten_text = ' '.join(shorten_text[:length])
                    try:
                        emb = openai.Embedding.create(input=[shorten_text], model=model)
                        print(f'length={length} works! reduce_rate={0.1 * reduce_rate}.')
                        break
                    except: pass
    else:
        client = openai.OpenAI()
        for retry in range(MAX_OPENAI_RETRY):
            emb = None
            try:
                emb = client.embeddings.create(input=[text], model=model)
                break
            except openai.BadRequestError as e:
                print(f'{e}')
                e = str(e)
                ori_length = len(text.split(' '))
                match = re.search(r'maximum context length is (\d+) tokens, however you requested (\d+) tokens', e)
                if match is not None:
                    max_length = int(match.group(1))
                    cur_length = int(match.group(2))
                    ratio = float(max_length) / cur_length
                    for reduce_rate in range(9, 0, -1):
                        shorten_text = text.split(' ')
                        length = int(ratio * ori_length * (reduce_rate * 0.1))
                        shorten_text = ' '.join(shorten_text[:length])
                        try:
                            emb = client.embeddings.create(input=[shorten_text], model=model)
                            print(f'length={length} works! reduce_rate={0.1 * reduce_rate}.')
                            break
                        except: 
                            pass
            except (openai.RateLimitError, openai.APITimeoutError) as e:
                print(f'{e}, sleep for 1 min')
                time.sleep(100)
                emb = client.embeddings.create(input=[text], model=model)
                break
            if emb is not None:
                break
        if emb is None:
            import pdb; pdb.set_trace()
    emb = torch.FloatTensor(emb.data[0].embedding).view(1, -1)
    return emb


def get_ada_embeddings(texts, n_max_nodes=5, model="text-embedding-ada-002"):
    """ 
    Get embeddings for a list of texts.
    """
    assert isinstance(texts, list), f'texts must be list, but got {type(texts)}'
    assert all([len(s) > 0 for s in texts]), f'every string in the `texts` list to be embedded should be non-empty'

    processes = min(len(texts), n_max_nodes)
    ada_encoder = partial(get_ada_embedding, model=model)
    with multiprocessing.Pool(processes=processes) as pool:
        results = pool.map(ada_encoder, texts)

    results = torch.cat(results, dim=0)
    return results


def get_gpt4v_output(image_path, message, model="gpt-4-turbo", max_tokens=50, max_retry=3, json_object=False):
    with open(image_path, "rb") as image_file:
        base64_image =  base64.b64encode(image_file.read()).decode('utf-8')
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
        }

    payload = {"model": model,
               "max_tokens": max_tokens, 
               "messages": [
                {"role": "user", 
                 "content": [
                    {"type": "text", "text": message},
                    {"type": "image_url", 
                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }]}
    if json_object:
        payload['response_format'] = {"type": "json_object"}
    for i in range(max_retry):
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", 
                                     headers=headers, json=payload)
            break
        except Exception as e:
            print(f'{e}, sleep for 1 min')
            time.sleep(100)
    result = response.json()['choices'][0]['message']['content']
    if json_object:
        return json.loads(result)
    return result


def get_gpt4v_outputs(image_path_list: list, message: str, model="gpt-4-turbo", n_max_nodes=5, json_object=False):
    processes = min(len(image_path_list), n_max_nodes)
    gpt4vqa = partial(get_gpt4v_output, model=model, message=message, json_object=json_object)
    with multiprocessing.Pool(processes=processes) as pool:
        results = pool.map(gpt4vqa, image_path_list)
    return results


def get_gpt_output(message, model="gpt-4-1106-preview", max_tokens=2048, temperature=1, json_object=False):

    if json_object:
        if isinstance(message, str) and not 'json' in message.lower():
            message = 'You are a helpful assistant designed to output JSON. ' + message
    if 'instruct' in model:
        assert isinstance(message, str)
        if openai.__version__.startswith('0.'):
            try:
                reply = openai.Completion.create(
                    model=model,
                    prompt=message,
                    max_tokens=max_tokens
                    )
            except openai.error.RateLimitError:
                time.sleep(100)
                reply = openai.Completion.create(
                    model=model,
                    prompt=message,
                    max_tokens=max_tokens
                    )
            reply = reply['choices'][0]['text']
        else:
            messages = [{"role": "user", "content": message}]
            try:
                response = openai.OpenAI().chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature
                    )
            except openai.error.RateLimitError:
                time.sleep(100)
                response = openai.OpenAI().chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature
                    )
            reply = response.choices[0].message.content
    else:
        if openai.__version__.startswith('0.'):
            if isinstance(message, str):
                messages = [{"role": "user", "content": message}] 
            else:
                messages = message
            try:
                chat = openai.ChatCompletion.create(
                    model=model, messages=messages
                ) 
            except Exception as e:
                print(f'{e}\nTry after 1 min')
                time.sleep(100)
                chat = openai.ChatCompletion.create(
                    model=model, messages=messages
                ) 
            reply = chat.choices[0].message.content 
        else:
            if isinstance(message, str):
                messages = [{"role": "user", "content": message}] 
            else:
                messages = message
            kwargs = {"response_format": { "type": "json_object" }} if json_object else {}
            cnt = 0
            while True:
                try:
                    chat = openai.OpenAI().chat.completions.create(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        **kwargs
                        )
                    break
                except Exception as e:
                    print(cnt, "=>", e, '. Try after 1 min')
                    time.sleep(100)
                cnt += 1
                if cnt > MAX_OPENAI_RETRY:
                    raise e
            reply = chat.choices[0].message.content 

    return reply


def get_gpt_outputs(texts: list, model="gpt-4-1106-preview", temperature=0.5, json_object=False, n_max_nodes=5):
    """ 
    Get embeddings for a list of texts.
    """
    processes = min(len(texts), n_max_nodes)
    partial_func = partial(get_gpt_output, model=model, temperature=temperature, json_object=json_object)
    with multiprocessing.Pool(processes=processes) as pool:
        results = pool.map(partial_func, texts)
    return results

