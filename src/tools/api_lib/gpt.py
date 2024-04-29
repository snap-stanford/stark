import time
import openai


def get_gpt_output(message, 
                   model="gpt-4-1106-preview", 
                   max_tokens=2048, 
                   temperature=1, 
                   max_retry=1,
                   sleep_time=60,
                   json_object=False):

    if json_object:
        if isinstance(message, str) and not 'json' in message.lower():
            message = 'You are a helpful assistant designed to output JSON. ' + message

    if isinstance(message, str):
        messages = [{"role": "user", "content": message}] 
    else:
        messages = message
    kwargs = {"response_format": { "type": "json_object" }} if json_object else {}

    for cnt in range(max_retry):
        try:
            chat = openai.OpenAI().chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
                )
            return chat.choices[0].message.content 
        except Exception as e:
            print(cnt, "=>", e, f' [sleep for {sleep_time} sec]')
            time.sleep(sleep_time)
    raise e
    
