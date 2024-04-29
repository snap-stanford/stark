import time
from src.tools.api_lib import anthropic_client


def complete_text_claude(message, 
                         model="claude-2.1",
                         json_object=False,
                         max_tokens=2048, 
                         temperature=1, 
                         max_retry=1,
                         sleep_time=0,
                         tools=[],
                         **kwargs
                         ):
    """ Call the Claude API to complete a prompt."""
    if isinstance(message, str):
        if json_object:
            message = "You are a helpful assistant designed to output in JSON format." + message
        messages = [{"role": "user", "content": message}] 
    else:
        messages = message

    for cnt in range(max_retry):
        try:
            message = anthropic_client.beta.tools.messages.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                **kwargs
            )
            completion = message.to_dict()
            return completion["content"][0]['text']
        except Exception as e:
            print(cnt, "=>", e, f' [sleep for {sleep_time} sec]')
            time.sleep(sleep_time)
    raise e
    
