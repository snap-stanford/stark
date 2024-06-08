import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Dictionary to cache loaded Hugging Face models and tokenizers
loaded_hf_models = {}


def complete_text_hf(message: str, 
                     model: str = "huggingface/codellama/CodeLlama-7b-hf", 
                     max_tokens: int = 2000, 
                     temperature: float = 0.5, 
                     json_object: bool = False,
                     max_retry: int = 1,
                     sleep_time: int = 0,
                     stop_sequences: list = [], 
                     **kwargs) -> str:
    """
    Generate text completion using a specified Hugging Face model.

    Args:
        message (str): The input text message for completion.
        model (str): The Hugging Face model to use. Default is "huggingface/codellama/CodeLlama-7b-hf".
        max_tokens (int): The maximum number of tokens to generate. Default is 2000.
        temperature (float): Sampling temperature for generation. Default is 0.5.
        json_object (bool): Whether to format the message for JSON output. Default is False.
        max_retry (int): Maximum number of retries in case of an error. Default is 1.
        sleep_time (int): Sleep time between retries in seconds. Default is 0.
        stop_sequences (list): List of stop sequences to halt the generation.
        **kwargs: Additional keyword arguments for the `generate` function.

    Returns:
        str: The generated text completion.
    """
    if json_object:
        message = "You are a helpful assistant designed to output in JSON format." + message
    
    # Determine the device to run the model on (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = model.split("/", 1)[1]
    
    # Load the model and tokenizer if not already loaded
    if model_name in loaded_hf_models:
        hf_model, tokenizer = loaded_hf_models[model_name]
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(model).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model)
        loaded_hf_models[model_name] = (hf_model, tokenizer)
    
    # Encode the input message
    encoded_input = tokenizer(message, return_tensors="pt", return_token_type_ids=False).to(device)
    
    for cnt in range(max_retry):
        try:
            # Generate text completion
            output = hf_model.generate(
                **encoded_input,
                temperature=temperature,
                max_new_tokens=max_tokens,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                **kwargs,
            )
            # Decode the generated sequences
            sequences = output.sequences
            sequences = [sequence[len(encoded_input.input_ids[0]):] for sequence in sequences]
            all_decoded_text = tokenizer.batch_decode(sequences)
            completion = all_decoded_text[0]
            return completion
        except Exception as e:
            print(f"Retry {cnt}: {e}")
            time.sleep(sleep_time)
    
    raise RuntimeError("Failed to generate text completion after max retries")
