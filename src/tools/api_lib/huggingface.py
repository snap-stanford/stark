import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

loaded_hf_models = {}


class StopAtSpecificTokenCriteria(StoppingCriteria):
    def __init__(self, stop_sequence):
        super().__init__()
        self.stop_sequence = stop_sequence

    def __call__(self, input_ids, scores, **kwargs):
        # Create a tensor from the stop_sequence
        stop_sequence_tensor = torch.tensor(self.stop_sequence, 
                                            device=input_ids.device, 
                                            dtype=input_ids.dtype
                                            )

        # Check if the current sequence ends with the stop_sequence
        current_sequence = input_ids[:, -len(self.stop_sequence) :]
        return bool(torch.all(current_sequence == stop_sequence_tensor).item())


def complete_text_hf(message, 
                     model="huggingface/codellama/CodeLlama-7b-hf", 
                     max_tokens=2000, 
                     temperature=0.5, 
                     json_object=False,
                     max_retry=1,
                     sleep_time=0,
                     stop_sequences=[], 
                     **kwargs):
    if json_object:
        message = "You are a helpful assistant designed to output in JSON format." + message
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.split("/", 1)[1]
    if model in loaded_hf_models:
        hf_model, tokenizer = loaded_hf_models[model]
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(model).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model)
        loaded_hf_models[model] = (hf_model, tokenizer)
        
    encoded_input = tokenizer(message, 
                              return_tensors="pt", 
                              return_token_type_ids=False
                              ).to(device)
    for cnt in range(max_retry):
        try:
            output = hf_model.generate(
                **encoded_input,
                temperature=temperature,
                max_new_tokens=max_tokens,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                **kwargs,
            )
            sequences = output.sequences
            sequences = [sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences]
            all_decoded_text = tokenizer.batch_decode(sequences)
            completion = all_decoded_text[0]
            return completion
        except Exception as e:
            print(cnt, "=>", e)
            time.sleep(sleep_time)
    raise e

