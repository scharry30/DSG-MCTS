import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Function to load the Llama model
def get_inference_model_llama(model_dir):
    inference_model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)
    inference_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    device = "cuda"
    inference_model.to(device)
    return inference_tokenizer, inference_model

# Function to generate a response using the Llama model
def get_local_response_llama(query, model, tokenizer, max_length=2048, truncation=True, max_new_tokens=1024, temperature=0.7, do_sample=False):
    attempt = 2
    response = ''
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    message = f'<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id>assistant<|end_header_id|>\n\n'
    data = tokenizer.encode_plus(message, max_length=max_length, truncation=truncation, return_tensors='pt')
    input_ids = data['input_ids'].to('cuda')
    attention_mask = data['attention_mask'].to('cuda')
    
    while attempt:
        try:
            output = model.generate(input_ids, attention_mask=attention_mask, do_sample=do_sample, max_new_tokens=max_new_tokens, temperature=temperature, eos_token_id=terminators, pad_token_id=tokenizer.eos_token_id)
            processed_string = tokenizer.decode(output[0], skip_special_tokens=False).split('<|end_header_id|>')[2].strip().split('<|eot_id|>')[0]
            response = processed_string.split('<|end_of_text|>')[0].strip()
            break
        except Exception as e:
            attempt -= 1
    if not attempt:
        return []
    return response.strip().split('\n')

# Function to calculate the step value using a local value model
def get_local_value_model(prompt_answer, value_model, value_tokenizer, max_length=2048, low=0, high=1):
    return get_local_value(prompt_answer, value_model, value_tokenizer, max_length=max_length, low=low, high=high)

# Function to handle the inference response generation based on Llama
def local_inference_model(query, inference_model, inference_tokenizer, max_length=2048, truncation=True, do_sample=False, max_new_tokens=1024, temperature=0.7):
    return get_local_response_llama(query, inference_model, inference_tokenizer, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=do_sample)
