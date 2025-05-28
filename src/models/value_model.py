import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

class Llama_VM(nn.Module):
    def __init__(self, base, vocab_size=32000):
        super(Llama_VM, self).__init__()
        self.base_model = base
        self.LN = nn.Linear(vocab_size, 1, dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
        value_outputs = self.LN(outputs)
        return value_outputs.squeeze(dim=1)

class Qwen_VM(nn.Module):
    def __init__(self, base, vocab_size=152064):
        super(Qwen_VM, self).__init__()
        self.base_model = base
        self.LN = nn.Linear(vocab_size, 1, dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
        value_outputs = self.LN(outputs)
        return value_outputs.squeeze(dim=1)

class Llama_PRM(nn.Module):
    def __init__(self, base):
        super(Llama_PRM, self).__init__()
        self.base_model = base

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(outputs, dim=-1)
        output = probs[:, -1, 7081]  # Adjust token ID as needed for "True"
        return output

class Qwen_PRM(nn.Module):
    def __init__(self, base):
        super(Qwen_PRM, self).__init__()
        self.base_model = base

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(outputs, dim=-1)
        output = probs[:, -1, 7081]  # Adjust token ID as needed for "True"
        return output

def get_llama3_model(model_dir, variant="8b"):
    model_path = f"{model_dir}/Llama3-{variant}" if variant else model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    model.eval()
    return tokenizer, model

def get_llama31_model(model_dir, variant="8b"):
    model_path = f"{model_dir}/Llama3.1-{variant}" if variant else model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    model.eval()
    return tokenizer, model

def get_qwen2_model(model_dir, variant="8b"):
    model_path = f"{model_dir}/Qwen2-{variant}" if variant else model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    model.eval()
    return tokenizer, model

def get_qwen25_model(model_dir, variant="8b"):
    model_path = f"{model_dir}/Qwen2.5-{variant}" if variant else model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    model.eval()
    return tokenizer, model

def get_llama_value_model(base_model_dir, state_dict_file, variant="8b"):
    model_path = f"{base_model_dir}/Llama3-{variant}" if variant else base_model_dir
    value_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    value_base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    if state_dict_file is None:
        return value_tokenizer, value_base_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = value_base_model.config.vocab_size
    VM = Llama_VM(value_base_model, vocab_size)
    VM.load_state_dict(torch.load(state_dict_file))
    VM.to(device)
    VM.eval()
    return value_tokenizer, VM

def get_qwen_value_model(base_model_dir, state_dict_file, variant="8b"):
    model_path = f"{base_model_dir}/Qwen2-{variant}" if variant else base_model_dir
    value_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    value_base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    if state_dict_file is None:
        return value_tokenizer, value_base_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = value_base_model.config.vocab_size
    VM = Qwen_VM(value_base_model, vocab_size)
    VM.load_state_dict(torch.load(state_dict_file))
    VM.to(device)
    VM.eval()
    return value_tokenizer, VM

def get_llama_prm(base_model_dir, state_dict_file, variant="8b"):
    model_path = f"{base_model_dir}/Llama3-{variant}" if variant else base_model_dir
    prm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prm_base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    if state_dict_file is None:
        return prm_tokenizer, prm_base_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prm = Llama_PRM(prm_base_model)
    prm.load_state_dict(torch.load(state_dict_file))
    prm.to(device)
    prm.eval()
    return prm_tokenizer, prm

def get_qwen_prm(base_model_dir, state_dict_file, variant="8b"):
    model_path = f"{base_model_dir}/Qwen2-{variant}" if variant else base_model_dir
    prm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prm_base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    if state_dict_file is None:
        return prm_tokenizer, prm_base_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prm = Qwen_PRM(prm_base_model)
    prm.load_state_dict(torch.load(state_dict_file))
    prm.to(device)
    prm.eval()
    return prm_tokenizer, prm

def get_llama_response(query, model, tokenizer, max_length=2048, truncation=True, max_new_tokens=1024, temperature=0.7, do_sample=False):
    cnt = 2
    all_response = ''
    messages = [{"role": "user", "content": query}]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", max_length=max_length, truncation=truncation).to('cuda')
    
    while cnt:
        try:
            output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)
            response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
            all_response = response
            break
        except Exception as e:
            print(f'Error:{e}, obtain response again...\n')
            cnt -= 1
    
    if not cnt:
        return []
    
    split_response = all_response.strip().split('\n')
    return split_response

def get_qwen_response(query, model, tokenizer, max_length=2048, truncation=True, max_new_tokens=1024, temperature=0.7, do_sample=False):
    cnt = 2
    all_response = ''
    messages = [{"role": "user", "content": query}]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", max_length=max_length, truncation=truncation).to('cuda')
    
    while cnt:
        try:
            output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)
            response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
            all_response = response
            break
        except Exception as e:
            print(f'Error:{e}, obtain response again...\n')
            cnt -= 1
    
    if not cnt:
        return []
    
    split_response = all_response.strip().split('\n')
    return split_response

def get_local_value(prompt_answer, model, tokenizer, max_length=2048, low=0, high=1):
    encoded_pair = tokenizer.encode_plus(
        prompt_answer,
        padding='max_length',
        max_length=max_length,
        truncation=True,
        return_tensors='pt',
    )
    input_ids = encoded_pair['input_ids'].to('cuda')
    attention_mask = encoded_pair['attention_mask'].to('cuda')
    value = model(input_ids, attention_mask).item()
    value = min(high, max(value, low))
    return value

