import os
import sys
import argparse
import torch
import transformers
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, Dict, Sequence, List
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import random
import numpy as np

# Define necessary constants and special tokens
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{query}\n\n### Response:"
    ),
}

# Function to simulate MDP-based strategy selection
def select_strategy(state):
    strategies = ['decomposition', 'analogy', 'backward_reasoning', 'hypothesis_testing']
    # Randomly choose a strategy for simplicity in this example (should be optimized using RL)
    return random.choice(strategies)

# MDP-based strategy optimization
def optimize_strategy(state, reward):
    # In real implementation, this would involve updating a policy network based on reward
    pass

# Function to simulate strategy-guided MCTS
def mcts_select_node(Q, N, strategy, C=1):
    # Modify the selection criteria by considering strategy alignment
    strategy_bonus = 0.1 if strategy in ['decomposition', 'hypothesis_testing'] else 0
    return Q + C * np.sqrt(np.log(N)) + strategy_bonus

def smart_tokenizer_and_embedding_resize(special_tokens_dict: Dict, tokenizer: transformers.PreTrainedTokenizer, model: transformers.PreTrainedModel):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

# Update preprocessing for dynamic strategy-based MCTS
def preprocess_with_strategy(sources, targets, tokenizer, strategy):
    sources_tokenized = _tokenize_fn(sources, tokenizer)
    input_ids = sources_tokenized["input_ids"]
    
    # Apply strategy adjustment to the input_ids (e.g., additional tokens for strategies)
    if strategy == 'decomposition':
        input_ids = [id + [tokenizer.encode('<decompose>')] for id in input_ids]
    elif strategy == 'analogy':
        input_ids = [id + [tokenizer.encode('<analogy>')] for id in input_ids]
    
    return dict(input_ids=input_ids, labels=copy.deepcopy(input_ids))

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, strategy):
        super(SupervisedDataset, self).__init__()
        with open(data_path, 'r') as f:
            dataset_for_eval = f.readlines()

        dataset_for_eval = [json.loads(item.strip()) for item in dataset_for_eval]
        sources = [PROMPT_DICT["prompt_no_input"].format_map(item) for item in dataset_for_eval]
        targets = [item['response'] for item in dataset_for_eval]
        
        self.strategy = strategy
        data_dict = preprocess_with_strategy(sources, targets, tokenizer, strategy)

        self.input_ids = data_dict["input_ids"] + data_dict["input_ids"][-100:]
        self.labels = data_dict["labels"] + data_dict["labels"][-100:]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], id=i)

# Main function integrating MDP-based strategy selection and MCTS
def main(rank, args):
    dist.init_process_group("nccl")
    world_size = torch.cuda.device_count()
    base_model = args.base_model
    data_path = args.data_path
    batch_size = args.batch_size

    model = LlamaForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)
    
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN), tokenizer=tokenizer, model=model)

    torch.cuda.set_device(rank)
    model.to(torch.cuda.current_device())
    model = DDP(model, device_ids=[torch.cuda.current_device()])
    model.eval()

    # Select strategy using MDP
    strategy = select_strategy('some_state')
    print(f"Selected strategy: {strategy}")
    
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path, strategy=strategy)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    for tempera in [0.7]:
        sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        dataloader = DataLoader(
            eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=batch_size, sampler=sampler, drop_last=True
        )

        generation_config = GenerationConfig(
            temperature=tempera,
            do_sample=args.do_sample,
            num_beams=1,
            max_new_tokens=256,
            num_return_sequences=1,
        )

        all_outputs = []
        for step, batch in tqdm(enumerate(dataloader)):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            
            # Simulate MCTS-based node selection
            node_score = mcts_select_node(0, 1, strategy)  # Example Q, N
            print(f"Node score based on strategy: {node_score}")
            
            with torch.no_grad():
                generation_output = model.module.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config, return_dict_in_generate=True)
            
            s = generation_output.sequences
            gather_outputs = sequence_gather(s, world_size, tokenizer.pad_token_id)
            gathered_inputs = sequence_gather(input_ids, world_size, tokenizer.pad_token_id)

            # Post-process output as needed
            outputs_string = tokenizer.batch_decode(torch.stack(gather_outputs), skip_special_tokens=True)
            all_outputs.append(outputs_string)

        if rank == 0:
            with open(args.out_path + '/raw_generation.json', 'w') as f:
                for item in all_outputs:
                    f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--base_model", default="", type=str, help="model path")
    parser.add_argument("--data_path", default="", type=str, help="config path")
    parser.add_argument("--batch_size", type=int, default=0, help="batch size")
    parser.add_argument("--out_path", default="", type=str, help="config path")
    parser.add_argument("--do_sample", default=False, type=bool, help="config path")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    main(local_rank, args)
