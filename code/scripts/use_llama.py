import multiprocessing
import os


def run_pipeline(input_file, output_file, temperature):
    from huggingface_hub import login
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch
    import pandas as pd
    from tqdm import tqdm
    import prompts
    from vllm import LLM, SamplingParams
    import numpy as np
    from peft import PeftModel
    import sys

    SAFE_MAX_TOKENS = 8000
    
    login(token='YOUR_HF_TOKEN')
    
    base_model_name = "meta-llama/Llama-3.3-70B-Instruct"
        
    merged_output_path = "PATH_TO_FINETUNED_LLAMA_MODEL"

    model_name = merged_output_path
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    def truncate_prompt(prompt, context, newcomer, reply):
        '''
        function to truncate prompts so they fit within the model's token limit. Removes tokens one at a time from each comment in the triplet.
        '''
        prompt_tokens = tokenizer(prompt, add_special_tokens=False, return_tensors=None)['input_ids']
        context_tokens = tokenizer(context, add_special_tokens=False, return_tensors=None)['input_ids']
        newcomer_tokens = tokenizer(newcomer, add_special_tokens=False, return_tensors=None)['input_ids']
        reply_tokens = tokenizer(reply, add_special_tokens=False, return_tensors=None)['input_ids']
        
        
        parts = [
            ('context', context_tokens),
            ('newcomer', newcomer_tokens),
            ('reply', reply_tokens)
        ]
        
        excess_tokens = len(prompt_tokens) - SAFE_MAX_TOKENS
         
        while excess_tokens > 0:
            parts.sort(key=lambda x: len(x[1]), reverse=True)
            name, tokens = parts[0]
    
            if len(tokens) > 1:
                tokens = tokens[:-1]  # Remove one token
                excess_tokens -= 1
            
            parts[0] = (name, tokens)
    

        context = tokenizer.decode([t for name, tks in parts if name == 'context' for t in tks], skip_special_tokens=True)
        newcomer = tokenizer.decode([t for name, tks in parts if name == 'newcomer' for t in tks], skip_special_tokens=True)
        reply = tokenizer.decode([t for name, tks in parts if name == 'reply' for t in tks], skip_special_tokens=True)
    
        return prompts.prompt_few_shot(context, newcomer, reply)    


    counterspeech_df = pd.read_csv(input_file, lineterminator='\n')
    counterspeech_df['context'] = counterspeech_df['context'].fillna('<MISSING>') #fill empty comments with a token to indicate missingness to the LLM
    counterspeech_df['newcomer'] = counterspeech_df['newcomer'].fillna('<MISSING>')
    counterspeech_df['reply'] = counterspeech_df['reply'].fillna('<MISSING>')
    
    counterspeech_df['prompt'] = counterspeech_df.apply(lambda row: prompts.prompt_few_shot(row['context'], row['newcomer'], row['reply']), axis=1).to_list() #put each comment triplet into a prompt

    counterspeech_df['prompt'] = counterspeech_df.apply(lambda row: truncate_prompt(row['prompt'], row['context'], row['newcomer'], row['reply']), axis=1).to_list() #truncate the prompts

    llm = LLM(model=model_name, tokenizer=model_name, tensor_parallel_size=4,
          download_dir="path_to_download_dir", max_model_len=8500, max_num_seqs=16,
          generation_config="vllm", gpu_memory_utilization=0.75)

    for i in range(5):
        if f'response_{i}' not in counterspeech_df:
            sampling_params = SamplingParams(temperature=round(float(temperature), 2), top_p=0.9, seed=i)
            
            outputs = llm.generate(counterspeech_df['prompt'], sampling_params)
        
            responses = []
            for output in outputs:
                responses.append(output.outputs[0].text)
                
            # Store responses in DataFrame
            counterspeech_df[f"response_{i}"] = responses
            
    
    counterspeech_df.to_csv(output_path, index=False)



@click.command()
@click.option("--input_file", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to input file to make predictions for. Assumes 3 text cols: context, newcomer, reply")
@click.option("--output_file", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Name of the output file with prediction columns.")
@click.option("--temperature", default=1.2, show_default=True, help="Temperature of the LLM")
def main(input_file, output_file, temperature):
    multiprocessing.set_start_method("spawn", force=True)
    run_pipeline(input_file, output_file, temperature)


if __name__ == "__main__":
    main()
