from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
from tqdm import tqdm
import argparse
from datasets import Dataset
import torch

parser = argparse.ArgumentParser(description="Extração de logitos")

parser.add_argument('--debug', action = 'store_true', dest = 'debug',
                           default = False, required = False,
                           help = 'Debug')
parser.add_argument('--prompt', action = 'store', dest = 'prompt_file',
                           default = False, required = False,
                           help = 'redacted images')
parser.add_argument('--model', action = 'store', dest = 'model',
                           default = None, required = False,
                           help = 'model to use')
parser.add_argument('--qtd_contos', action = 'store', dest = 'qtd_contos',
                           default = 1, required = False, type=int,
                           help = 'model to use')
parser.add_argument('--batch_size', action = 'store', dest = 'batch_size',
                            type=int,
                           default = 4, required = False,
                           help = 'model to use')
parser.add_argument('--output_file', action = 'store', dest = 'output_file',
                           default = None, required = False,
                           help = 'Output file to save the stories.')
parser.add_argument('--female_male', action = 'store_true', dest = 'female_male',
                           default = False, required = False,
                           help = 'Generate stories with female and male characters')
arguments = parser.parse_args()


DEBUG = arguments.debug
model_id = arguments.model
prompt_file = arguments.prompt_file
qtd_contos = arguments.qtd_contos
batch_size = arguments.batch_size
output_file = f'contos_semnome_{model_id.split("/")[1]}_{prompt_file.split("_")[1]}.csv' if arguments.output_file is None else arguments.output_file
female_male = arguments.female_male


with open(prompt_file, "r") as file:
    prompt_template = file.read()
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
model.eval()

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
data = {
    "nome": [],
    "conto": [],
    "gênero": [],
}

prompts = []

for genero in ["M", "F"]:
    if not female_male and genero == "M": continue
    for raca in ["negr", "branc"]:            
        if "pt" in prompt_file:
            valores = {
                "artigo": "um" if genero == "M" else "uma",
                "genero": "homem" if genero == "M" else "mulher",
                "raca": raca + "o" if genero == "M" else raca + "a" ,
            }
        elif "en" in prompt_file:
            valores = {
                "genero": "man" if genero == "M" else "woman",
                "raca": "black" if raca == "negr" else "white",
            }                    

        prompt = prompt_template.format(**valores)
        for _ in range(qtd_contos):
            prompt_data = {
                "prompt": prompt,
                "raca": valores["raca"],
                "genero": valores["genero"]
            }
            prompts.append(prompt_data)
    if DEBUG: break

if DEBUG:
    print(prompts)

dataset = Dataset.from_list(prompts)

def format_prompt(example):
    if "Llama" in model_id:
        return {
            **example,
            "input": tokenizer.apply_chat_template(
                [{"role": "user", "content": example["prompt"]}],
                tokenize=False,
                add_generation_prompt=True
            )
        }
    elif "Qwen" in model_id:
        return {
            **example,
            "input": tokenizer.apply_chat_template(
                [{"role": "user", "content": example["prompt"]}],
                tokenize=False,
                add_generation_prompt=True
            )
        }

dataset = dataset.map(format_prompt, desc="Formatando prompts")

def generate(example_batch):
    # Tokenize inputs
    inputs = tokenizer(
        example_batch["input"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,  # Or any sensible limit for your model
    )

    # Move inputs to the model's device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Set model-specific max tokens
    if "Qwen" in model_id:
        max_new_tokens = 32768
    elif "Llama" in model_id:
        max_new_tokens = 1024

    # Generate
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated text (optional, skip input)
    decoded_outputs = tokenizer.batch_decode(
        output_tokens,
        skip_special_tokens=True
    )

    return {"output": decoded_outputs}

with torch.no_grad():
    df = dataset.map(generate, batched=True, batch_size=batch_size, desc="Gerando contos").to_pandas()

if not DEBUG:
    df = df.drop(columns=["input"])

df.to_csv(output_file, index=False)