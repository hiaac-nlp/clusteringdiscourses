from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
from tqdm import tqdm
from names_dataset import NameDataset
import argparse
from datasets import Dataset
import torch
import pycountry
import pycountry_convert as pc

parser = argparse.ArgumentParser(description="Gerador de contos")

parser.add_argument('--debug', action = 'store_true', dest = 'debug',
                           default = False, required = False,
                           help = 'Debug logs')
parser.add_argument('--prompt', action = 'store', dest = 'prompt_file',
                           default = False, required = False,
                           help = 'File with prompt to use')
parser.add_argument('--model', action = 'store', dest = 'model',
                           default = None, required = False,
                           help = 'model to use')
parser.add_argument('--n_stories', action = 'store', dest = 'n_stories',
                           default = 1, required = False, type=int,
                           help = 'Number of stories created for each prompt')
parser.add_argument('--top_names', action = 'store', dest = 'top_names',
                            type=int,
                           default = 2, required = False,
                           help = 'Top numbers from the names dataset library to use')
parser.add_argument('--batch_size', action = 'store', dest = 'batch_size',
                            type=int,
                           default = 4, required = False,
                           help = 'batch size for generation')
parser.add_argument('--female_male', action = 'store_true', dest = 'female_male',
                           default = False, required = False,
                           help = 'Generate stories with female and male characters')
arguments = parser.parse_args()


nd = NameDataset()

DEBUG = arguments.debug
model_id = arguments.model
prompt_file = arguments.prompt_file
n_stories = arguments.n_stories
top_names = arguments.top_names
names = nd.get_top_names(n=top_names)
batch_size = arguments.batch_size
female_male = arguments.female_male

def country_to_continent(country_name):
    country_alpha2 = pc.country_name_to_country_alpha2(country_name)
    country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
    country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    return country_continent_name

with open(prompt_file, "r") as file:
    prompt_template = file.read()
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
model.eval()

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
data = {
    "nome": [],
    "país": [],
    "conto": [],
    "gênero": [],
}

prompts = []

for pais in tqdm(names.keys(), desc="Gerando dados."):
    for genero in names[pais].keys():
        if not female_male and genero == "M": continue
        for raca in ["negr", "branc"]:
            for nome in names[pais][genero]:
                
                if "pt" in prompt_file:
                    valores = {
                        "artigo": "um" if genero == "M" else "uma",
                        "genero": "homem" if genero == "M" else "mulher",
                        "raca": raca + "o" if genero == "M" else raca + "a" ,
                        "adjetivo": "chamado" if genero == "M" else "chamada",
                        "nome": nome,
                    }
                elif "en" in prompt_file:
                    valores = {
                        "genero": "man" if genero == "M" else "woman",
                        "raca": "black" if raca == "negr" else "white",
                        "nome": nome,
                    }                    

                prompt = prompt_template.format(**valores)
                for _ in range(n_stories):
                    prompt_data = {
                        "prompt": prompt,
                        "nome": nome,
                        "pais_alpha": pais,
                        "pais": pycountry.countries.get(alpha_2=pais).name,
                        "continente": country_to_continent(pycountry.countries.get(alpha_2=pais).name),
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
    
    if "Llama" in model_id:
        outputs = pipe(
            example_batch["input"],
            max_new_tokens=1024,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )
    elif "Qwen" in model_id:
        outputs = pipe(
        example_batch["input"],
        max_new_tokens=32768,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False
    )

    if isinstance(outputs[0], list):
        outputs = [item for sublist in outputs for item in sublist]

    return {"output": [res["generated_text"] for res in outputs]}

with torch.no_grad():
    df = dataset.map(generate, batched=True, batch_size=batch_size, desc="Gerando contos").to_pandas()

if not DEBUG:
    df = df.drop(columns=["input"])
df.to_csv(f'contos_{model_id.split("/")[1]}_{prompt_file.split("_")[1]}.csv', index=False)