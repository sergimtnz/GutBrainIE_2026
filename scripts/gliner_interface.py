import json
from gliner import GLiNER

import torch
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import os

from codecarbon import EmissionsTracker
import time
from importlib.metadata import version
version('GLiNER')
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/deberta-v3-large",
    use_fast=True
)
# Set the GLiNER model to be used (from HuggingFace)
model = GLiNER.from_pretrained("numind/NuNerZero")
model_name = "NuNerZero"

# Define the confidence threshold to be used in evaluation
THRESHOLD = 0.7 

# Define whether the code should be used for fine-tuning
finetune_model = True

# Define the path to articles for which the final trained will generate predicted entities
generate_predictions = True
PATH_ARTICLES = "./Train/NER/data/raw_test.json" 
PATH_OUTPUT_NER_PREDICTIONS = "./inferencia_test2.json"

print('## LOADING TRAINING DATA ##')
# PATH_GOLD_TRAIN = "data/train_gold.json"
# PATH_SILVER_TRAIN = "data/train_silver.json"
# PATH_SILVER_TRAIN_2025 = "data/train_silver_2025.json"
# PATH_BRONZE_TRAIN = "data/train_bronze.json"
# PATH_DEV = "data/dev.json"
PATH_TRAIN = "Train/NER/data/train_data_full.json"
PATH_DEV = "Train/NER/data/processed_val_chunked.json"

with open(PATH_TRAIN, 'r', encoding='utf-8') as file:
	train_data = json.load(file)


with open(PATH_DEV, 'r', encoding='utf-8') as file:
	eval_data = json.load(file)

# Set the data to be used for training


# converting entities-level train data to token-level 
new_data = []
for d in train_data:
    new_ner = []
    for s, f, c in d["ner"]:
        for i in range(s, f + 1):
            # labels are intended to be lower-case
            new_ner.append((i, i, c.lower()))
    new_d = {
        "tokenized_text": d["tokenized_text"],
        "ner": new_ner,
    }
    new_data.append(new_d)
train_data = new_data

# converting entities-level eval data to token-level 
new_data = []
for d in eval_data:
    new_ner = []
    for s, f, c in d["ner"]:
        for i in range(s, f + 1):
            # labels are intended to be lower-case
            new_ner.append((i, i, c.lower()))
    new_d = {
        "tokenized_text": d["tokenized_text"],
        "ner": new_ner,
    }
    new_data.append(new_d)
eval_data = new_data

from types import SimpleNamespace

# Define the hyperparameters in a config variable
config = SimpleNamespace(
    num_steps=6000, # regulate number train, eval steps depending on the data size
    eval_every=400,
    
    train_batch_size=4, # regulate batch size depending on GPU memory available.
    
    max_len=384, # maximum sentence length. 2048 for NuNerZero_long_context
    
    save_directory="logs", # log dir
    device='cuda' if torch.cuda.is_available() else 'cpu', #'cuda', # training device - cpu or cuda

    warmup_ratio=0.1, # Other parameters
    lr_encoder=1e-5,
    lr_others=5e-5,
    freeze_token_rep=False,

    max_types=15,
    shuffle_types=True,
    random_drop=True,
    max_neg_type_ratio=1,
)

# modify this to your own test data!
# don't forget to do the same preprocessing as for the train data:
# * converting entities-level data to token-level data
# * making entity_types lower-cased!!!
eval_data = {
    "entity_types": [
        "anatomical location",
        "animal",
        "biomedical technique",
        "bacteria",
        "chemical",
        "dietary supplement",
        "ddf",
        "drug",
        "food",
        "gene",
        "human",
        "microbiome",
        "statistical technique",
    ],
    "samples": eval_data
}

print('## DEFINING TRAINING FUNCTION ##') 
def train(model, config, train_data, eval_data=None):
    model = model.to(config.device)

    # Set sampling parameters from config
    model.set_sampling_params(
        max_types=config.max_types,
        shuffle_types=config.shuffle_types,
        random_drop=config.random_drop,
        max_neg_type_ratio=config.max_neg_type_ratio,
        max_len=config.max_len
    )

    model.train()

    # Initialize data loaders
    train_loader = model.create_dataloader(train_data, batch_size=config.train_batch_size, shuffle=True)

    # Optimizer
    optimizer = model.get_optimizer(config.lr_encoder, config.lr_others, config.freeze_token_rep)

    pbar = tqdm(range(config.num_steps))

    if config.warmup_ratio < 1:
        num_warmup_steps = int(config.num_steps * config.warmup_ratio)
    else:
        num_warmup_steps = int(config.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=config.num_steps
    )

    iter_train_loader = iter(train_loader)

    best_f1 = -1
    best_step = None
    best_model_path = f"{config.save_directory}/best_model"

    for step in pbar:
        try:
            x = next(iter_train_loader)
        except StopIteration:
            iter_train_loader = iter(train_loader)
            x = next(iter_train_loader)

        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                x[k] = v.to(config.device)

        loss = model(x)  # Forward pass

        # Check if loss is nan
        if torch.isnan(loss):
            continue

        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters
        scheduler.step()  # Update learning rate schedule
        optimizer.zero_grad()  # Reset gradients

        description = f"step: {step} | epoch: {step // len(train_loader)} | loss: {loss.item():.2f} | best_f1: {best_f1:.4f}"
        pbar.set_description(description)

        if (step + 1) % config.eval_every == 0:

            model.eval()

            if eval_data is not None:
                results, f1 = model.evaluate(eval_data["samples"], flat_ner=True, threshold=THRESHOLD, batch_size=32,
                                     entity_types=eval_data["entity_types"])

                print(f"Step={step}\n{results}")
                print(f"F1 actual: {f1:.4f}")
                print(f"Mejor F1 hasta ahora: {best_f1:.4f}")

                if f1 > best_f1:
                    best_f1 = f1
                    best_step = step

                    if not os.path.exists(config.save_directory):
                        os.makedirs(config.save_directory)

                    model.save_pretrained(best_model_path)

                    print(
                        f"Nuevo mejor modelo guardado en {best_model_path} "
                        f"con F1={best_f1:.4f} en step={best_step}"
                    )
                

            if not os.path.exists(config.save_directory):
                os.makedirs(config.save_directory)

            model.save_pretrained(f"{config.save_directory}/finetuned_{step}")

            model.train()

    print("\n========== FIN DEL ENTRENAMIENTO ==========")
    print(f"Mejor F1: {best_f1:.4f}")
    print(f"Mejor step: {best_step}")
    print(f"Modelo guardado en: {best_model_path}")

if finetune_model:
    print('## LAUNCHING TRAINING ##')
    tracker_train = EmissionsTracker(
        project_name=f"{model_name}_training",
        output_dir="./emissions_gliner",
        output_file=f"emissions_train_{model_name}.csv",
        measure_power_secs=60,
        log_level="error"
    )

    start_train = time.time()

    tracker_train.start()

    train(model, config, train_data, eval_data)

    emissions_train = tracker_train.stop()

    end_train = time.time()
    train_time = end_train - start_train

    print('## SAVING TRAINED MODEL ##')
    output_path = f"outputs/{model_name}_finetuned_T{str(THRESHOLD*100)}"
    model.save_pretrained(output_path, safe_serialization=True)
    os.system(f'cp {output_path}/gliner_config.json {output_path}/config.json')
    md = GLiNER.from_pretrained(output_path)

def chunk_raw_text(text, tokenizer, max_tokens=384, overlap=50):
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False
    )

    offsets = encoding["offset_mapping"]
    chunks = []
    step = max_tokens - overlap

    for start_tok in range(0, len(offsets), step):
        end_tok = min(start_tok + max_tokens, len(offsets))

        chunk_offsets = offsets[start_tok:end_tok]
        chunk_offsets = [
            offset for offset in chunk_offsets
            if offset[0] != offset[1]
        ]

        if not chunk_offsets:
            continue

        start_char = chunk_offsets[0][0]
        end_char = chunk_offsets[-1][1]

        chunk_text = text[start_char:end_char]

        chunks.append({
            "text": chunk_text,
            "start_char": start_char,
            "end_char": end_char,
            "start_token": start_tok,
            "end_token": end_tok
        })

        if end_tok >= len(offsets):
            break

    return chunks
def predict_entities_chunked(
    text,
    model,
    tokenizer,
    entity_labels,
    threshold=0.7,
    max_tokens=384,
    overlap=50,
    global_offset=0
):
    all_entities = []

    chunks = chunk_raw_text(
        text=text,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        overlap=overlap
    )

    for chunk in chunks:
        chunk_text = chunk["text"]

        chunk_entities = model.predict_entities(
            chunk_text,
            entity_labels,
            threshold=threshold,
            flat_ner=True,
            multi_label=False
        )

        for entity in chunk_entities:
            entity_start = entity["start"] + chunk["start_char"] + global_offset
            entity_end = entity["end"] + chunk["start_char"] + global_offset

            all_entities.append({
                "start": entity_start,
                "end": entity_end,
                "text": entity["text"],
                "label": entity["label"],
                "score": entity["score"]
            })

    return all_entities
if generate_predictions:
    output_path = f"logs/best_model"
    print(f"## LOADING PRE-TRAINED MODEL {output_path} ##")
    md = GLiNER.from_pretrained(output_path, local_files_only=False)
    print(f"## GENERATING NER PREDICTIONS FOR {PATH_ARTICLES}")
    with open(PATH_ARTICLES, 'r', encoding='utf-8') as file:
        articles = json.load(file)

    print(f"len(articles): {len(articles)}")
    entity_labels = eval_data['entity_types']

    # Dictionary to hold predicted entities
    # PMID -> {{'start_idx': ..., 'end_idx': ..., 'text_span': ..., 'entity_label': ..., 'score': ...}, ...}
    predictions = {} 

    for pmid, content in tqdm(articles.items(), total=len(articles), desc="Predicting entities..."):
        title = content["metadata"]['title']
        abstract = content["metadata"]['abstract']

        # Predict entities 
        title_entities = md.predict_entities(title, entity_labels, threshold=THRESHOLD, flat_ner=True, multi_label=False)
        abstract_entities = predict_entities_chunked(text=abstract,
                                                     model=md,
                                                     tokenizer=tokenizer,
                                                     entity_labels=entity_labels,
                                                     threshold = THRESHOLD,
                                                     max_tokens = 384,
                                                     overlap = 50,
                                                     global_offset = 0)
                                                     

        # Adjust indices for predicted entities in the abstract

        # Remove duplicates from predicted entities
        unique_entities = []
        seen_entities = set()

        # Remove duplicates from title entities and add tag field with value 't'
        for entity in title_entities:
            key = (entity['start'], entity['end'], entity['text'], entity['label'], entity['score'])
            if key not in seen_entities:
                tmp_entity = {
                    'start_idx': entity['start'],
                    'end_idx': entity['end'],
                    'location': 'title',
                    'text_span': entity['text'],
                    'label': entity['label'] if entity["label"] != "ddf" else entity["label"].upper(),
                    'score': entity['score']
                }
                unique_entities.append(tmp_entity)
                seen_entities.add(key)

        # Remove duplicates from abstract entities and add tag field with value 'a'
        for entity in abstract_entities:
            key = (entity['start'], entity['end'], entity['text'], entity['label'], entity['score'])
            if key not in seen_entities:
                tmp_entity = {
                    'start_idx': entity['start'],
                    'end_idx': entity['end'],
                    'location': 'abstract',
                    'text_span': entity['text'],
                    'label': entity['label'] if entity["label"] != "ddf" else entity["label"].upper(),
                    'score': entity['score']
                }
                unique_entities.append(tmp_entity)
                seen_entities.add(key)

        predictions[pmid] = {"entities":unique_entities}
        articles[pmid]['pred_entities'] = unique_entities

    # Convert any non-serializable data if necessary
    def default_serializer(obj):
        if isinstance(obj, set):
            return list(obj)
        # Add other types if needed
        raise TypeError(f'Type {type(obj)} not serializable')

    def replace_label(data):
        if isinstance(data, dict):
            for key, value in data.items():
                # Cambiar valor si es label
                if key in ["entity_label", "label", "tag"] and value == "ddf":
                    data[key] = "DDF"
                else:
                    replace_label(value)
    
        elif isinstance(data, list):
            for item in data:
                replace_label(item)
    
        return data
from Train.NER.funciones import merge_consecutive_predictions, adjust_predicted_indices
merge_consecutive_predictions(predictions)
adjust_predicted_indices(predictions)
with open(PATH_OUTPUT_NER_PREDICTIONS, 'w', encoding='utf-8') as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2, default=default_serializer)

    print(f"## Predictions have been exported in JSON format to '/{PATH_OUTPUT_NER_PREDICTIONS}' ##")

from Train.NER.evaluate import eval_submission_NER
with open("Train/NER/dev.json", "r")as file:
    ground_truth = json.load(file)

claves_validas = predictions.keys()

ground_truth = {
    k: v
    for k, v in ground_truth.items()
    if k in claves_validas
}
    
precision, recall, f1, micro_precision, micro_recall, micro_f1 = eval_submission_NER(predictions, ground_truth)
round_to_decimal_position = 4
print("\n\n=== NER ===")
print(f"Macro-precision: {round(precision, round_to_decimal_position)}")
print(f"Macro-recall: {round(recall, round_to_decimal_position)}")
print(f"Macro-F1: {round(f1, round_to_decimal_position)}")
print(f"Micro-precision: {round(micro_precision, round_to_decimal_position)}")
print(f"Micro-recall: {round(micro_recall, round_to_decimal_position)}")
print(f"Micro-F1: {round(micro_f1, round_to_decimal_position)}")
print(
    "\n"
    "========== RESUMEN ENTRENAMIENTO ==========\n"
    f"Tiempo de entrenamiento: {train_time:.2f} s | {train_time / 60:.2f} min\n"
    f"Emisiones entrenamiento: {emissions_train:.6f} kg CO2eq | {emissions_train * 1000:.3f} g CO2eq\n"
    "===========================================\n"
)