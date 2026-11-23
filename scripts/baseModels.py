import json
import torch
from datasets import Dataset
import numpy as np
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
# from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification, set_seed
)
import time
from codecarbon import EmissionsTracker
from funciones import tokenize_and_align_labels, compute_metrics2, predictions_to_json
SEED = 42


np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

set_seed(SEED)

####### MODELOS
model_names = ["microsoft/deberta-v3-large",
               "michiyasunaga/BioLinkBERT-large",
               "mingxiaw/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
               "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
               "microsoft/BiomedNLP-BiomedELECTRA-large-uncased-abstract"]

#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
MODEL_NAME = model_names[4]
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
print("Modelo: ", MODEL_NAME)
# label_list = ['dietary supplement', 'chemical', 'animal', 'ddf',
        # 'anatomical location', 'gene', 'bacteria', 'microbiome', 'human', 'biomedical technique', 'statistical technique', 'food', 'drug', 'O']
label_list = [
    'O',
    'B-dietary supplement', 'I-dietary supplement',
    'B-chemical', 'I-chemical',
    'B-animal', 'I-animal',
    'B-ddf', 'I-ddf',
    'B-anatomical location', 'I-anatomical location',
    'B-gene', 'I-gene',
    'B-bacteria', 'I-bacteria',
    'B-microbiome', 'I-microbiome',
    'B-human', 'I-human',
    'B-biomedical technique', 'I-biomedical technique',
    'B-statistical technique', 'I-statistical technique',
    'B-food', 'I-food',
    'B-drug', 'I-drug',
    'O'
]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

####### LEER DATOS 
RUTA_TRAIN = "bio_train.json"
RUTA_DEV = "dev_chunked_bio_sp.json"

with open(RUTA_TRAIN, "r", encoding='utf-8') as f:
    train_data = json.load(f)
with open(RUTA_DEV, "r", encoding='utf-8') as f:
    dev_data = json.load(f)

###### 1- CARGAR MODELOS Y TOKENIZER

model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(label_list))
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


###### 2- PROCESAR LOS DATOS

train_dataset = Dataset.from_list(train_data)
dev_dataset = Dataset.from_list(dev_data)
###### 3- TOKENIZAR LOS DATOS
tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer, "label2id":label2id})
tokenized_dev_dataset = dev_dataset.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer, "label2id":label2id})

print("####  DATOS PROCESADOS  ####")
from scipy.special import softmax

import numpy as np
from collections import defaultdict





###### 4- CONFIGURAR EL ENTRENAMIENTO
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
training_args = TrainingArguments(
    output_dir="./results3",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    seed = SEED,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
    run_name="def_electra_base_13",
    fp16=True, save_total_limit=2
)
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, label_pad_token_id=-100)

from functools import partial
compute_metrics_fn = partial(
    compute_metrics2,
    id2label=id2label
)
###### 5- ENTRENAR EL MODELO
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_dev_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics_fn
)
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
tracker_train = EmissionsTracker(
    project_name="ner_train_base_electra",
    output_dir="./emissions_base",
    output_file="train_base_electra.csv"
)
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
start_train = time.time()
tracker_train.start()
trainer.train()
emissions_train = tracker_train.stop()
end_train = time.time()
train_time = round(end_train - start_train,2)
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
trainer.save_model("./best_model_electra_def")
tokenizer.save_pretrained("./best_model_electra_def")
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000

print("ENTRENAMIENTO TERMINADO")
print()
print("COMIENZA INFERENCIA")

with open("test_chunked2.json", "r", encoding='utf-8') as f:
    dev = json.load(f)

all_predictions = []
model = trainer.model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
tracker_inference = EmissionsTracker(
    project_name="ner_inferencia_base_electra",
    output_dir="./emissions_base",
    output_file="inference_base_electra.csv"
)
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
start_inf = time.time()
tracker_inference.start()
for example in tqdm(dev):

    pmid = example["pmid"]
    chunks = example["data"]

    for chunk in chunks:

        field = chunk["location"]  # "title" o "abstract"
        tokens = chunk["chunk_tokens"]

        # tokenización
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids(batch_index=0)
        inputs = {k: v.to(device) for k, v in encoding.items()}

        # forward
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits

        # predicción
        predictions = torch.argmax(logits, dim=-1)[0]

        # mapping subtokens -> palabras
        word_predictions = []
        previous_word_idx = None

        for pred, word_idx in zip(predictions, word_ids):

            # saltar special tokens
            if word_idx is None:
                continue

            # quedarse solo con primer subtoken
            if word_idx != previous_word_idx:

                label = id2label[pred.item()]
                word_predictions.append(label)

            previous_word_idx = word_idx

        all_predictions.append({
            "pmid": pmid,
            "location": field,
            "tokens": tokens,
            "predictions": word_predictions
        })

emissions_inference = tracker_inference.stop()
end_inf = time.time()
time_inf = round(end_inf - start_inf, 2)

print("INFERENCIA HECHA, PREDICCIONES GUARDADAS")
print()
print("RECONSTRUYENDO ENTIDADES")
with open("test_chunked_reconstruct.json", "r")as file:
    data = json.load(file)



output_json = predictions_to_json(data, all_predictions)

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

output_json = replace_label(output_json)

#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
with open("predictions/13-06/electra_base.json", "w", encoding="utf-8") as f:
    json.dump(output_json, f, ensure_ascii=False, indent=2)
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
with open("dev.json", "r")as file:
    ground_truth = json.load(file)
claves_validas = output_json.keys()

ground_truth = {
    k: v
    for k, v in ground_truth.items()
    if k in claves_validas
}
from evaluate import eval_submission_NER
precision, recall, f1, micro_precision, micro_recall, micro_f1 = eval_submission_NER(output_json, ground_truth)
round_to_decimal_position = 4
print("\n\n=== NER ===")
print(f"Macro-precision: {round(precision, round_to_decimal_position)}")
print(f"Macro-recall: {round(recall, round_to_decimal_position)}")
print(f"Macro-F1: {round(f1, round_to_decimal_position)}")
print(f"Micro-precision: {round(micro_precision, round_to_decimal_position)}")
print(f"Micro-recall: {round(micro_recall, round_to_decimal_position)}")
print(f"Micro-F1: {round(micro_f1, round_to_decimal_position)}")


total_time = train_time + time_inf
total_emissions = emissions_train + emissions_inference

print(
    "\n"
    "========== RESUMEN DE EFICIENCIA Y EMISIONES ==========\n"
    f"Tiempo de entrenamiento: {train_time:.2f} s | {train_time / 60:.2f} min\n"
    f"Tiempo de inferencia:    {time_inf:.2f} s | {time_inf / 60:.2f} min\n"
    f"Tiempo total:           {total_time:.2f} s | {total_time / 60:.2f} min\n"
    "-------------------------------------------------------\n"
    f"Emisiones entrenamiento: {emissions_train:.6f} kg CO2eq | {emissions_train * 1000:.3f} g CO2eq\n"
    f"Emisiones inferencia:    {emissions_inference:.6f} kg CO2eq | {emissions_inference * 1000:.3f} g CO2eq\n"
    f"Emisiones totales:       {total_emissions:.6f} kg CO2eq | {total_emissions * 1000:.3f} g CO2eq\n"
    "=======================================================\n"
)