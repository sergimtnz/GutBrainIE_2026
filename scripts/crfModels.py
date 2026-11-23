import json
import torch
import torch.nn as nn
from torchcrf import CRF
from datasets import Dataset
import numpy as np
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from tqdm import tqdm
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
) 

import time 
from codecarbon import EmissionsTracker
from funciones import tokenize_and_align_labels, compute_metrics2, predictions_to_json

###### CLASE CRF

class DebertaCRF(nn.Module):
    def __init__(self, model_name, num_labels, id2label, label2id):
        super().__init__()

        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(hidden_size, num_labels)

        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        x = self.dropout(outputs.last_hidden_state)

        emissions = self.classifier(x)

        loss = None
        if labels is not None:
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = 0

            mask = attention_mask.bool()

            loss = -self.crf(
                emissions,
                crf_labels,
                mask=mask,
                reduction="mean"
            )

        return {"loss": loss, "logits": emissions}

    def decode(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        x = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(x)

        return self.crf.decode(emissions, mask=attention_mask.bool())


####### MODELOS
model_names = ["microsoft/deberta-v3-large",
               "michiyasunaga/BioLinkBERT-large",
               "mingxiaw/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
               "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
               "microsoft/BiomedNLP-BiomedELECTRA-large-uncased-abstract"]

#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
MODEL_NAME = model_names[4]
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000

print(MODEL_NAME)
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
    'B-drug', 'I-drug'
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

def build_token_labels(example):
    tokens = example["tokenized_text"]
    ner = example["ner"]

    labels = ["O"] * len(tokens)

    for start, end, label in ner:

        # end es inclusivo
        for idx in range(start, end + 1):
            labels[idx] = label

    return labels
# for example in train_data:
#     example["labels"] = build_token_labels(example)
#     del example["ner"]

# for example in dev_data:
#     example["labels"] = build_token_labels(example)
#     del example["ner"]





###### 1- CARGAR MODELOS Y TOKENIZER

model = DebertaCRF(
    model_name=MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

from scipy.special import softmax
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]

    probs = softmax(logits, axis=-1)
    predictions = np.argmax(logits, axis=-1)
    max_probs = np.max(probs, axis=-1)

    # Threshold
    threshold = 0.65

    # Si confianza < threshold -> O
    # predictions[max_probs < threshold] = label2id["O"]

    true_predictions = []
    true_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        pred_tags, gold_tags = [], []
        for pred_id, label_id in zip(pred_seq, label_seq):

            label_id = int(label_id)

            if label_id == -100:
                continue

            pred_tags.append(id2label[int(pred_id)])
            gold_tags.append(id2label[label_id])

        true_predictions.append(pred_tags)
        true_labels.append(gold_tags)
    """
    entity_label_ids = [
        idx
        for label, idx in label2id.items()
        if label != "O"
    ]

    return {
        "f1_micro": f1_score(true_labels, true_predictions, labels=entity_label_ids, average="micro", zero_division=0),
        "f1_macro": f1_score(true_labels, true_predictions, labels=entity_label_ids, average="macro", zero_division=0),
        "precision_macro": precision_score(true_labels, true_predictions, labels=entity_label_ids, average="macro", zero_division=0),
        "recall_macro": recall_score(true_labels, true_predictions, labels=entity_label_ids, average="macro", zero_division=0),
        "precision_micro": precision_score(true_labels, true_predictions, labels=entity_label_ids, average="micro", zero_division=0),
        "recall_micro": recall_score(true_labels, true_predictions, labels=entity_label_ids, average="micro", zero_division=0),
    }
    
    """
    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)
    accuracy = accuracy_score(true_labels, true_predictions)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }

###### 2- PROCESAR LOS DATOS

train_dataset = Dataset.from_list(train_data)
dev_dataset = Dataset.from_list(dev_data)
print("Va bien hasta aquí")
###### 3- TOKENIZAR LOS DATOS
tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer, "label2id":label2id})
tokenized_dev_dataset = dev_dataset.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer, "label2id":label2id})


###### 4- CONFIGURAR EL ENTRENAMIENTO
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
training_args = TrainingArguments(
    output_dir="./results_crf",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
    run_name = "def_electra_crf_13",
    save_total_limit = 2,
    save_safetensors=False
)
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, label_pad_token_id=-100)

from functools import partial
compute_metrics_fn = partial(
    compute_metrics2,
    id2label=id2label)
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
    project_name="ner_train_crf_electra",
    output_dir = "./emissions_crf",
    output_file = "train_crf_electra.csv"
)
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000

start_train = time.time()
tracker_train.start()
trainer.train()
emissions_train = tracker_train.stop()
end_train = time.time()
train_time = round(end_train - start_train, 2)
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
trainer.save_model("./best_model_electra_crf")
tokenizer.save_pretrained("./best_model_electra_crf")
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000

model = trainer.model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
tracker_inference = EmissionsTracker(
    project_name = "ner_inferencia_crf_electra",
    output_dir = "./emissions_crf",
    output_file = "inference_crf_electra.csv"
)
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
start_inf = time.time()
tracker_inference.start()


all_predictions = []
with open("test_chunked2.json", "r", encoding='utf-8') as f:
    dev = json.load(f)
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
            predictions = model.decode(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )[0]

        # mapping subtokens -> palabras
        word_predictions = []
        previous_word_idx = None

        for pred, word_idx in zip(predictions, word_ids):

            # saltar special tokens
            if word_idx is None:
                continue

            # quedarse solo con primer subtoken
            if word_idx != previous_word_idx:

                label = id2label[pred]
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
# -------------------
# preview
# -------------------



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
with open("predictions/13-06/electra_crf.json", "w", encoding="utf-8") as f:
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