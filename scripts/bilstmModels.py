import json
import torch
import torch.nn as nn
from datasets import Dataset
from transformers.modeling_outputs import TokenClassifierOutput
import numpy as np
from tqdm import tqdm
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification, set_seed
) 

import time
from codecarbon import EmissionsTracker
from funciones import tokenize_and_align_labels, compute_metrics2, predictions_to_json

###### CLASE BILSTM

class BertBiLSTMNER(nn.Module):

    def __init__(
        self,
        model_name,
        num_labels,
        lstm_hidden_size=256,
        lstm_layers=1,
        dropout=0.1
    ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)

        # encoder pretrained
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.config.hidden_size

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        # classifier
        self.classifier = nn.Linear(
            lstm_hidden_size * 2,
            num_labels
        )

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        token_type_ids = None
    ):

        # transformer output
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs.last_hidden_state

        # BiLSTM
        lstm_output, _ = self.bilstm(sequence_output)

        lstm_output = self.dropout(lstm_output)

        # token logits
        logits = self.classifier(lstm_output)

        loss = None

        if labels is not None:
            loss = self.loss_fn(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1)
            )

        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

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
MODEL_NAME = model_names[4]
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000

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
print("Modelo: ", MODEL_NAME)
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}
####### LEER DATOS 
RUTA_TRAIN = "bio_train.json"
RUTA_DEV = "data/dev_chunked_bio.json"

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

def tokenize_and_align_labels(examples):

    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    labels = []
    label_all_tokens = True
    for i, word_labels in enumerate(examples["bio"]):

        word_ids = tokenized_inputs.word_ids(batch_index=i)

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:

            # special tokens
            if word_idx is None:
                label_ids.append(-100)

            # primer subtoken de palabra
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[word_labels[word_idx]])

            # subtokens siguientes
            else:
                if label_all_tokens:
                    label_ids.append(label2id[word_labels[word_idx]])
                else:
                    label_ids.append(-100)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    return tokenized_inputs




###### 1- CARGAR MODELOS Y TOKENIZER

model = BertBiLSTMNER(
    model_name=MODEL_NAME,
    num_labels=len(label2id)
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


###### 2- PROCESAR LOS DATOS

train_dataset = Dataset.from_list(train_data)
dev_dataset = Dataset.from_list(dev_data)
###### 3- TOKENIZAR LOS DATOS
tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_dev_dataset = dev_dataset.map(tokenize_and_align_labels, batched=True)


from collections import defaultdict
def remove_duplicated_token_entities_tuple(entities):
    seen = set()
    deduped = []

    for ent in entities:
        if ent not in seen:
            seen.add(ent)
            deduped.append(ent)

    return deduped
def remove_overlapping_token_entities_tuple(entities):
    """
    entities: list[(start_tok, end_tok, label)]
    end_tok inclusivo
    """

    sorted_entities = sorted(entities, key=lambda x: x[0])

    clusters = []
    cluster = []
    current_end = None

    for ent in sorted_entities:
        start_tok, end_tok, label = ent

        if not cluster:
            cluster = [ent]
            current_end = end_tok
        else:
            if start_tok <= current_end:
                cluster.append(ent)
                current_end = max(current_end, end_tok)
            else:
                clusters.append(cluster)
                cluster = [ent]
                current_end = end_tok

    if cluster:
        clusters.append(cluster)

    keepers = set()

    for clust in clusters:
        longest = max(
            clust,
            key=lambda x: x[1] - x[0] + 1
        )
        keepers.add(longest)

    output = []

    for ent in entities:
        if ent in keepers:
            output.append(ent)
            keepers.remove(ent)

    return output
def bio_to_entities(labels):
    entities = []
    start = None
    current_label = None

    for i, tag in enumerate(labels):
        if tag == "O":
            if current_label is not None:
                entities.append((start, i - 1, current_label))
                start = None
                current_label = None
            continue

        prefix, label = tag.split("-", 1)

        if prefix == "B":
            if current_label is not None:
                entities.append((start, i - 1, current_label))

            start = i
            current_label = label

        elif prefix == "I":
            if current_label != label:
                if current_label is not None:
                    entities.append((start, i - 1, current_label))

                start = i
                current_label = label

    if current_label is not None:
        entities.append((start, len(labels) - 1, current_label))

    return entities


def compute_metrics2(eval_pred):
    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)

    count_annotated_entities_per_label = defaultdict(int)
    count_predicted_entities_per_label = defaultdict(int)
    count_true_positives_per_label = defaultdict(int)

    for pred_seq, label_seq in zip(predictions, labels):
        true_preds = []
        true_labels = []

        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue

            true_preds.append(id2label[int(pred_id)])
            true_labels.append(id2label[int(label_id)])

        pred_entities = bio_to_entities(true_preds)
        gold_entities = bio_to_entities(true_labels)

        pred_entities = remove_duplicated_token_entities_tuple(pred_entities)
        pred_entities = remove_overlapping_token_entities_tuple(pred_entities)

        gold_entities_set = set(gold_entities)

        for entity in gold_entities:
            label = entity[2]
            count_annotated_entities_per_label[label] += 1

        for entity in pred_entities:
            label = entity[2]
            count_predicted_entities_per_label[label] += 1

            if entity in gold_entities_set:
                count_true_positives_per_label[label] += 1

    labels_present = list(count_annotated_entities_per_label.keys())

    count_annotated_entities = sum(
        count_annotated_entities_per_label[label]
        for label in labels_present
    )

    count_predicted_entities = sum(
        count_predicted_entities_per_label[label]
        for label in labels_present
    )

    count_true_positives = sum(
        count_true_positives_per_label[label]
        for label in labels_present
    )

    micro_precision = count_true_positives / (count_predicted_entities + 1e-10)
    micro_recall = count_true_positives / (count_annotated_entities + 1e-10)
    micro_f1 = 2 * (
        micro_precision * micro_recall /
        (micro_precision + micro_recall + 1e-10)
    )

    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0

    for label in labels_present:
        tp = count_true_positives_per_label[label]
        pred = count_predicted_entities_per_label[label]
        gold = count_annotated_entities_per_label[label]

        precision = tp / (pred + 1e-10)
        recall = tp / (gold + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

    n = len(labels_present)

    if n > 0:
        macro_precision /= n
        macro_recall /= n
        macro_f1 /= n

    return {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_macro": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "f1": micro_f1,
    }
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

#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
###### 4- CONFIGURAR EL ENTRENAMIENTO
training_args = TrainingArguments(
    output_dir="./results5_bi",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=10,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
    run_name = "21-5-mingxiaw-bi-v1",
    save_total_limit=2
)
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, label_pad_token_id=-100)



###### 5- ENTRENAR EL MODELO
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_dev_dataset,
    data_collator=data_collator,
    compute_metrics = compute_metrics2
)

for name, param in model.named_parameters():
    if not param.is_contiguous():
        param.data = param.data.contiguous()

for name, buffer in model.named_buffers():
    if not buffer.is_contiguous():
        buffer.data = buffer.data.contiguous()
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
#0000000000000000000000000000000000000000000000000000000000000000000000000000000000000
tracker_train = EmissionsTracker(
    project_name="ner_train_lstm_electra",
    output_dir="./emissions_lstm",
    output_file="train_lstm.csv"
)
tracker_train.start()
trainer.train()
emissions_train = tracker_train.stop()
trainer.save_model("./best_model_electra_bi")
tokenizer.save_pretrained("./best_model_electra_bi")
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
for example in tqdm(dev):
    pmid = example["pmid"]
    chunks = example["data"]
    for chunk in chunks:
        field = chunk["location"]
        tokens = chunk["chunk_tokens"]

        # tokenización
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
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
            "pmid":pmid,
            "location": field,
            "tokens": tokens,
            "predictions": word_predictions
        })

# with open("predictions/13-06/21_mingxiaw_bi.json", "w") as file:
# 	json.dump(all_predictions, file, indent=2)

print("INFERENCIA HECHA, PREDICCIONES GUARDADAS")
print()
print("RECONSTRUYENDO ENTIDADES")
with open("test_chunked_reconstruct.json", "r")as file:
    data = json.load(file)

from collections import OrderedDict


def bio_to_spans(tokens, labels, spans):
    entities = []

    current_tokens = []
    current_spans = []
    current_label = None

    for token, bio_label, span in zip(tokens, labels, spans):

        if bio_label == "O":
            if current_label is not None:
                entities.append(build_entity(
                    current_tokens,
                    current_spans,
                    current_label
                ))

            current_tokens = []
            current_spans = []
            current_label = None
            continue

        prefix, label = bio_label.split("-", 1)

        if prefix == "B":
            if current_label is not None:
                entities.append(build_entity(
                    current_tokens,
                    current_spans,
                    current_label
                ))

            current_tokens = [token]
            current_spans = [span]
            current_label = label

        elif prefix == "I":

            if current_label == label:
                current_tokens.append(token)
                current_spans.append(span)

            else:
                if current_label is not None:
                    entities.append(build_entity(
                        current_tokens,
                        current_spans,
                        current_label
                    ))

                current_tokens = [token]
                current_spans = [span]
                current_label = label

    if current_label is not None:
        entities.append(build_entity(
            current_tokens,
            current_spans,
            current_label
        ))

    return entities
def reconstruct_text(tokens, spans):
    if not tokens:
        return ""

    text = tokens[0]

    for i in range(1, len(tokens)):
        prev_end = spans[i - 1][1]
        curr_start = spans[i][0]

        n_spaces = curr_start - prev_end

        if n_spaces > 0:
            text += " " * n_spaces

        text += tokens[i]

    return text

def build_entity(tokens, spans, label):

    start_idx = spans[0][0]
    end_idx = spans[-1][1] - 1

    text_span = reconstruct_text(tokens, spans)

    return {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "text_span": text_span,
        "label": label
    }


def predictions_to_json(gold_data, pred_data):

    gold_index = {}

    for item in gold_data:

        key = (
            item["pmid"],
            item["location"],
            tuple(item["chunk_tokens"])
        )

        gold_index[key] = item["chunk_spans"]

    result = OrderedDict()

    for item in pred_data:

        pmid = item["pmid"]
        location = item["location"]
        tokens = item["tokens"]
        predictions = item["predictions"]

        key = (
            pmid,
            location,
            tuple(tokens)
        )

        if key not in gold_index:
            print(f"Missing spans for PMID {pmid}")
            continue

        spans = gold_index[key]

        entities = bio_to_spans(
            tokens,
            predictions,
            spans
        )

        if pmid not in result:
            result[pmid] = {
                "entities": []
            }

        for ent in entities:
            ent["location"] = location
            result[pmid]["entities"].append(ent)

    return result
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
with open("predictions/13-06/electra_lstm.json", "w", encoding="utf-8") as f:
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
print(
    "\n"
    "========== RESUMEN DE EFICIENCIA Y EMISIONES ==========\n"
    f"Emisiones entrenamiento: {emissions_train:.6f} kg CO2eq | {emissions_train * 1000:.3f} g CO2eq\n"
    "=======================================================\n"
)