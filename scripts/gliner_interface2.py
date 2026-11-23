import json
import os
from copy import deepcopy
from types import SimpleNamespace
from importlib.metadata import version

import torch


from gliner import GLiNER
# =========================================================
# Configuración general
# =========================================================

model = GLiNER.from_pretrained("numind/NuNerZero")
model_name = "NuNerZero"

THRESHOLD = 0.75
finetune_model = True

generate_predictions = True
PATH_ARTICLES = "./articles_dev.json"
PATH_OUTPUT_NER_PREDICTIONS = "./inferencia.json"

print("## LOADING TRAINING DATA ##")
# PATH_GOLD_TRAIN = "./Train/NER/data/train_gold.json"
# PATH_SILVER_TRAIN = "./Train/NER/data/train_silver.json"
# PATH_SILVER_TRAIN_2025 = "./Train/NER/data/train_silver_2025.json"
# PATH_BRONZE_TRAIN = "./Train/NER/data/train_bronze.json"
# PATH_DEV = "./Train/NER/dev.json"

# with open(PATH_GOLD_TRAIN, "r", encoding="utf-8") as file:
#     train_gold = json.load(file)

# with open(PATH_SILVER_TRAIN, "r", encoding="utf-8") as file:
#     train_silver = json.load(file)

# with open(PATH_SILVER_TRAIN_2025, "r", encoding="utf-8") as file:
#     train_silver_2025 = json.load(file)

# # with open(PATH_BRONZE_TRAIN, "r", encoding="utf-8") as file:
# #     train_bronze = json.load(file)
PATH_TRAIN = "./Train/NER/data/train_data_full.json"
PATH_DEV = "./Train/NER/data/chunked_dev.json"
with open(PATH_TRAIN, "r", encoding="utf-8") as file:
    train_raw = json.load(file)
with open(PATH_DEV, "r", encoding="utf-8") as file:
    dev_raw = json.load(file)

# # Aquí eliges qué usar para entrenar
# train_raw = train_gold + train_silver + train_silver_2025


ENTITY_TYPES = [
    "anatomical location",
    "animal",
    "biomedical technique",
    "bacteria",
    "chemical",
    "dietary supplement",
    "DDF",
    "drug",
    "food",
    "gene",
    "human",
    "microbiome",
    "statistical technique",
]

config = SimpleNamespace(
    num_steps=5000,
    eval_every=2500,
    train_batch_size=4,
    max_len=512,  # en esta versión no se pasa a train_model
    save_directory="logs_comp",
    device="cuda" if torch.cuda.is_available() else "cpu",
    warmup_ratio=0.1,
    lr_encoder=1e-5,
    lr_others=5e-5,
    freeze_token_rep=False,   # no lo uso aquí porque train_model actual usa freeze_components
    max_types=15,
    shuffle_types=True,
    random_drop=True,
    max_neg_type_ratio=1,
    seed=42,
)

# =========================================================
# Preprocesado
# =========================================================

def convert_entities_to_token_level(data):
    """
    Convierte spans de entidades a formato token-level:
    (start, end, label) -> (i, i, label) para cada token del span
    """
    new_data = []
    for d in data:
        new_ner = []
        for s, f, c in d["ner"]:
            for i in range(s, f + 1):
                new_ner.append((i, i, c.lower()))
        new_data.append(
            {
                "tokenized_text": d["tokenized_text"],
                "ner": new_ner,
            }
        )
    return new_data


def build_label_vocab(data):
    labels = set()
    for d in data:
        for _, _, label in d["ner"]:
            labels.add(label.lower())
    return sorted(labels)


def prepare_gliner_025_dataset(
    data,
    max_types=15,
    shuffle_types=True,
    random_drop=True,
    max_neg_type_ratio=1.0,
    seed=42,
):
    """
    Replica fuera del modelo la lógica antigua de sampling:
    - max_types
    - shuffle_types
    - random_drop
    - max_neg_type_ratio

    Añade:
    - ner_labels
    - ner_negatives
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    import random
    py_rng = random.Random(seed)

    all_labels = build_label_vocab(data)
    prepared = []

    for d in data:
        ex = deepcopy(d)

        positive_labels = sorted({lab.lower() for _, _, lab in ex["ner"]})
        negative_labels = [lab for lab in all_labels if lab not in positive_labels]

        # random_drop sobre positivas
        kept_positive = positive_labels[:]
        if random_drop and len(kept_positive) > 1:
            kept_positive = [lab for lab in kept_positive if py_rng.random() > 0.15]
            if not kept_positive:
                kept_positive = [py_rng.choice(positive_labels)]

        # ratio de negativas
        if len(kept_positive) == 0:
            n_negs = min(len(negative_labels), max_types)
        else:
            n_negs = int(round(len(kept_positive) * max_neg_type_ratio))
            if negative_labels:
                n_negs = max(1, n_negs)

        sampled_negs = negative_labels[:]
        if shuffle_types:
            py_rng.shuffle(sampled_negs)
        sampled_negs = sampled_negs[:n_negs]

        candidate_labels = kept_positive + sampled_negs
        if shuffle_types:
            py_rng.shuffle(candidate_labels)

        candidate_labels = candidate_labels[:max_types]

        # Asegurar que no desaparezcan todas las positivas
        if kept_positive and not any(l in candidate_labels for l in kept_positive):
            candidate_labels = kept_positive[: min(len(kept_positive), max_types)]
            extras = [l for l in sampled_negs if l not in candidate_labels]
            candidate_labels += extras[: max(0, max_types - len(candidate_labels))]

        ex["ner_labels"] = candidate_labels
        ex["ner_negatives"] = [l for l in candidate_labels if l not in positive_labels]

        prepared.append(ex)

    return prepared


train_token = convert_entities_to_token_level(train_raw)
dev_token = convert_entities_to_token_level(dev_raw)

train_data = prepare_gliner_025_dataset(
    train_token,
    max_types=config.max_types,
    shuffle_types=config.shuffle_types,
    random_drop=config.random_drop,
    max_neg_type_ratio=config.max_neg_type_ratio,
    seed=config.seed,
)

# En validación mejor no hacer random_drop
eval_data = prepare_gliner_025_dataset(
    dev_token,
    max_types=config.max_types,
    shuffle_types=config.shuffle_types,
    random_drop=False,
    max_neg_type_ratio=config.max_neg_type_ratio,
    seed=config.seed,
)

# Esto lo dejamos aparte para la evaluación manual posterior
eval_for_metrics = {
    "entity_types": ENTITY_TYPES,
    "samples": dev_token,
}

# =========================================================
# Train
# =========================================================

print("## DEFINING TRAINING FUNCTION ##")

def train(model, config, train_data, eval_data=None):
    model = model.to(config.device)

    model.config.max_len = config.max_len
    print("GLiNER max_len:", config.max_len)
    masking = "global" if config.random_drop else "none"

    model.train_model(
        train_dataset=train_data,
        eval_dataset=eval_data,
        save_strategy = "no",
        output_dir=config.save_directory,

        # schedule
        max_steps=config.num_steps,
        warmup_ratio=config.warmup_ratio,

        # batch + optimizer
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.train_batch_size,
        learning_rate=float(config.lr_encoder),
        others_lr=float(config.lr_others),

        # loss / sampling aprox
        negatives=float(config.max_neg_type_ratio),
        masking=masking,

        # logging / save
        # save_steps=config.eval_every,
        # logging_steps=config.eval_every,
        # SIN GUARDAR CHECKPOINT PARA COMPROBAR ESPACIO
    )

# =========================================================
# Fine-tuning
# =========================================================

if finetune_model:
    print("## LAUNCHING TRAINING ##")
    train(model, config, train_data, eval_data)

    print("## SAVING TRAINED MODEL ##")
    output_path = f"outputs/{model_name}_finetuned"
    model.save_pretrained(output_path)

    # src_cfg = os.path.join(output_path, "gliner_config.json")
    # dst_cfg = os.path.join(output_path, "config.json")
    # if os.path.exists(src_cfg) and not os.path.exists(dst_cfg):
    #     import shutil
    #     shutil.copy(src_cfg, dst_cfg)

    # md = GLiNER.from_pretrained(output_path, local_files_only=True)
    # print("modelo cargado")
    # evaluación manual opcional después de entrenar

# =========================================================
# Predicción
# =========================================================

if generate_predictions:
    output_path = "../modeloNER_final"
    print(f"## LOADING PRE-TRAINED MODEL {output_path} ##")
    md = GLiNER.from_pretrained(output_path, local_files_only=True)

    print(f"## GENERATING NER PREDICTIONS FOR {PATH_ARTICLES}")
    with open(PATH_ARTICLES, "r", encoding="utf-8") as file:
        articles = json.load(file)

    print(f"len(articles): {len(articles)}")
    entity_labels = eval_for_metrics["entity_types"]

    predictions = {}

    for pmid, content in articles.items():
        title = content["title"]
        abstract = content["abstract"]

        title_entities = md.predict_entities(
            title,
            entity_labels,
            threshold=THRESHOLD,
            flat_ner=True,
            multi_label=False,
        )
        abstract_entities = md.predict_entities(
            abstract,
            entity_labels,
            threshold=THRESHOLD,
            flat_ner=True,
            multi_label=False,
        )


        unique_entities = []
        seen_entities = set()

        for entity in title_entities:
            key = (entity["start"], entity["end"], entity["text"], entity["label"], entity["score"])
            if key not in seen_entities:
                tmp_entity = {
                    "start_idx": entity["start"],
                    "end_idx": entity["end"],
                    "tag": "t",
                    "text_span": entity["text"],
                    "entity_label": entity["label"],
                    "score": entity["score"],
                }
                unique_entities.append(tmp_entity)
                seen_entities.add(key)

        for entity in abstract_entities:
            key = (entity["start"], entity["end"], entity["text"], entity["label"], entity["score"])
            if key not in seen_entities:
                tmp_entity = {
                    "start_idx": entity["start"],
                    "end_idx": entity["end"],
                    "tag": "a",
                    "text_span": entity["text"],
                    "entity_label": entity["label"],
                    "score": entity["score"],
                }
                unique_entities.append(tmp_entity)
                seen_entities.add(key)

        predictions[pmid] = unique_entities
        articles[pmid]["entities"] = unique_entities

    def default_serializer(obj):
        if isinstance(obj, set):
            return list(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    with open(PATH_OUTPUT_NER_PREDICTIONS, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2, default=default_serializer)

    print(f"## Predictions have been exported in JSON format to '/{PATH_OUTPUT_NER_PREDICTIONS}' ##")
