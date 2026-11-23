from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torch 
import torch.nn as nn
import json
from safetensors.torch import load_file
from torchcrf import CRF
# 1. Elegimos un modelo preentrenado 
model_name = "./best_model_deb_large_crf" 
model_names = ["microsoft/deberta-v3-large",
               "michiyasunaga/BioLinkBERT-large",
               "mingxiaw/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
               "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
               "microsoft/BiomedNLP-BiomedELECTRA-large-uncased-abstract"]
model_base = model_names[0]
label_list = ENTITIES = ['dietary supplement', 'chemical', 'animal', 'ddf',
        'anatomical location', 'gene', 'bacteria', 'microbiome', 'human',
        'biomedical technique', 'statistical technique', 'food', 'drug', 'O']
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}
# # 2. Cargamos tokenizer y modelo 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name) 
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
model = DebertaCRF(
    model_name=model_base,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)
state_dict = load_file(
    f"{model_name}/model.safetensors"
)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

with open("data/chunked_dev_partes.json", "r", encoding='utf-8') as f:
    dev = json.load(f)



all_predictions = []

for example in dev:
    chunks = example[1:]
    for chunk in chunks:

        field = list(chunk.keys())[0].split("_")[1]
        tokens = chunk[f"tokenized_{field}"]

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
            pred_ids = model.decode(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        pred_ids = pred_ids[0]

        # predicción
        predictions = [id2label[pred_id] for pred_id in pred_ids]

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
            "pmid": example[0],
            "location": field,
            "tokens": tokens,
            "predictions": word_predictions
        })

# -------------------
# preview
# -------------------

print(all_predictions[0])

with open("predicciones/mue_deb_crf.json", "w", encoding='utf-8') as f:
    json.dump(all_predictions, f, ensure_ascii=False, indent=2)


