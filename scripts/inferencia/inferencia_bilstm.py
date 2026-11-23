from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torch 
import torch.nn as nn
import json
from safetensors.torch import load_file
# 1. Elegimos un modelo preentrenado 
model_name = "./best_model_electra_bi" 
model_names = ["microsoft/deberta-v3-large",
               "michiyasunaga/BioLinkBERT-large",
               "mingxiaw/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
               "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
               "microsoft/BiomedNLP-BiomedELECTRA-large-uncased-abstract"]
model_path = model_names[0]
label_list = ENTITIES = ['dietary supplement', 'chemical', 'animal', 'ddf',
        'anatomical location', 'gene', 'bacteria', 'microbiome', 'human',
        'biomedical technique', 'statistical technique', 'food', 'drug', 'O']
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}
# # 2. Cargamos tokenizer y modelo 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name) 
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
        labels=None
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
model = BertBiLSTMNER(
    model_name=model_path,
    num_labels=len(label_list)
)
state_dict = load_file(
    f"{model_name}/model.safetensors"
)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

with open("data/chunked_test.json", "r", encoding='utf-8') as f:
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
        inputs.pop("token_type_ids", None)
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
            "pmid": example[0],
            "location": field,
            "tokens": tokens,
            "predictions": word_predictions
        })


# -------------------
# preview
# -------------------

print(all_predictions[0])

with open("predictions/test10.json", "w", encoding='utf-8') as f:
    json.dump(all_predictions, f, ensure_ascii=False, indent=2)


