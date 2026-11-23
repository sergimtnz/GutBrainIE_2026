from transformers import AutoTokenizer, AutoModelForTokenClassification 
import torch 
import json
# 1. Elegimos un modelo preentrenado 
model_name = "./best_model_electra"

label_list = ENTITIES = ['dietary supplement', 'chemical', 'animal', 'ddf',
        'anatomical location', 'gene', 'bacteria', 'microbiome', 'human',
        'biomedical technique', 'statistical technique', 'food', 'drug', 'O']
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}
# # 2. Cargamos tokenizer y modelo 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name) 
model = AutoModelForTokenClassification.from_pretrained( model_name, num_labels=len(label_list) ) 
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
            "pmid":example[0],
            "location": field,
            "tokens": tokens,
            "predictions": word_predictions
        })
with open("predictions/test5.json", "w") as file:
	json.dump(all_predictions, file, indent=2)
# -------------------
# preview
# -------------------

print(all_predictions[0])
# all_predictions = []
# # # 4. Tokenización 
# inputs = tokenizer( text, return_tensors="pt", padding=True, truncation=True ) 
# # # 5. Pasamos los tokens por el modelo 
# with torch.no_grad(): 
#     outputs = model(**inputs) 
# # 6. Obtenemos los logits 
# logits = outputs.logits 
# # 7. Convertimos logits a probabilidades 
# probabilities = torch.softmax(logits, dim=-1) 
# # 8. Obtenemos la clase predicha 
# predicted_class = torch.argmax(probabilities, dim=-1) 
# print("Logits:", logits) print("Probabilidades:", probabilities) print("Clase predicha:", predicted_class.item()) 


