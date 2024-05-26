from transformers import BertForSequenceClassification, BertTokenizer, TextClassificationPipeline
import torch

pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
device = torch.device('cpu')
model = BertForSequenceClassification.from_pretrained('RabehOmrani/toxicity', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('RabehOmrani/toxicity')
# Function to classify text based on toxicity threshold
def classify_text(text: str):
    classification_result = pipeline(text)[0]
    return classification_result['label'], classification_result['score']
