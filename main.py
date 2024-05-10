import torch
from transformers import BertForSequenceClassification, BertTokenizer, TextClassificationPipeline
from fastapi import FastAPI, HTTPException

# Load model and tokenizer
device = torch.device('cpu')
model = BertForSequenceClassification.from_pretrained('./final/my_outputs', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('./final/my_outputs')

# Create TextClassificationPipeline
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)

# Initialize FastAPI app
app = FastAPI()

# Function to classify text based on toxicity threshold
def classify_text(text):
    classification_result = pipeline(text)[0]
    return classification_result['label'], classification_result['score']

# Endpoint to classify comments, posts, and messages
@app.get("/classify/")
async def classify_text_type(text: str, text_type: str):
    '''
    -if it's a message => classify as usual
    -comments are more sensitive and needs more censoring
    -posts must be subdivied to not loose accuracy
    '''
    try:
        if text_type == 'message':
            label, score = classify_text(text)
            return {'label': label, 'score': score}
        elif text_type == 'comment':
            label, score = classify_text(text)
            if label == 'non-toxic' and score < 0.7:
                return {'label': 'toxic', 'score': score}
            else:
                return {'label': 'non-toxic', 'score': score}
        elif text_type == 'post':
            sentences = text.split('.')  # Split post into sentences
            for sentence in sentences:
                label, score = classify_text(sentence)
                if label == 'toxic':
                    return {'label': 'toxic', 'score': score}
            return {'label': 'non-toxic', 'score': 1.0}  # If no toxic sentences found
        else:
            raise HTTPException(status_code=400, detail="Invalid text type. Allowed types: message, comment, post")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Example usage:
# http://127.0.0.1:8000/classify/?text=Your%20text%20here&text_type=message
# http://127.0.0.1:8000/classify/?text=Your%20comment%20here&text_type=comment
# http://127.0.0.1:8000/classify/?text=Your%20post%20here&text_type=post
