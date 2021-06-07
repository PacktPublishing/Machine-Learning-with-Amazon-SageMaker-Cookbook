import json

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.nn import functional as F
from transformers import AutoTokenizer


JSON_CONTENT_TYPE = 'application/json'


def model_fn(model_dir):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    return model


def predict_fn(input_data, model):
    tokenizer_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    sentence = input_data['text']
    
    batch = tokenizer(
        [sentence],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    output = model(**batch)

    class_a, class_b = F.softmax(output[0][0], dim = 0).tolist()
    
    prediction = "-"
    if class_a > class_b:
        prediction = "NEGATIVE"

    else:
        prediction = "POSITIVE"
    
    return prediction


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):  
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        
        return input_data
    else:
        pass

    
def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    
    raise Exception('Unsupported Content Type')