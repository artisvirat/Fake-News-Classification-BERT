from transformers import BertForSequenceClassification
import torch

def get_model():
    model = BertForSequenceClassification.from_pretrained("bert-base-english", num_labels=3)
    return model

def get_prediction(model, dataloader, compute_acc=False):
    predictions = None
    total_correct = 0
    total = 0
    
    with torch.no_grad():
        for data in dataloader:
            
            #Put all tensors to the GPU
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if not (t is None)]
                
            # get the tensors and give it to the model for prediction and get the output
            tokens, segments, masks = data[:3]
            outputs = model(input_ids=tokens, 
                            token_type_ids=segments, 
                            attention_mask=masks)
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)
            
            if compute_acc:
                true_labels = data[3]
                total += true_labels.size(0) # record how many data in this batch
                total_correct += (pred == true_labels).sum().item() # count how many labels are correct in this batch
                
            # record all the batches
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
    
    if compute_acc:
        acc = total_correct / total
        return predictions ,acc
    else:
        return predictions

