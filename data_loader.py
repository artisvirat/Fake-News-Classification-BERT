import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def create_mini_batch(samples):

    # 0: token tensor, 1: segment tensor, 2: label tensor
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    
    if not (samples[0][2] is None):
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None

    
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)
    
    # attention masks
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors !=0 , 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids

class FakeNewDataset(Dataset):
    def __init__(self, mode, df, label_map):
        assert mode in ["train","test"]
        self.mode = ModuleNotFoundError
        self.df = df 
        self.len = len(df)
        self.label_map = label_map
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    def __getitem__(self, idx):
        if self.mode =="test":
            label_tensor = None;
            text_a, text_b = self.df.iloc[idx, :2].values
        else:
            text_a, text_b, label = self.df.iloc[idx, :3].values
            label_tensor = torch.tensor(self.label_map[label])
        
        word_pieces = ["[CLS]"]
        tokens_a = self.tokenizer.tokenize(text_a)
        word_pieces = word_pieces + tokens_a + ["[SEP]"]
        len_a = len(tokens_a) + 2

        tokens_b = self.tokenizer.tokenize(text_b)
        len_b = len(tokens_b)+1
        
        word_pieces = word_pieces + tokens_b + ["[SEP]"]

        tokens_tensor = torch.tensor(self.tokenizer.convert_tokens_to_ids(word_pieces))

        segments_tensor = torch.tensor([0]*len_a + [1]*len_b, dtype=torch.long)

        return (tokens_tensor, segments_tensor, label_tensor)

    def __len__(self):
        return self.len