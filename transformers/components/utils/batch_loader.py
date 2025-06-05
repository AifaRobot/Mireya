import torch
import json
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

'''The Dataset class is used to store all the data necessary for training the agent. Usually the agent is not 
trained with all the data but rather it is divided into smaller batches (minibatches) to reduce memory requirements 
and allow the data to be processed in a more manageable way.'''
class Dataset(torch.utils.data.Dataset):

    def __init__(self, data):
        super().__init__()

        self.input, self.target = data

    def __len__(self):
        return self.input.shape[0]

    '''This method returns an item. An item represents a batch with a previously established size. For example, if 
    we have an array of 120 elements and set a batch of 30 elements, when we use this method we will get a 
    An array of 30 elements, each random from the other (if we set Shuffle = True)'''
    def __getitem__(self, i):

        batch = dict()

        if (self.input != None):
            batch['input_batch'] = self.input[i]

        if (self.target != None):
            batch['target_batch'] = self.target[i]

        return batch

class JsonlDataset(torch.utils.data.Dataset):

    def __init__(self, filepath):
        self.filepath = filepath
        self.offsets = []

        with open(filepath, "r", encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                self.offsets.append(offset)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        with open(self.filepath, "r", encoding="utf-8") as f:
            f.seek(self.offsets[idx])
            line = f.readline()
            sample = json.loads(line)

            return sample["input"][0], sample["output"][0]

class BatchLoader():

    def __init__(self, data = None, path_file = '', batch_size = 32, num_workers = 0, shuffle = True, device = 'cpu'):
        self.dataloader = DataLoader(
            dataset = JsonlDataset(path_file) if data == None else Dataset(data),
            batch_size = batch_size,
            num_workers = num_workers,
            shuffle = shuffle,
            collate_fn = collate_fn if data == None else None
        )

        self.device = device

    def get_batch(self):
        batch = next(iter(self.dataloader))
        return batch['input_batch'].to(self.device), batch['target_batch'].to(self.device)

    def get_all(self):
        inputs = []
        targets = []

        for dict in self.dataloader:
            inputs.append(dict['input_batch'])
            targets.append(dict['target_batch'])

        inputs = torch.cat(inputs, dim=0)
        targets = torch.cat(targets, dim=0)

        return inputs, targets

def collate_fn(batch):
    inputs, outputs = zip(*batch)

    inputs = [torch.tensor(x) for x in inputs]
    outputs = [torch.tensor(x) for x in outputs]

    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    outputs_padded = pad_sequence(outputs, batch_first=True, padding_value=0)

    return { 'input_batch' : inputs_padded, 'target_batch' : outputs_padded}
