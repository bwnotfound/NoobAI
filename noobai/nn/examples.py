import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


def rnn_pack_pad_collate_fn(examples):
    examples.sort(key=lambda data: len(data[0]), reverse=True)
    lengths = [len(data[0]) for data in examples]
    input_ids = [torch.LongTensor(data[0]) for data in examples]
    labels = torch.LongTensor([data[1] for data in examples])
    mask = [torch.ones(l, dtype=torch.uint8) for l in lengths]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    mask = pad_sequence(mask, batch_first=True, padding_value=0)
    return input_ids, labels, lengths, mask


class PaddingRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        x = pack_padded_sequence(x, lengths, batch_first=True)
        lstm_out, _ = self.lstm(x)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        out = self.fc(lstm_out[-1])
        return out
