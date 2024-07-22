# TODO: 此代码用于简单的CausalLM训练与测试，由于本地环境和repository代码要求的问题，目前先放在这，等以后维护一下

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from functools import partial

# import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from tqdm import tqdm
import numpy as np

from eval import CLMBleu

split_ratio = 0.5
max_length = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# lr = 2e-5
lr = 3e-5
epochs = 20
# batch_size = 16
batch_size = 4


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")

# dataset_path = "dataset/test_cn.json"
dataset_path = "dataset/test_name.json"
with open(dataset_path, "r", encoding="utf-8") as f:
    raw_ds = json.loads(f.read())
dataset_pair_list = [
    (block["instruction"] + block["input"], block["output"]) for block in raw_ds
]
np.random.shuffle(dataset_pair_list)
split_index = int(split_ratio * len(dataset_pair_list))
train_data, test_data = dataset_pair_list[:split_index], dataset_pair_list[split_index:]


class SimpleQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.prompt = tokenizer([d[0] for d in self.data])["input_ids"]
        self.output = tokenizer([d[1] for d in self.data])["input_ids"]
        self.max_length = max_length
        self.eos_token_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"prompt": self.prompt[idx], "output": self.output[idx]}

    def get_collate_fn(self):
        return partial(
            self.collate_fn, eos_token_id=self.eos_token_id, max_len=self.max_length
        )

    @staticmethod
    def collate_fn(examples, eos_token_id, max_len):
        prompt = [d["prompt"] for d in examples]
        output = [d["output"] for d in examples]
        input_ids = [prompt[i] + output[i] for i in range(len(examples))]
        labels = [
            [
                -100 if j < len(prompt[i]) - 1 else input_ids[i][j + 1]
                for j in range(len(input_ids[i]) - 1)
            ]
            + [eos_token_id]
            for i in range(len(examples))
        ]
        input_ids = [torch.LongTensor(d) for d in input_ids]
        labels = [torch.LongTensor(d) for d in labels]
        input_ids = pad_sequence(input_ids, batch_first=True)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return input_ids[:, :max_len], labels[:, :max_len]


train_ds = SimpleQADataset(train_data, tokenizer, max_length=max_length)
eval_ds = SimpleQADataset(test_data, tokenizer, max_length=max_length)
collate_fn = train_ds.get_collate_fn()

train_dataloader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn
)
eval_dataloader = DataLoader(
    eval_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn
)


# wandb.init(project="test", config={"lr": lr, "epochs": epochs})

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()
step = 0

eval_metrics = CLMBleu(tokenizer)

try:
    data = torch.load("checkpoints/train.pt")
    if "model" in data:
        model.load_state_dict(data["model"])
    if "optim" in data:
        optimizer.load_state_dict(data["optim"])
    if "step" in data:
        step = data["step"]
except:
    pass

total_eval_step = 0

def eval(dataloader):
    global total_eval_step
    total_eval_step += 1
    with torch.no_grad():
        # logits_list, labels_list = [], []
        # for input_ids, labels in tqdm(
        #     eval_dataloader, ncols=80, colour="blue", leave=False
        # ):
        #     input_ids, labels = input_ids.to(device), labels.to(device)
        #     output = model(input_ids)
        #     logits = output.logits
        #     logits_list.append(logits)
        #     labels_list.append(labels)
        # mean_result = {}
        # for logits, labels in tqdm(zip(logits_list, labels_list), ncols=80, colour="red", leave=False):
        #     result = eval_metrics.compute_metrics((logits.cpu().numpy(), labels.cpu().numpy()))
        #     for k, v in list(result.items()):
        #         try:
        #             total = mean_result.get(k, 0)
        #             total = total + v
        #             mean_result[k] = total
        #         except:
        #             pass
        # for k, v in list(mean_result.items()):
        #     mean_result[k] = v / len(eval_dataloader)
        # wandb.log(mean_result)

        input_text_list, pred_text_list, target_text_list = [], [], []
        for input_ids, labels in tqdm(dataloader, ncols=80, colour="blue", leave=False):
            input_ids, labels = input_ids.to(device), labels.to(device)

            mask = labels.squeeze(0) != -100
            start_index = mask.nonzero()[0]
            target_text_list.append(
                tokenizer.decode(labels[:, start_index:].cpu().numpy().flatten())
            )

            input_text_list.append(tokenizer.decode(input_ids.cpu().numpy().flatten()))

            next_input = input_ids[:, : start_index + 1]
            for _ in tqdm(range(8), ncols=80, colour="yellow", leave=False):
                output = model(next_input)
                pred = output.logits.argmax(dim=-1)
                next_input = torch.cat([next_input, pred[:, -1:]], dim=1)
                if pred[0, -1] == tokenizer.eos_token_id:
                    break
            pred = next_input.squeeze(0).cpu().numpy()
            pred_text_list.append(tokenizer.decode(pred))

        result = []
        for i, a, b in zip(input_text_list, pred_text_list, target_text_list):
            result.append(
                f"---input_ids: {i}\n---目标：{b}\n---预测：{a}\n-------------------------------\n"
            )
        result = "\n".join(result)
        result = f"----Eval Step: {total_eval_step}----\n" + result
        with open("test.txt", "a", encoding="utf-8") as f:
            f.write(result)


for epoch in range(epochs):
    t_bar = tqdm(total=len(train_dataloader), colour="green", ncols=80, leave=False)
    total_loss = 0
    eval(eval_dataloader)
    for input_ids, labels in train_dataloader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        output = model(input_ids)
        logits = output.logits
        # Should be batch_first
        loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t_bar.set_postfix_str(f"Loss: {loss.item():.5f}")
        t_bar.update()
        step += 1
        total_loss += loss.item()

        # if step % 400 == 0:
        #     torch.save(
        #         {
        #             "model": model.state_dict(),
        #             "optim": optimizer.state_dict(),
        #             "step": step,
        #         },
        #         "checkpoints/train.pt",
        #     )
        # if step % 200 == 0:
        #     eval()
        # if step % 3 == 0:
        #     wandb.log({"loss": loss.item()})
    tqdm.write(f"Avg_loss: {total_loss / len(train_dataloader):.5f}")
    t_bar.close()
