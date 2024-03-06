import json
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from simpleai.model.sam.model import STM
from simpleai.task.memory.number_recall import NumberRecallDataset
from simpleai.data.util import load_model
from simpleai.util.trainer import SimpleTrainer


class Config:
    task = "number_recall"
    epoch = 100
    eval_iter = 1000
    save_iter = 200
    batch_size = 32
    in_dim = 128
    controller_size = 256
    memory_units = 256
    memory_unit_size = 64
    num_heads = 4
    num_slot = 2
    slot_size = 96
    rel_size = 96
    clip_grad = 10
    num_workers = 0
    resume = True
    model_dir = None
    data_path = "./data/associative-retrieval.pkl"
    log_dir = "./log"


def train(config_path=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "configs/number_recall.json"
        )
    with open(config_path, encoding="utf-8") as f:
        config_raw_data = json.load(f)
    config = Config()
    for k, v in config_raw_data.items():
        setattr(config, k, v)
    log_dir = os.path.join(config.log_dir, config.task)
    os.makedirs(log_dir, exist_ok=True)

    save_dir = config.model_dir
    if save_dir is None:
        save_dir = os.path.join(log_dir, "model")
    os.makedirs(save_dir, exist_ok=True)

    dataset = NumberRecallDataset(path=config.data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=dataset.collate_fn,
    )
    model = STM(
        dataset.n_vocab,
        config.in_dim,
        num_slot=config.num_slot,
        slot_size=config.slot_size,
        rel_size=config.rel_size,
        init_alphas=[None, None, None],
    )
    model.to(device)

    # print("====num params=====")
    # print(model.calculate_num_params())
    # print("========")

    # optimizer = optim.RMSprop(model.parameters(), lr=1e-4, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    if config.resume:
        step = load_model(save_dir, model, optimizer)
    else:
        step = 0

    model.train()
    trainer = SimpleTrainer(
        model,
        dataloader,
        optimizer,
        device=device,
        save_dir=save_dir,
    )
    trainer.set_criteria(nn.CrossEntropyLoss(ignore_index=-1))
    print("===training===")
    trainer.train(
        step=step,
        epochs=config.epoch,
        save_iter=config.save_iter,
        eval_iter=config.eval_iter,
        clip_grad=config.clip_grad,
        simple_accuracy=True,
    )


if __name__ == '__main__':
    train()
