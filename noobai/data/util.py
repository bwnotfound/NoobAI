import os
import logging

import torch


def get_dir_list_sorted(save_dir, model_name):
    if not os.path.exists(save_dir):
        return []
    model_list = [
        name
        for name in os.listdir(save_dir)
        if name.endswith(".pt") and name.startswith(model_name)
    ]
    if len(model_list) == 0:
        return []
    model_list.sort(key=lambda x: int(x.split("_")[-1][:-3]), reverse=True)
    model_list = [os.path.join(save_dir, name) for name in model_list]
    return model_list


def save_model(
    save_dir,
    model,
    optimizer,
    step,
    scheduler=None,
    model_name="model",
    max_num=2,
    model_kwargs=None,
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_list = get_dir_list_sorted(save_dir, model_name=model_name)
    while len(model_list) >= max_num:
        p = model_list.pop()
        os.remove(p)
    data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "model_kwargs": model_kwargs,
    }
    if scheduler is not None:
        data["scheduler"] = scheduler.state_dict()
    with open(os.path.join(save_dir, "{}_{}.pt".format(model_name, step)), "wb") as f:
        torch.save(data, f)


def load_model(save_dir, model, optimizer, scheduler=None, model_name="model") -> int:
    model_list = get_dir_list_sorted(save_dir, model_name=model_name)
    if len(model_list) == 0:
        return 0
    with open(model_list[0], "rb") as f:
        data = torch.load(f)
    model.load_state_dict(data['model'], strict=False)
    optimizer.load_state_dict(data['optimizer'])
    step = data['step']
    if scheduler is not None:
        if "scheduler" not in data:
            logging.warning(
                "When load_model load data from data file, scheduler is supplied but its data is not found in data file."
            )
        else:
            scheduler.load_state_dict(data['scheduler'])
    return step


def classify_accuracy_calc(pred, tgt, dim=-1, ignore_token=None):
    pred = pred.argmax(dim=dim)
    if ignore_token is not None:
        mask = tgt != ignore_token
    correct = (pred == tgt) & mask
    correct = correct.sum().item()
    total = mask.sum().item()
    return correct / total, correct, total
