import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable
from tqdm import tqdm

from noobai.data.util import save_model, load_model


class SimpleTrainer:

    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        scheduler=None,
        loss_func=None,
        eval_func=None,
        device=None,
        save_dir=None,
        model_name="model",
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.scheduler = scheduler
        self.loss_func = loss_func
        self.device = device
        self.eval_func = eval_func
        self.save_dir = save_dir
        self.model_name = model_name
        if self.loss_func is None:
            self.loss_func = self.simple_loss_func
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.save_dir is None:
            self.save_dir = "./output"
            os.makedirs(self.save_dir, exist_ok=True)
        self.model.to(self.device)

    def simple_loss_func(self, output, data, has_target=True):
        criteria = None
        if hasattr(self, "criteria"):
            criteria = self.criteria
        if isinstance(output, Iterable) and not isinstance(output, torch.Tensor):
            output = output[0]
        if isinstance(data, Iterable) and not isinstance(data, torch.Tensor):
            if len(data) > 1:
                raise ValueError(
                    f'data: {data} cannot be used as loss function input. Because len(data): {len(data)} > 1.'
                )
            data = data[0]

        if criteria is None:
            if has_target:
                if len(output.shape) <= 3 and len(output.shape) != len(data.shape):
                    criteria = F.cross_entropy
                else:
                    criteria = F.mse_loss
            else:
                criteria = F.mse_loss
        if (
            len(output.shape) == 3
            and output.size(1) == data.size(1)
            and output.size(1) != output.size(2)
        ):
            output = output.permute(0, 2, 1)
        return criteria(output, data)

    def simple_accuracy(self, output, label):
        if not (len(output.shape) - len(label.shape) == 1):
            return None
        if isinstance(output, Iterable) and not isinstance(output, torch.Tensor):
            output = output[0]
        if (
            len(output.shape) == 3
            and output.size(1) == label.size(1)
            and output.size(1) != output.size(2)
        ):
            dim = 1
        else:
            dim = -1
        return (output.argmax(dim=dim) == label).sum().float() / label.numel()

    def set_criteria(self, func):
        self.criteria = func

    def load_model(self):
        return load_model(
            self.save_dir,
            self.model,
            self.optimizer,
            scheduler=self.scheduler,
            model_name=self.model_name,
        )

    def train(
        self,
        step=0,
        epochs=100,
        save_iter=200,
        eval_iter=None,
        has_target=True,
        simple_accuracy=True,
        clip_grad=None,
        load_model=True,
    ):
        r'''
        accuracy_mode: None=no acc. ""
        '''
        if load_model:
            step = self.load_model()
        if self.scheduler is not None:
            self.scheduler.step()
        for epoch in range(epochs):
            t_bar = tqdm(
                total=len(self.dataloader),
                ncols=100,
                desc=f"Epoch {epoch}",
                colour="green",
            )
            for data in self.dataloader:
                if not isinstance(data, Iterable) or isinstance(data, torch.Tensor):
                    data = [data]
                if has_target:
                    data, label = data[:-1], data[-1].to(self.device, non_blocking=True)
                data = [d.to(self.device, non_blocking=True) for d in data]

                self.optimizer.zero_grad()
                output = self.model(*data)
                if has_target:
                    loss = self.loss_func(output, label)
                else:
                    loss = self.loss_func(output, data)
                loss.backward()
                if clip_grad is not None:
                    nn.utils.clip_grad_value_(self.model.parameters(), clip_grad)
                self.optimizer.step()

                msg = f"loss: {loss.item():.5f}"
                if simple_accuracy:
                    acc = self.simple_accuracy(output, label)
                    if acc is not None:
                        msg += f", accuracy: {acc:.4f}"
                    else:
                        msg += ", accuracy: None"
                if self.scheduler is not None:
                    try:
                        lr = self.scheduler.get_lr()
                    except NotImplementedError:
                        lr = self.scheduler.get_last_lr()
                    if isinstance(lr, Iterable):
                        lr = lr[0]
                    msg += f", lr: {lr:.3e}"
                t_bar.set_postfix_str(msg)
                t_bar.update()
                step += 1

                if step % save_iter == 0:
                    save_model(
                        self.save_dir,
                        self.model,
                        self.optimizer,
                        step,
                        scheduler=self.scheduler,
                        model_name=self.model_name,
                    )
                if (
                    eval_iter is not None
                    and self.eval_func is not None
                    and step % eval_iter == 0
                ):
                    self.eval_func()
            t_bar.close()
