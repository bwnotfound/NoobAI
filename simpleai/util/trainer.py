import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable
from tqdm import tqdm

from simpleai.data.util import save_model


class SimpleTrainer:

    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        loss_func=None,
        eval_func=None,
        device=None,
        save_dir=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.loss_func = loss_func
        self.device = device
        self.eval_func = eval_func
        self.save_dir = save_dir
        if self.loss_func is None:
            self.loss_func = self.simple_loss_func

    def simple_loss_func(self, output, data, has_target=True):
        criteria = None
        if hasattr(self, "criteria"):
            criteria = self.criteria
        if isinstance(output, Iterable):
            output = output[0]
        if has_target:
            if criteria is not None:
                return criteria(output, data)
            return F.cross_entropy(output, data)
        if len(data) == 1:
            if criteria is not None:
                return criteria(output, data[0])
            return F.mse_loss(output, data[0])
        raise ValueError(
            f'data: {data} cannot be used as loss function input. Because len(data): {len(data)} > 1.'
        )

    def simple_accuracy(self, output, label):
        if isinstance(output, Iterable):
            output = output[0]
        return (output.argmax(dim=-1) == label).sum().float() / label.numel()

    def set_criteria(self, func):
        self.criteria = func

    def train(
        self,
        step=0,
        epochs=100,
        save_iter=200,
        eval_iter=None,
        has_target=True,
        simple_accuracy=False,
        clip_grad=None,
    ):
        r'''
        accuracy_mode: None=no acc. ""
        '''
        for epoch in range(epochs):
            t_bar = tqdm(
                total=len(self.dataloader),
                ncols=100,
                desc=f"Epoch {epoch}",
                colour="green",
            )
            for data in self.dataloader:
                if not isinstance(data, Iterable):
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
                    msg += f", accuracy: {acc:.4f}"
                t_bar.set_postfix_str(msg)
                t_bar.update()
                step += 1

                if step % save_iter == 0:
                    save_model(self.save_dir, self.model, self.optimizer, step)
                if (
                    eval_iter is not None
                    and self.eval_func is not None
                    and step % eval_iter == 0
                ):
                    self.eval_func()
            t_bar.close()
