import os
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from typing import Iterable
from tqdm import tqdm


from noobai.data.util import save_model, load_model
from noobai.module.nn.loss import MaskedCrossEntropyLoss


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
        log_name=None,
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
        self.save_dir = os.path.join(self.save_dir, self.model_name)

        self.log_dir = os.path.join(self.save_dir, "logs")
        if log_name is not None:
            self.log_dir = os.path.join(self.log_dir, log_name)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.model.to(self.device)
        self.writer = None

    def simple_loss_func(self, output, data, len_mask=None):
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
        if isinstance(len_mask, Iterable) and not isinstance(len_mask, torch.Tensor):
            if len(len_mask) > 1:
                raise ValueError(
                    f'len_mask: {len_mask} cannot be used as loss function input. Because len(len_mask): {len(len_mask)} > 1.'
                )
            if len(len_mask) == 0:
                len_mask = None
            else:
                len_mask = len_mask[0]

        if criteria is None:
            if len(output.shape) <= 3 and len(output.shape) != len(data.shape):
                if len_mask is None:
                    criteria = F.cross_entropy
                else:
                    criteria = MaskedCrossEntropyLoss()
            else:
                criteria = F.mse_loss
        if (
            len(output.shape) == 3
            and output.size(1) == data.size(1)
            and output.size(1) != output.size(2)
        ):
            output = output.permute(1, 0, 2)
            data = data.permute(1, 0)
        if isinstance(criteria, MaskedCrossEntropyLoss):
            return criteria(output, data, len_mask)
        return criteria(output, data)

    def simple_accuracy(self, output, label):
        output, label = output[0], label[0]
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
        self.step = load_model(
            self.save_dir,
            self.model,
            self.optimizer,
            scheduler=self.scheduler,
            model_name=self.model_name,
        )
        return self.step

    def train(
        self,
        step=None,
        epochs=100000,
        save_iter=200,
        eval_iter=None,
        log_iter=None,
        has_target=True,
        data_mode=None,
        output_mode=None,
        load_model=True,
        simple_accuracy=True,
        clip_grad=None,
        save_model_num=2,
    ):
        r'''
        accuracy_mode: None=no acc.
        data_mode: a list whose elements are the list contains mode enum.
            0: ignore 1: in model 2: in loss 3: is len_mask 4: in eval

        output_mode: a list is alike to data_mode
            0: ignore 1: seen as normal output 2: as loss 3: as eval input
        '''
        if data_mode is not None:
            cnt = 0
            for i in range(len(data_mode)):
                if isinstance(data_mode[i], int):
                    data_mode[i] = [data_mode[i]]
                for j in data_mode[i]:
                    if j == 3:
                        cnt += 1
            if cnt > 1:
                raise ValueError("data_mode can only have one element with enum 3.")
        if output_mode is not None:
            cnt = 0
            for i in range(len(output_mode)):
                if isinstance(output_mode[i], int):
                    output_mode[i] = [output_mode[i]]
                for j in output_mode[i]:
                    if j == 2:
                        cnt += 1
            if cnt > 1:
                raise ValueError("output_mode can only have one element with enum 2.")
        if load_model:
            self.step = self.load_model()
        if step is not None:
            self.step = step
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
                if data_mode is None:
                    data_mode = [[1] for _ in range(len(data))]
                    if has_target:
                        data_mode[-1] = [2]
                    else:
                        for i in range(len(data_mode)):
                            data_mode[i] = [1, 2]
                if not isinstance(data, Iterable) or isinstance(data, torch.Tensor):
                    data = [data]
                data = [item for i, item in enumerate(data) if 0 not in data_mode[i]]
                data_mode = [
                    item for i, item in enumerate(data_mode) if 0 not in data_mode[i]
                ]
                data = [d.to(self.device, non_blocking=True) for d in data]
                in_model = [item for i, item in enumerate(data) if 1 in data_mode[i]]
                in_loss = [item for i, item in enumerate(data) if 2 in data_mode[i]]
                len_mask = [item for i, item in enumerate(data) if 3 in data_mode[i]]

                self.optimizer.zero_grad()
                output = self.model(*in_model)
                if not isinstance(output, Iterable) or isinstance(output, torch.Tensor):
                    output = [output]
                if output_mode is None:
                    output_mode = [[1] for _ in range(len(output))]
                output = [
                    item for i, item in enumerate(output) if 0 not in output_mode[i]
                ]
                output_mode = [
                    item
                    for i, item in enumerate(output_mode)
                    if 0 not in output_mode[i]
                ]
                normal_output = [
                    item for i, item in enumerate(output) if 1 in output_mode[i]
                ]
                for i, item in enumerate(output_mode):
                    if 2 in item:
                        loss = output[i]
                        break
                else:
                    loss = self.loss_func(normal_output, in_loss, len_mask=len_mask)
                loss.backward()
                if clip_grad is not None:
                    nn.utils.clip_grad_value_(self.model.parameters(), clip_grad)
                self.optimizer.step()

                msg = f"loss: {loss.item():.5f}"
                if simple_accuracy:
                    acc = self.simple_accuracy(normal_output, in_loss)
                    if acc is not None:
                        msg += f", accuracy: {acc:.4f}"
                    else:
                        msg += ", accuracy: None"
                if self.scheduler is not None:
                    self.scheduler.step()
                    try:
                        lr = self.scheduler.get_lr()
                    except NotImplementedError:
                        lr = self.scheduler.get_last_lr()
                    if isinstance(lr, Iterable):
                        lr = lr[0]
                    msg += f", lr: {lr:.3e}"
                t_bar.set_postfix_str(msg)
                t_bar.update()
                self.step += 1
                if log_iter is not None and self.step % log_iter == 0:
                    if self.writer is None:
                        self.writer = SummaryWriter(self.log_dir, flush_secs=5)
                    self.writer.add_scalar(
                        "loss", loss.item(), self.step, double_precision=True
                    )
                    if self.scheduler is not None:
                        self.writer.add_scalar(
                            "lr", lr, self.step, double_precision=True
                        )
                    if simple_accuracy:
                        self.writer.add_scalar(
                            "accuracy", acc, self.step, double_precision=True
                        )

                if self.step % save_iter == 0:
                    save_model(
                        self.save_dir,
                        self.model,
                        self.optimizer,
                        self.step,
                        scheduler=self.scheduler,
                        model_name=self.model_name,
                        max_num=save_model_num,
                    )
                if (
                    eval_iter is not None
                    and self.eval_func is not None
                    and self.step % eval_iter == 0
                ):
                    self.eval_func(
                        [item for i, item in enumerate(output) if 3 in output_mode[i]],
                        [item for i, item in enumerate(data) if 4 in data_mode[i]],
                    )
            t_bar.close()
