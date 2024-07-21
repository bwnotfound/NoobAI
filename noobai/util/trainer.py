import os
import datetime
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from typing import Iterable
from tqdm import tqdm


from noobai.data.util import save_model, load_model
from noobai.module.nn.loss import MaskedCrossEntropyLoss


'''
    2D is seen as batch_first.
    3D is seen as seq_first.
'''


class DataMode(Enum):
    ignore = 0
    data_rest_in_model = 1
    data_in_model = 2
    data_in_loss = 3
    data_is_len_mask = 4
    data_in_eval = 5
    data_in_acc = 6
    model_output_rest_ignore = 7
    model_output_in_loss = 8
    model_output_as_loss = 9
    model_output_in_eval = 10
    model_output_log = 11
    model_output_in_acc = 12
    loss_cross_entropy = 13
    loss_cross_entropy_mask = 14  # cross_entropy is last dim by default.
    loss_mse = 15
    straight_loss = 16


'''
    all data is considered as batch_first and tensor.
'''


class TrainerMode(Enum):
    classify = 1
    image_auto_regression = 2
    nlp_sentence_generate = 3
    straight_loss = 4


class NoobTrainer:

    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        train_mode=None,
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

        self.set_train_mode(train_mode)

    def set_train_mode(self, train_mode):
        if train_mode is None:
            train_mode = TrainerMode.classify
        if isinstance(train_mode, TrainerMode):
            train_mode = self.generate_three_mode_from_trainer_mode(train_mode)
        self.data_mode, self.output_mode, self.loss_mode = self._preprocess_three_mode(
            train_mode
        )

    def generate_three_mode_from_trainer_mode(self, trainer_mode: TrainerMode):
        '''
        return:
            (data_mode, output_mode, loss_mode)
        '''
        match trainer_mode:
            case TrainerMode.classify:
                return (
                    [
                        DataMode.data_rest_in_model,
                        [
                            DataMode.data_in_loss,
                            DataMode.data_in_eval,
                            DataMode.data_in_acc,
                        ],
                    ],
                    [
                        [
                            DataMode.model_output_in_loss,
                            DataMode.model_output_in_eval,
                            DataMode.model_output_in_acc,
                        ]
                    ],
                    DataMode.loss_cross_entropy,
                )
            case TrainerMode.image_auto_regression:
                return (
                    [
                        [
                            DataMode.data_in_model,
                            DataMode.data_in_loss,
                            DataMode.data_in_eval,
                        ]
                    ],
                    [DataMode.model_output_in_loss, DataMode.model_output_in_eval],
                    DataMode.loss_mse,
                )
            case TrainerMode.nlp_sentence_generate:
                return (
                    [
                        [
                            DataMode.data_rest_in_model,
                            DataMode.data_in_eval,
                            DataMode.data_in_acc,
                        ],
                        DataMode.data_is_len_mask,
                    ],
                    [
                        [
                            DataMode.model_output_in_loss,
                            DataMode.data_in_eval,
                            DataMode.model_output_in_acc,
                        ]
                    ],
                    DataMode.loss_cross_entropy_mask,
                )
            case TrainerMode.straight_loss:
                return (
                    [DataMode.data_rest_in_model],
                    [DataMode.model_output_rest_ignore, DataMode.model_output_in_loss],
                    DataMode.straight_loss,
                )

    def _preprocess_three_mode(self, three_mode):
        data_mode, output_mode, loss_mode = three_mode
        if not isinstance(data_mode, (list, tuple)):
            data_mode = [data_mode]
        if not isinstance(output_mode, (list, tuple)):
            output_mode = [output_mode]
        if not isinstance(loss_mode, (list, tuple)):
            loss_mode = [loss_mode]
        new_data_mode = []
        has_len_mask = False
        for m in data_mode:
            if not isinstance(m, (list, tuple)):
                new_data_mode.append([m])
            else:
                new_data_mode.append(m)
            if (
                DataMode.data_rest_in_model in new_data_mode[-1]
                and len(new_data_mode[-1]) != 1
            ):
                raise ValueError(
                    "DataMode.data_rest_in_model should be alone at one element."
                )
            if DataMode.data_is_len_mask in new_data_mode[-1]:
                if has_len_mask:
                    raise ValueError("DataMode.has_len_mask should only appear once.")
                has_len_mask = True
        new_output_mode = []
        for m in output_mode:
            if not isinstance(m, (list, tuple)):
                new_output_mode.append([m])
            else:
                new_output_mode.append(m)
            if (
                DataMode.model_output_rest_ignore in new_output_mode[-1]
                and len(new_output_mode[-1]) != 1
            ):
                raise ValueError(
                    "DataMode.model_output_rest_ignore should be alone at one element."
                )
            if DataMode.ignore in new_output_mode[-1] and len(new_output_mode[-1]) != 1:
                raise ValueError("DataMode.ignore should be alone at one element.")
        if isinstance(loss_mode, (list, tuple)):
            loss_mode = loss_mode[0]
        return new_data_mode, new_output_mode, loss_mode

    def simple_loss_func(self, output, data, len_mask=None):
        criteria = None
        if hasattr(self, "criteria"):
            criteria = self.criteria
        elif self.loss_mode == DataMode.loss_cross_entropy:
            criteria = nn.CrossEntropyLoss()
        elif self.loss_mode == DataMode.loss_mse:
            criteria = nn.MSELoss()
        elif self.loss_mode == DataMode.loss_cross_entropy_mask:
            criteria = MaskedCrossEntropyLoss()
        assert criteria is not None, "criteria should be set firstly."

        if isinstance(output, (list, tuple)):
            assert len(output) == 1, "output should have only one element."
            output = output[0]
        if isinstance(data, (list, tuple)):
            assert len(data) == 1, "data should have only one element."
            data = data[0]
        if len_mask is not None and isinstance(len_mask, (list, tuple)):
            if len(len_mask) == 0:
                len_mask = None
            else:
                assert len(len_mask) == 1, "len_mask should have only one element."
                len_mask = len_mask[0]

        # seq_first default.
        if len_mask is not None:
            return criteria(output, data, len_mask)
        return criteria(output, data)

    def simple_accuracy(self, output, label):
        # only work for logits prediction.
        if isinstance(output, (list, tuple)):
            assert len(output) == 1, "output should have only one element."
            output = output[0]
        if isinstance(label, (list, tuple)):
            assert len(label) == 1, "label should have only one element."
            label = label[0]
        if not (len(output.shape) - len(label.shape) == 1):
            return None
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
        load_model=True,
        simple_accuracy=True,
        clip_grad=None,
        save_model_num=2,
    ):
        r'''
        accuracy_mode: None=no acc.
        '''
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
                if not isinstance(data, tuple):
                    data = [data]
                else:
                    data = list(data)
                data_length = len(data)
                data_mode = []
                for m in self.data_mode:
                    if DataMode.data_rest_in_model in m:
                        for _ in range(data_length - len(self.data_mode) + 1):
                            data_mode.append([DataMode.data_in_model])
                        continue
                    data_mode.append(m)
                assert len(data_mode) == len(
                    data
                ), "data_mode should have same size of data"

                data = [
                    d.to(self.device, non_blocking=True)
                    for i, d in enumerate(data)
                    if DataMode.ignore not in data_mode[i]
                ]
                data_in_model = [
                    d
                    for i, d in enumerate(data)
                    if DataMode.data_in_model in data_mode[i]
                ]
                data_in_loss = [
                    d
                    for i, d in enumerate(data)
                    if DataMode.data_in_loss in data_mode[i]
                ]
                data_in_eval = [
                    d
                    for i, d in enumerate(data)
                    if DataMode.data_in_eval in data_mode[i]
                ]
                data_in_acc = [
                    d
                    for i, d in enumerate(data)
                    if DataMode.data_in_acc in data_mode[i]
                ]
                len_mask = None
                for i in range(len(data)):
                    if DataMode.data_in_loss in data_mode[i]:
                        len_mask = data[i]
                        break

                self.optimizer.zero_grad()
                output = self.model(*data_in_model)
                if not isinstance(output, tuple):
                    output = [output]
                else:
                    output = list(output)

                output_length = len(output)
                output_mode = []
                for m in self.output_mode:
                    if DataMode.model_output_rest_ignore in m:
                        for _ in range(output_length - len(self.output_mode) + 1):
                            output_mode.append([DataMode.ignore])
                        continue
                    output_mode.append(m)
                assert len(output_mode) == len(
                    output
                ), "output_mode should have same size of output"

                output_in_loss = [
                    item
                    for i, item in enumerate(output)
                    if DataMode.model_output_in_loss in output_mode[i]
                ]
                output_as_loss = [
                    item
                    for i, item in enumerate(output)
                    if DataMode.model_output_as_loss in output_mode[i]
                ]
                output_in_eval = [
                    item
                    for i, item in enumerate(output)
                    if DataMode.model_output_in_eval in output_mode[i]
                ]
                output_in_acc = [
                    item
                    for i, item in enumerate(output)
                    if DataMode.model_output_in_acc in output_mode[i]
                ]
                if len(output_in_loss) == 0:
                    loss = self.loss_func(
                        output_in_loss, data_in_loss, len_mask=len_mask
                    )
                else:
                    assert (
                        len(output_in_loss) == 1
                    ), "output_in_loss should have only one element."
                    loss = output_as_loss[0]
                loss.backward()
                if clip_grad is not None:
                    nn.utils.clip_grad_value_(self.model.parameters(), clip_grad)
                self.optimizer.step()

                msg = f"loss: {loss.item():.5f}"
                if simple_accuracy:
                    acc = self.simple_accuracy(output_in_acc, data_in_acc)
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
                        output_in_eval,
                        data_in_eval,
                    )
            t_bar.close()
