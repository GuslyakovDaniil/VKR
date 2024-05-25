import torch
import math
import sys
import time
import datetime
from collections import deque


def collate_fn(batch):
    return tuple(zip(*batch))


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter(delimiter=self.delimiter)
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = AverageMeter()
        data_time = AverageMeter()
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))


class SmoothedValue:
    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)  # Используем maxlen для ограничения размера deque
        self.window_size = window_size

    def update(self, value):
        self.deque.append(value)

    @property
    def value(self):
        return sum(self.deque) / len(self.deque)

    @property
    def median(self):
        return sorted(self.deque)[len(self.deque) // 2]


class AverageMeter:
    def __init__(self, fmt=':f', delimiter='\t'):
        self.deque = deque(maxlen=100)  # Используем maxlen для ограничения размера deque
        self.fmt = fmt
        self.delimiter = delimiter

    def update(self, value, n=1):
        self.deque.append(value)
        if len(self.deque) > 100:  # Изменяем размер окна на нужное значение
            self.deque.popleft()

    def reset(self):
        self.deque.clear()  # Используем метод clear() для очистки deque

    def __str__(self):
        fmtstr = '{avg' + self.fmt + '} ({global_avg' + self.fmt + '})'
        avg = self.avg
        global_avg = self.global_avg
        return fmtstr.format(avg=avg, global_avg=global_avg)

    @property
    def avg(self):
        return sum(self.deque) / max(len(self.deque), 1)

    @property
    def global_avg(self):
        return self.avg  # Это необходимо изменить в зависимости от вашего использования

    def synchronize_between_processes(self):
        pass  # Мы не используем распределенное обучение, поэтому это не нужно


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def reduce_dict(input_dict):
    output_dict = {}
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            output_dict[k] = v.item()
        else:
            output_dict[k] = v
    return output_dict

