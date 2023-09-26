from collections import defaultdict, deque
import datetime
import time
from audtorch.metrics.functional import pearsonr
import torch
import cv2
import numpy as np
import torch.nn.functional as F


# 皮尔逊相关系数 np.corrcoef(x, y)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def discretize_gt(gt):
    import warnings
    warnings.warn('can improve the way GT is discretized')
    return gt / 255


def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    min_s_map = torch.min(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, w, h)
    max_s_map = torch.max(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, w, h)

    norm_s_map = (s_map - min_s_map) / (max_s_map - min_s_map * 1.0)
    return norm_s_map


def auc_shuff(s_map, gt, splits=100, stepsize=0.1):
    s_map = normalize_map(s_map)
    if len(s_map.size()) == 3:
        s_map = s_map[0, :, :]
        gt = gt[0, :, :]

    s_map = s_map.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    c = gt.mean()
    other_map = np.zeros_like(gt)
    other_map[gt > c] = 1

    num_fixations = np.sum(gt)

    x, y = np.where(other_map == 1)
    other_map_fixs = []
    for j in zip(x, y):
        other_map_fixs.append(j[0] * other_map.shape[0] + j[1])
    ind = len(other_map_fixs)
    assert ind == np.sum(other_map), 'something is wrong in auc shuffle'

    num_fixations_other = min(ind, num_fixations)

    num_pixels = s_map.shape[0] * s_map.shape[1]
    random_numbers = []
    for i in range(0, splits):
        temp_list = []
        t1 = np.random.permutation(ind)
        for k in t1:
            temp_list.append(other_map_fixs[k])
        random_numbers.append(temp_list)

    aucs = []
    # for each split, calculate auc
    for i in random_numbers:
        r_sal_map = []
        for k in i:
            r_sal_map.append(s_map[k % s_map.shape[0] - 1, int(k / s_map.shape[0])])
        # in these values, we need to find thresholds and calculate auc
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        r_sal_map = np.array(r_sal_map)

        # once threshs are got
        thresholds = sorted(set(thresholds))
        area = []
        area.append((0.0, 0.0))
        for thresh in thresholds:
            # in the salience map, keep only those pixels with values above threshold
            temp = np.zeros(s_map.shape)
            temp[s_map >= thresh] = 1.0
            num_overlap = np.where(np.add(temp, gt) == 2)[0].shape[0]
            tp = num_overlap / (num_fixations * 1.0)

            # fp = (np.sum(temp) - num_overlap)/((np.shape(gt)[0] * np.shape(gt)[1]) - num_fixations)
            # number of values in r_sal_map, above the threshold, divided by num of random locations = num of fixations
            fp = len(np.where(r_sal_map > thresh)[0]) / (num_fixations * 1.0)

            area.append((round(tp, 4), round(fp, 4)))

        area.append((1.0, 1.0))
        area.sort(key=lambda x: x[0])
        tp_list = [x[0] for x in area]
        fp_list = [x[1] for x in area]

        aucs.append(np.trapz(np.array(tp_list), np.array(fp_list)))

    return np.mean(aucs)


class SAuc(object):
    def __init__(self):
        self.aucs = 0
        self.sum = 0
        # map = cv2.imread('D:\Master\gaze_estimation\\test\dataset\\test/other_map.jpg', cv2.IMREAD_GRAYSCALE)
        # map = cv2.resize(map, (224, 224))
        # self.other_map = torch.from_numpy(map[np.newaxis, :, :] / map.max())

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        aucs = auc_shuff(pred[0], gt[0])
        self.aucs += aucs
        self.sum += 1

    def compute(self):
        return self.aucs / self.sum

    def __str__(self):
        aucs = self.compute()
        return f'aucs: {aucs:.3f}'


class CC(object):
    def __init__(self):
        self.pearson = 0
        self.sum = 0

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        a = pearsonr(pred.flatten(), gt.flatten())
        a = torch.where(torch.isnan(a), torch.full_like(a, 0), a)
        a = a.mean()
        self.pearson += float(a)
        self.sum += 1

    def compute(self):
        return self.pearson / (self.sum + 1e-7)

    def __str__(self):
        cc = self.compute()
        return f'CC: {cc:.3f}'


class KLDivergence(object):
    def __init__(self, eps=1e-7):
        self.num = 0
        self.eps = eps
        self.kld = 0

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        P = pred
        P = P / (self.eps + torch.sum(P, dim=[1, 2, 3], keepdim=True))
        Q = gt
        Q = Q / (self.eps + torch.sum(Q, dim=[1, 2, 3], keepdim=True))

        R = Q * torch.log(self.eps + Q / (self.eps + P))
        R = R.sum()
        self.num += 1
        self.kld += float(R)

    def compute(self):
        return self.kld / (self.num + 1e-7)

    def __str__(self):
        kld = self.compute()
        return f'KLDivergence: {kld:.3f}'


class F1Score(object):
    """
    refer: https://github.com/xuebinqin/DIS/blob/main/IS-Net/basics.py
    """

    def __init__(self, threshold: float = 0.5):
        self.precision_cum = None
        self.recall_cum = None
        self.num_cum = None
        self.threshold = threshold

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        batch_size, c, h, w = gt.shape
        assert batch_size == 1, f"validation mode batch_size must be 1, but got batch_size: {batch_size}."
        resize_pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=False)
        gt_num = torch.sum(torch.gt(gt, self.threshold).float())

        pp = resize_pred[torch.gt(gt, self.threshold)]  # 对应预测map中GT为前景的区域
        nn = resize_pred[torch.le(gt, self.threshold)]  # 对应预测map中GT为背景的区域

        pp_hist = torch.histc(pp, bins=255, min=0.0, max=1.0)
        nn_hist = torch.histc(nn, bins=255, min=0.0, max=1.0)

        # Sort according to the prediction probability from large to small
        pp_hist_flip = torch.flipud(pp_hist)
        nn_hist_flip = torch.flipud(nn_hist)

        pp_hist_flip_cum = torch.cumsum(pp_hist_flip, dim=0)
        nn_hist_flip_cum = torch.cumsum(nn_hist_flip, dim=0)

        precision = pp_hist_flip_cum / (pp_hist_flip_cum + nn_hist_flip_cum + 1e-4)
        recall = pp_hist_flip_cum / (gt_num + 1e-4)

        if self.precision_cum is None:
            self.precision_cum = torch.full_like(precision, fill_value=0.)

        if self.recall_cum is None:
            self.recall_cum = torch.full_like(recall, fill_value=0.)

        if self.num_cum is None:
            self.num_cum = torch.zeros([1], dtype=gt.dtype, device=gt.device)

        self.precision_cum += precision
        self.recall_cum += recall
        self.num_cum += batch_size

    def compute(self):
        pre_mean = self.precision_cum / self.num_cum
        rec_mean = self.recall_cum / self.num_cum
        f1_mean = (1 + 1) * pre_mean * rec_mean / (1 * pre_mean + rec_mean + 1e-8)
        max_f1 = torch.amax(f1_mean).item()
        return max_f1

    def __str__(self):
        max_f1 = self.compute()
        return f'maxF1: {max_f1:.3f}'


class MeanAbsoluteError(object):
    def __init__(self):
        self.mae_list = []

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        batch_size, c, h, w = gt.shape
        assert batch_size == 1, f"validation mode batch_size must be 1, but got batch_size: {batch_size}."
        # resize_pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=False)
        error_pixels = torch.sum(torch.abs(gt - pred), dim=(1, 2, 3)) / (h * w)
        self.mae_list.extend(error_pixels.tolist())

    def compute(self):
        mae = sum(self.mae_list) / len(self.mae_list)
        return mae

    def __str__(self):
        mae = self.compute()
        return f'MAE: {mae:.3f}'


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
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

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
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
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
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
