import copy
import logging
import os
import os.path as osp
from os.path import join

import torch
from torch.utils.data import ConcatDataset, DataLoader

from utils.optimizer import create_optimizer
from utils.scheduler import create_scheduler

logger = logging.getLogger(__name__)


def get_media_types(datasources):
    """get the media types for for all the dataloaders.

    Args:
        datasources (List): List of dataloaders or datasets.

    Returns: List. The media_types.

    """
    if isinstance(datasources[0], DataLoader):
        datasets = [dataloader.dataset for dataloader in datasources]
    else:
        datasets = datasources
    media_types = [
        dataset.datasets[0].media_type
        if isinstance(dataset, ConcatDataset)
        else dataset.media_type
        for dataset in datasets
    ]

    return media_types


def setup_model(
    config, model_cls, find_unused_parameters=False
):
    logger.info("Creating model")
    config = copy.deepcopy(config)

    model = model_cls(config=config.model)
    print(model)
    # exit()

    model = model.to(torch.device(config.device))
    model_without_ddp = model
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.gpu],
            find_unused_parameters=find_unused_parameters,  # `False` for image-only task
        )

    optimizer = create_optimizer(config.optimizer, model)
    scheduler = create_scheduler(config.scheduler, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)

    start_epoch = 0
    global_step = 0

    # auto resume the latest checkpoint
    if config.get("auto_resume", False):
        logger.info("Auto resuming")
        model_latest = join(config.output_dir, "ckpt_latest.pth")
        model_best = join(config.output_dir, "ckpt_best.pth")
        large_num = -1
        for p in os.listdir(config.output_dir):
            if 'ckpt' in p:
                num = p.split('_')[1].split('.')[0]
                if str.isnumeric(num):
                    if int(num) > large_num:
                        large_num = int(num)
        if large_num != -1:
            model_latest = join(config.output_dir, f"ckpt_{large_num:02d}.pth")
        if osp.isfile(model_latest):
            config.pretrained_path = model_latest
            config.resume = True
        elif osp.isfile(model_best):
            config.pretrained_path = model_best
            config.resume = True
        else:
            logger.info(f"Not found checkpoint in {config.output_dir}")

    if osp.isfile(config.pretrained_path):
        checkpoint = torch.load(config.pretrained_path, map_location="cpu")
        state_dict = checkpoint["model"]

        if config.resume:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = checkpoint["epoch"] + 1
            global_step = checkpoint["global_step"]

        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"Loaded checkpoint from {config.pretrained_path}")
    else:
        logger.warning("No pretrained checkpoint provided, training from scratch")

    return (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        start_epoch,
        global_step,
    )


def check_ans(pred, gt):
    flag = False

    pred_list = pred.lower().split(' ')
    pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
    gt_list = gt.lower().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]

    if pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True

    return flag

def getCount(freq):
    count, total = freq[0], freq[1]
    return count / total if total != 0 else 0.0

def check_qtype_acc(type, eval, metric_logger, config):
    ep = 1e-10

    if config.dataset == 'nextqa':
        qtype2id = {'CH': 1, 'CW': 2, 'TN': 3, 'TC': 4, 'TP': 5, 'DL': 6, 'DC': 7, 'DO': 8}
    elif config.dataset == "star":
        qtype2id = {'Int': 1, 'Seq': 2, 'Pre': 3, 'Fea': 4}
    else:
        return

    q_freq = {i: [0., 0.] for i in qtype2id.values()}
    q_freq[0] = [0., 0.]
    for i, v in enumerate(eval):
        qt = qtype2id[type[i]]
        q_freq[qt][0] += v.item()  # acc
        q_freq[qt][1] += 1  # total
        q_freq[0][0] += v.item()  # all question
        q_freq[0][1] += 1

    if config.dataset == 'nextqa':
        metric_logger.update(n=(q_freq[1][1] + ep), CH=q_freq[1][0] / (q_freq[1][1] + ep))
        metric_logger.update(n=(q_freq[2][1] + ep), CW=q_freq[2][0] / (q_freq[2][1] + ep))
        metric_logger.update(n=(q_freq[3][1] + ep), TN=q_freq[3][0] / (q_freq[3][1] + ep))
        metric_logger.update(n=(q_freq[4][1] + ep), TC=q_freq[4][0] / (q_freq[4][1] + ep))
        metric_logger.update(n=(q_freq[5][1] + ep), TP=q_freq[5][0] / (q_freq[5][1] + ep))
        metric_logger.update(n=(q_freq[6][1] + ep), DL=q_freq[6][0] / (q_freq[6][1] + ep))
        metric_logger.update(n=(q_freq[7][1] + ep), DC=q_freq[7][0] / (q_freq[7][1] + ep))
        metric_logger.update(n=(q_freq[8][1] + ep), DO=q_freq[8][0] / (q_freq[8][1] + ep))

        metric_logger.update(n=(q_freq[1][1] + q_freq[2][1] + ep),
                             C=(q_freq[1][0] + q_freq[2][0]) / (q_freq[1][1] + q_freq[2][1] + ep))
        metric_logger.update(n=(q_freq[3][1] + q_freq[4][1] + q_freq[5][1] + ep),
                             T=(q_freq[3][0] + q_freq[4][0] + q_freq[5][0]) / (q_freq[3][1] + q_freq[4][1] + q_freq[5][1] + ep))
        metric_logger.update(n=(q_freq[6][1] + q_freq[7][1] + q_freq[8][1] + ep),
                             D=(q_freq[6][0] + q_freq[7][0] + q_freq[8][0]) / (q_freq[6][1] + q_freq[7][1] + q_freq[8][1] + ep))

        metric_logger.update(n=q_freq[0][1] + ep, Total=getCount(q_freq[0]))

    elif config.dataset == "star":
        metric_logger.update(n=q_freq[1][1] + ep, In=getCount(q_freq[1]))
        metric_logger.update(n=q_freq[2][1] + ep, Seq=getCount(q_freq[2]))
        metric_logger.update(n=q_freq[3][1] + ep, Pre=getCount(q_freq[3]))
        metric_logger.update(n=q_freq[4][1] + ep, Feas=getCount(q_freq[4]))
        metric_logger.update(n=q_freq[0][1] + ep, Total=getCount(q_freq[0]))