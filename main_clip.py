# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main_clip.py --blr 0.03 --data_set 'HMDB51' --batch_size 32 --epochs 50
# python -m torch.distributed.launch --nproc_per_node=2 --use_env main_clip.py --blr 0.00035 --data_set 'SSV2' --batch_size 43 --epochs 60 --data_path '../Dataset/01_SSv2' --resume './output_dir/checkpoint-40.pth'
# python -m torch.distributed.launch --nproc_per_node=3 --use_env main_clip.py --blr 0.0007 --data_set 'Diving48' --batch_size 43 --epochs 50 --resume './output_dir/checkpoint-40.pth'
# python -m torch.distributed.launch --nproc_per_node=2 --use_env
# main_clip.py --blr 0.0005 --data_set 'SSV2' --batch_size 43 --epochs 60
# --data_path '../Dataset/01_SSv2' --output_dir './output_dir_SSv2'
# --resume './output_dir_SSv2/checkpoint-10.pth'

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from collections import OrderedDict
from easydict import EasyDict
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datasets.video_datasets import build_dataset
from datasets.kinetics import build_training_dataset

# assert timm.__version__ == "0.3.2" # version check
import util.misc as misc
from util.pos_embed import interpolate_pos_embed_ori as interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.argument_parser import get_args_parser


from engine_finetune_clip import train_one_epoch, evaluate
from engine_finetune_clip import merge, final_test
import clip.model
from CLIP_custom.clip import clip as clip


def convert_weights(model: nn.Module):
  """Convert applicable model parameters to fp16"""

  def _convert_weights_to_fp16(l):
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
      l.weight.data = l.weight.data.half()
      if l.bias is not None:
        l.bias.data = l.bias.data.half()

    if isinstance(l, nn.MultiheadAttention):
      for attr in [*[f"{s}_proj_weight" for s in ["in", "q",
                                                  "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
        tensor = getattr(l, attr)
        if tensor is not None:
          tensor.data = tensor.data.half()

    for name in ["text_projection", "proj"]:
      if hasattr(l, name):
        attr = getattr(l, name)
        if attr is not None:
          attr.data = attr.data.half()

  model.apply(_convert_weights_to_fp16)


def construct_optimizer(model, args):
  # Batchnorm parameters.
  bn_params = []
  # Non-batchnorm parameters.
  non_bn_parameters = []
  for name, p in model.named_parameters():
    if p.requires_grad:
      if "bn" in name:
        bn_params.append(p)
      else:
        non_bn_parameters.append(p)
  optim_params = [
      {"params": bn_params, "weight_decay": 0.},
      {"params": non_bn_parameters, "weight_decay": args.weight_decay},
  ]
  return torch.optim.AdamW(
      optim_params,
      lr=args.lr, weight_decay=args.weight_decay,
  )


class LabelSmoothLoss(torch.nn.Module):

  def __init__(self, smoothing=0.2):
    super(LabelSmoothLoss, self).__init__()
    self.smoothing = smoothing

  def forward(self, input, target):
    log_prob = F.log_softmax(input, dim=-1)
    weight = input.new_ones(input.size()) * \
        self.smoothing / (input.size(-1) - 1.)
    weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
    loss = (-weight * log_prob).sum(dim=-1).mean()
    return loss


def main(args):
  if args.log_dir is None:
    args.log_dir = args.output_dir
  misc.init_distributed_mode(args)
  print('here')

  print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
  print("{}".format(args).replace(', ', ',\n'))

  device = torch.device(args.device)

  # fix the seed for reproducibility
  seed = args.seed + misc.get_rank()
  torch.manual_seed(seed)
  np.random.seed(seed)

  cudnn.benchmark = True

  if args.data_set == 'SSV2':
    args.nb_classes = 174
    args.reprob = 0.1
    args.test_num_segment = 1
    args.test_num_crop = 3
  elif args.data_set == 'K400':
    args.nb_classes = 400
    args.reprob = 0.0
    args.test_num_segment = 3
    args.test_num_crop = 1
  elif args.data_set == 'HMDB51':
    args.nb_classes = 51
    args.reprob = 0.5
  elif args.data_set == 'Diving48':
    args.nb_classes = 48
    args.reprob = 0.1
    args.test_num_segment = 3
  else:
    raise ValueError(args.data_set)
  dataset_train, _ = build_dataset(is_train=True, test_mode=False, args=args)
  dataset_val, _ = build_dataset(is_train=False, test_mode=False, args=args)
  dataset_test, _ = build_dataset(is_train=False, test_mode=True, args=args)

  if True:  # args.distributed:
    print("Distributed On!!!!!!!!!!!")
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
      if len(dataset_val) % num_tasks != 0:
        print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
              'This will slightly alter validation results as extra duplicate entries are added to achieve '
              'equal num of samples per-process.')
      sampler_val = torch.utils.data.DistributedSampler(
          dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
      sampler_test = torch.utils.data.DistributedSampler(
          dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
      sampler_val = torch.utils.data.SequentialSampler(dataset_val)
      sampler_test = torch.utils.data.SequentialSampler(dataset_test)

  if global_rank == 0 and args.log_dir is not None and not args.eval:
    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)
  else:
    log_writer = None

  data_loader_train = torch.utils.data.DataLoader(
      dataset_train, sampler=sampler_train,
      batch_size=args.batch_size,
      num_workers=args.num_workers,
      pin_memory=args.pin_mem,
      drop_last=True,
  )

  data_loader_val = torch.utils.data.DataLoader(
      dataset_val, sampler=sampler_val,
      batch_size=args.batch_size,
      num_workers=args.num_workers,
      pin_memory=args.pin_mem,
      drop_last=False
  )

  data_loader_test = torch.utils.data.DataLoader(
      dataset_test, sampler=sampler_test,
      batch_size=args.batch_size,
      num_workers=args.num_workers,
      pin_memory=args.pin_mem,
      drop_last=False
  )

  # fine-tuning configs
  tuning_config = EasyDict(
      # AdaptFormer
      ffn_adapt=args.ffn_adapt,
      ffn_option="parallel",
      ffn_adapter_layernorm_option="none",
      ffn_adapter_init_option="lora",
      ffn_adapter_scalar="0.1",
      ffn_num=args.ffn_num,
      d_model=768,
      # VPT related
      vpt_on=args.vpt,
      vpt_num=args.vpt_num,
      protune=args.protune,
  )

  num_frame = int(args.num_frames / 16) + 8
  model, preprocess = clip.load(
      "ViT-B/16", device=device, num_classes=args.nb_classes, num_frame=num_frame)

  if args.fulltune:
    for name, p in model.named_parameters():
      if name.startswith('visual.'):
        p.requires_grad = True

  # if args.fulltune:
  #     for name, p in model.named_parameters():
  #         if name.startswith('visual.'):
  #             p.requires_grad = True
  # elif not args.fulltune:
  #     for _, p in model.named_parameters():
  #         p.requires_grad = False
  # model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)

  model_without_ddp = model
  n_model_parameters = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
  n_parameters = n_model_parameters  # + n_head_parameters

  print('number of whole params (M): %.2f' % (n_parameters / 1.e6))

  eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

  if args.lr is None:  # only base_lr is specified
    args.lr = args.blr * eff_batch_size / 256

  print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
  print("actual lr: %.2e" % args.lr)

  print("accumulate grad iterations: %d" % args.accum_iter)
  print("effective batch size: %d" % eff_batch_size)

  if args.distributed:
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu])
    model_without_ddp = model.module
  optimizer = construct_optimizer(model_without_ddp, args)
  print(optimizer)
  loss_scaler = NativeScaler()

  criterion = torch.nn.CrossEntropyLoss()

  print("criterion = %s" % str(criterion))

  misc.load_model(
      args=args,
      model_without_ddp=model_without_ddp,
      optimizer=optimizer,
      loss_scaler=loss_scaler)

  if args.eval:
    preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
    test_stats = final_test(
        data_loader=data_loader_test,
        model=model,
        device=device,
        file=preds_file,
        args=args)
    torch.distributed.barrier()
    if global_rank == 0:
      print("Start merging results...")
      final_top1, final_top5 = merge(
          args.output_dir, num_tasks, is_hmdb=args.data_set == 'HMDB51')
      print(
          f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
      log_stats = {'Final top-1': final_top1, 'Final Top-5': final_top1}
      if args.output_dir and misc.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
          f.write(json.dumps(log_stats) + "\n")
    exit(0)

  print(f"Start training for {args.epochs} epochs")
  start_time = time.time()
  max_accuracy = 0.0

  for epoch in range(args.start_epoch, args.epochs):
    if args.distributed:
      data_loader_train.sampler.set_epoch(epoch)
    train_stats = train_one_epoch(
        model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler,
        max_norm=None,
        log_writer=log_writer,
        args=args
    )
    if args.output_dir:
      misc.save_model(
          args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
          loss_scaler=loss_scaler, epoch=epoch)

    test_stats = evaluate(data_loader_val, model, device)
    print(
        f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    max_accuracy = max(max_accuracy, test_stats["acc1"])
    print(f'Max accuracy: {max_accuracy:.2f}%')

    if log_writer is not None:
      log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
      log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
      log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                 **{f'test_{k}': v for k, v in test_stats.items()},
                 'epoch': epoch,
                 'n_parameters': n_parameters}

    if args.output_dir and misc.is_main_process():
      if log_writer is not None:
        log_writer.flush()
      with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        f.write(json.dumps(log_stats) + "\n")

  preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
  test_stats = final_test(data_loader_test, model, device, preds_file, args)
  torch.distributed.barrier()
  if global_rank == 0:
    print("Start merging results...")
    final_top1, final_top5 = merge(
        args.output_dir, num_tasks, is_hmdb=args.data_set == 'HMDB51')
    print(
        f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
    log_stats = {'Final top-1': final_top1, 'Final Top-5': final_top5}
    if args.output_dir and misc.is_main_process():
      with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        f.write(json.dumps(log_stats) + "\n")

  total_time = time.time() - start_time
  total_time_str = str(datetime.timedelta(seconds=int(total_time)))
  print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
  args = get_args_parser()
  args = args.parse_args()
  if args.output_dir:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
  main(args)
