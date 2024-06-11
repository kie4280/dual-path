from argparse import ArgumentParser, Namespace


def to_dict(args: Namespace) -> dict:
  return vars(args)


def get_args_parser():
  parser = ArgumentParser(
      'AdaptFormer fine-tuning for action recognition',
      add_help=False)
  parser.add_argument('--batch_size', default=128, type=int,
                      help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
  parser.add_argument('--epochs', default=200, type=int)
  parser.add_argument('--accum_iter', default=1, type=int,
                      help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

  # Model parameters
  parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                      help='Name of model to train')
  # Optimizer parameters
  parser.add_argument('--weight_decay', type=float, default=5e-2,
                      help='weight decay (default: 0 for linear probe following MoCo v1)')
  parser.add_argument('--lr', type=float, default=None, metavar='LR',
                      help='learning rate (absolute lr)')
  parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                      help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

  parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                      help='lower lr bound for cyclic schedulers that hit 0')

  parser.add_argument('--warmup_epochs', type=int, default=3, metavar='N',
                      help='epochs to warmup LR')

  # * Finetuning params
  parser.add_argument('--finetune', default='./pre_checkpoint/mae_pretrain_vit_b.pth',
                      help='finetune from checkpoint')
  parser.add_argument('--global_pool', action='store_true')
  parser.set_defaults(global_pool=False)
  parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                      help='Use class token instead of global pool for classification')

  # Dataset parameters
  parser.add_argument('--data_path', default='../kinetics-dataset/k400', type=str,
                      help='dataset path')

  parser.add_argument('--output_dir', default='./output_dir',
                      help='path where to save, empty for no saving')
  parser.add_argument('--log_dir', default="./output_dir",
                      help='path where to tensorboard log')
  parser.add_argument('--device', default='cuda',
                      help='device to use for training / testing')
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--resume', default='',
                      help='resume from checkpoint')

  parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                      help='start epoch')
  parser.add_argument('--eval', action='store_true',
                      help='Perform evaluation only')
  parser.add_argument('--dist_eval', action='store_true', default=True,
                      help='Enabling distributed evaluation (recommended during training for faster monitor')
  parser.add_argument('--num_workers', default=4, type=int)
  parser.add_argument('--pin_mem', action='store_true',
                      help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
  parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
  parser.set_defaults(pin_mem=True)

  # distributed training parameters
  parser.add_argument('--distributed', default=True)
  parser.add_argument('--world_size', default=1, type=int,
                      help='number of distributed processes')
  parser.add_argument('--local_rank', default=-1, type=int)
  parser.add_argument('--dist_on_itp', action='store_true')
  parser.add_argument('--dist_url', default='env://',
                      help='url used to set up distributed training')

  # custom parameters
  parser.add_argument('--linprob', default=False)
  parser.add_argument('--tubelet_size', type=int, default=2)
  parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                      help='Dropout rate (default: 0.)')
  parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                      help='Attention dropout rate (default: 0.)')
  parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                      help='No drop path for linear probe')
  parser.add_argument('--use_mean_pooling', default=True)
  parser.add_argument('--init_scale', default=0.001, type=float)
  parser.add_argument("-v", "--verbose", action="store_true")

  # video data parameters
  parser.add_argument('--data_set', default='K400',
                      choices=[
                          'SSV2',
                          'HMDB51',
                          'Diving48',
                          'image_folder',
                          'K400',
                          'K700'],
                      type=str, help='dataset')
  parser.add_argument('--num_segments', type=int, default=1)
  parser.add_argument('--num_frames', type=int, default=16)
  parser.add_argument('--sampling_rate', type=int, default=4)
  parser.add_argument('--num_sample', type=int, default=1,
                      help='Repeated_aug (default: 1)')
  parser.add_argument('--crop_pct', type=float, default=None)
  parser.add_argument('--short_side_size', type=int, default=224)
  parser.add_argument('--test_num_segment', type=int, default=3)
  parser.add_argument('--test_num_crop', type=int, default=1)
  parser.add_argument(
      '--input_size',
      default=224,
      type=int,
      help='videos input size')
  parser.add_argument(
      '--reprob',
      default=0.0,
      type=float,
      help='random erase probability')
  parser.add_argument(
      '--remode',
      default='pixel',
      type=str,
      help='random erase mode')
  parser.add_argument(
      '--recount',
      default=56,
      type=int,
      help='maximum block size of erased region')
  parser.add_argument('--aa', default='rand', type=str,
                      help='random augmentation type')
  parser.add_argument(
      '--train_interpolation',
      default='bilinear',
      type=str,
      help='random augmentation interpolation')
  # AdaptFormer related parameters
  parser.add_argument(
      '--ffn_adapt',
      default=False,
      action='store_true',
      help='whether activate AdaptFormer')
  parser.add_argument('--ffn_num', default=64, type=int,
                      help='bottleneck middle dimension')
  parser.add_argument(
      '--vpt',
      default=False,
      action='store_true',
      help='whether activate VPT')
  parser.add_argument(
      '--vpt_num',
      default=8,
      type=int,
      help='number of VPT prompts')
  parser.add_argument(
      '--protune',
      default=False,
      help='whether activate Pro-Tuning')
  parser.add_argument(
      '--fulltune',
      default=False,
      action='store_true',
      help='full finetune model')
  parser.add_argument('--inception', default=False, action='store_true', help='whether use INCPETION mean and std'
                                                                              '(for Jx provided IN-21K pretrain')
  parser.add_argument("--wandb", action="store_true", help="use wandb to log training")

  return parser
