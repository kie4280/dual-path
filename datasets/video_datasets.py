import os
from datasets.video_transforms import *
from datasets.kinetics import VideoClsDataset
from torchvision.datasets import Kinetics


def build_dataset(is_train, test_mode, args):
  if args.data_set == 'K400':
    mode = None
    anno_path = None
    if is_train == True:
      mode = 'train'
      anno_path = os.path.join(args.data_path, "annotations", 'train.csv')
    elif test_mode == True:
      mode = 'test'
      anno_path = os.path.join(args.data_path, "annotations", 'test.csv')
    else:
      mode = 'validation'
      anno_path = os.path.join(args.data_path, "annotations", 'val.csv')

    dataset = VideoClsDataset(
        anno_path=anno_path,
        data_path=args.data_path,
        mode=mode,
        clip_len=args.num_frames,
        frame_sample_rate=args.sampling_rate,
        num_segment=1,
        test_num_segment=args.test_num_segment,
        test_num_crop=args.test_num_crop,
        num_crop=1 if not test_mode else 3,
        keep_aspect_ratio=True,
        crop_size=args.input_size,
        short_side_size=args.short_side_size,
        new_height=256,
        new_width=320,
        args=args)

    # dataset = Kinetics(root=args.data_path,
    #                    download=(not os.path.exists(args.data_path)),
    #                    num_classes="400",
    #                    split=mode,
    #                    frames_per_clip=args.num_frames,
    #                    frame_rate=args.sampling_rate,
    #                    num_workers=args.num_workers,)

    nb_classes = 400

  elif args.data_set == 'SSV2':
    mode = None
    anno_path = None
    if is_train == True:
      mode = 'train'
      anno_path = os.path.join(args.data_path, "annotations", 'train.csv')
    elif test_mode == True:
      mode = 'test'
      anno_path = os.path.join(args.data_path, "annotations", 'train.csv')
    else:
      mode = 'validation'
      anno_path = os.path.join(args.data_path, "annotations", 'train.csv')

    dataset = VideoClsDataset(
        anno_path=anno_path,
        data_path=os.path.join(args.data_path, 'videos'),
        mode=mode,
        clip_len=args.num_frames,
        frame_sample_rate=args.sampling_rate,
        num_segment=1,
        test_num_segment=args.test_num_segment,
        test_num_crop=args.test_num_crop,
        num_crop=1 if not test_mode else 3,
        keep_aspect_ratio=True,
        crop_size=args.input_size,
        short_side_size=args.short_side_size,
        new_height=256,
        new_width=320,
        args=args)
    nb_classes = 174

  elif args.data_set == 'UCF101':
    mode = None
    anno_path = None
    if is_train == True:
      mode = 'train'
      anno_path = os.path.join(args.data_path, "annotations", 'train.csv')
    elif test_mode == True:
      mode = 'test'
      anno_path = os.path.join(args.data_path, "annotations", 'test.csv')
    else:
      mode = 'validation'
      anno_path = os.path.join(args.data_path, "annotations", 'test.csv')

    dataset = VideoClsDataset(
        anno_path=anno_path,
        data_path='./videos/',
        mode=mode,
        clip_len=args.num_frames,
        frame_sample_rate=args.sampling_rate,
        num_segment=1,
        test_num_segment=args.test_num_segment,
        test_num_crop=args.test_num_crop,
        num_crop=1 if not test_mode else 3,
        keep_aspect_ratio=True,
        crop_size=args.input_size,
        short_side_size=args.short_side_size,
        new_height=256,
        new_width=320,
        args=args)
    nb_classes = 101

  elif args.data_set == 'HMDB51':
    mode = None
    anno_path = None
    if is_train == True:
      mode = 'train'
      anno_path = os.path.join(args.data_path, 'train.csv')
    elif test_mode == True:
      mode = 'test'
      anno_path = os.path.join(args.data_path, 'test.csv')
    else:
      mode = 'validation'
      anno_path = os.path.join(args.data_path, 'val.csv')

    dataset = VideoClsDataset(
        anno_path=anno_path,
        data_path=os.path.join(args.data_path, 'videos'),
        mode=mode,
        clip_len=args.num_frames,
        frame_sample_rate=args.sampling_rate,
        num_segment=1,
        test_num_segment=args.test_num_segment,
        test_num_crop=args.test_num_crop,
        num_crop=1 if not test_mode else 3,
        keep_aspect_ratio=True,
        crop_size=args.input_size,
        short_side_size=args.short_side_size,
        new_height=256,
        new_width=320,
        args=args)
    nb_classes = 51

  elif args.data_set == 'Diving48':
    mode = None
    anno_path = None
    if is_train == True:
      mode = 'train'
      anno_path = os.path.join(args.data_path, 'train.csv')
    elif test_mode == True:
      mode = 'test'
      anno_path = os.path.join(args.data_path, 'test.csv')
    else:
      mode = 'validation'
      anno_path = os.path.join(args.data_path, 'test.csv')

    dataset = VideoClsDataset(
        anno_path=anno_path,
        data_path=os.path.join(args.data_path, 'videos'),
        mode=mode,
        clip_len=args.num_frames,
        frame_sample_rate=args.sampling_rate,
        num_segment=1,
        test_num_segment=args.test_num_segment,
        test_num_crop=args.test_num_crop,
        num_crop=1 if not test_mode else 3,
        keep_aspect_ratio=True,
        crop_size=args.input_size,
        short_side_size=args.short_side_size,
        new_height=256,
        new_width=320,
        args=args)
    nb_classes = 48

  else:
    raise NotImplementedError()
  assert nb_classes == args.nb_classes
  print("Number of the class = %d" % args.nb_classes)

  return dataset, nb_classes
