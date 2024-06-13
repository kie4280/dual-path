import wandb
import random
from argparse import Namespace
from .argument_parser import to_dict


def wandb_init(args: Namespace) -> None:
  if args.tags is not None:
    args.tags = str(args.tags).replace(" ", "").split(",")
  # start a new wandb run to track this script
  wandb.init(
      # set the wandb project where this run will be logged
      project="dual-path-broken",
      # track hyperparameters and run metadata
      config=to_dict(args),
      notes=args.note,
      tags=args.tags,
  )
  wandb.define_metric("eval/test_acc1")
  wandb.define_metric("eval/test_acc5")
  wandb.define_metric("eval/test_loss")
  wandb.define_metric("train/loss")
  wandb.define_metric("train/lr")


if __name__ == "__main__":

  # simulate training
  epochs = 10
  offset = random.random() / 5
  for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})
