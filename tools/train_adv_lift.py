import os
import time
import datetime
import random
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter

from utils.config import _C as cfg
from utils.logger import setup_logger
from utils.meter import AverageMeter
from trainer import Trainer


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
DEFAULT_MEAN = [0.5, 0.5, 0.5]
DEFAULT_STD = [0.5, 0.5, 0.5]


def set_seed(seed):
    if seed is None:
        return
    print("Setting fixed seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_mean_std(cfg, device):
    if cfg.backbone.startswith("CLIP"):
        mean, std = CLIP_MEAN, CLIP_STD
    else:
        mean, std = DEFAULT_MEAN, DEFAULT_STD
    mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return mean, std


def denormalize(x, mean, std):
    return x * std + mean


def normalize(x, mean, std):
    return (x - mean) / std


def project_linf(x_adv, x_orig, eps):
    return torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)


def pgd_linf_attack(model, criterion, images_norm, labels, mean, std,
                    eps, alpha, steps, random_start=True):
    """Generate L_inf adversarial examples in raw pixel space.

    Args:
        images_norm: normalized images consumed by the LIFT model.
        eps/alpha: raw-pixel scale, e.g. 1.0 / 255.
    Returns:
        normalized adversarial images.
    """
    images_raw = denormalize(images_norm.detach(), mean, std).clamp(0.0, 1.0)

    if random_start and eps > 0:
        x_adv = images_raw + torch.empty_like(images_raw).uniform_(-eps, eps)
        x_adv = project_linf(x_adv, images_raw, eps).clamp(0.0, 1.0)
    else:
        x_adv = images_raw.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(normalize(x_adv, mean, std))
        loss = criterion(logits, labels)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = project_linf(x_adv, images_raw, eps)
            x_adv = x_adv.clamp(0.0, 1.0)
        x_adv = x_adv.detach()

    return normalize(x_adv, mean, std).detach()


def train_adv(trainer, adv_eps, adv_alpha, adv_steps, adv_lambda,
              adv_random_start=True):
    cfg = trainer.cfg
    device = trainer.device
    mean, std = get_mean_std(cfg, device)

    writer_dir = os.path.join(cfg.output_dir, "tensorboard")
    os.makedirs(writer_dir, exist_ok=True)
    print(f"Initialize tensorboard (log_dir={writer_dir})")
    trainer._writer = SummaryWriter(log_dir=writer_dir)

    print("Adversarial fine-tuning setting:")
    print(f"  attack=PGD-{adv_steps}, norm=L_inf")
    print(f"  eps={adv_eps * 255:.4f}/255, alpha={adv_alpha * 255:.4f}/255")
    print(f"  random_start={adv_random_start}, adv_lambda={adv_lambda}")

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter(ema=True)
    clean_loss_meter = AverageMeter(ema=True)
    adv_loss_meter = AverageMeter(ema=True)
    acc_meter = AverageMeter(ema=True)
    cls_meters = [AverageMeter(ema=True) for _ in range(trainer.num_classes)]

    time_start = time.time()
    num_epochs = cfg.num_epochs

    trainer.optim.zero_grad()
    for epoch_idx in range(num_epochs):
        trainer.tuner.train()
        trainer.head.train()
        end = time.time()

        num_batches = len(trainer.train_loader)
        for batch_idx, batch in enumerate(trainer.train_loader):
            data_time.update(time.time() - end)

            image = batch[0].to(device)
            label = batch[1].to(device)

            x_adv = pgd_linf_attack(
                trainer.model,
                trainer.criterion,
                image,
                label,
                mean,
                std,
                eps=adv_eps,
                alpha=adv_alpha,
                steps=adv_steps,
                random_start=adv_random_start,
            )

            if cfg.prec == "amp":
                with autocast():
                    output_clean = trainer.model(image)
                    output_adv = trainer.model(x_adv)
                    loss_clean = trainer.criterion(output_clean, label)
                    loss_adv = trainer.criterion(output_adv, label)
                    loss = loss_clean + adv_lambda * loss_adv
                    loss_micro = loss / trainer.accum_step
                    trainer.scaler.scale(loss_micro).backward()
                if ((batch_idx + 1) % trainer.accum_step == 0) or (batch_idx + 1 == num_batches):
                    trainer.scaler.step(trainer.optim)
                    trainer.scaler.update()
                    trainer.optim.zero_grad()
            else:
                output_clean = trainer.model(image)
                output_adv = trainer.model(x_adv)
                loss_clean = trainer.criterion(output_clean, label)
                loss_adv = trainer.criterion(output_adv, label)
                loss = loss_clean + adv_lambda * loss_adv
                loss_micro = loss / trainer.accum_step
                loss_micro.backward()
                if ((batch_idx + 1) % trainer.accum_step == 0) or (batch_idx + 1 == num_batches):
                    trainer.optim.step()
                    trainer.optim.zero_grad()

            with torch.no_grad():
                pred = output_clean.argmax(dim=1)
                correct = pred.eq(label).float()
                acc = correct.mean().mul_(100.0)

            current_lr = trainer.optim.param_groups[0]["lr"]
            loss_meter.update(loss.item())
            clean_loss_meter.update(loss_clean.item())
            adv_loss_meter.update(loss_adv.item())
            acc_meter.update(acc.item())
            batch_time.update(time.time() - end)

            for _c, _y in zip(correct, label):
                cls_meters[_y].update(_c.mul_(100.0).item(), n=1)
            cls_accs = [cls_meters[i].avg for i in range(trainer.num_classes)]

            mean_acc = np.mean(np.array(cls_accs))
            many_acc = np.mean(np.array(cls_accs)[trainer.many_idxs])
            med_acc = np.mean(np.array(cls_accs)[trainer.med_idxs])
            few_acc = np.mean(np.array(cls_accs)[trainer.few_idxs])

            meet_freq = (batch_idx + 1) % cfg.print_freq == 0
            only_few_batches = num_batches < cfg.print_freq
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += num_batches - batch_idx - 1
                nb_remain += (num_epochs - epoch_idx - 1) * num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{epoch_idx + 1}/{num_epochs}]"]
                info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                info += [f"clean_loss {clean_loss_meter.val:.4f} ({clean_loss_meter.avg:.4f})"]
                info += [f"adv_loss {adv_loss_meter.val:.4f} ({adv_loss_meter.avg:.4f})"]
                info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
                info += [f"(mean {mean_acc:.4f} many {many_acc:.4f} med {med_acc:.4f} few {few_acc:.4f})"]
                info += [f"lr {current_lr:.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = epoch_idx * num_batches + batch_idx
            trainer._writer.add_scalar("train/lr", current_lr, n_iter)
            trainer._writer.add_scalar("train/loss.val", loss_meter.val, n_iter)
            trainer._writer.add_scalar("train/loss.avg", loss_meter.avg, n_iter)
            trainer._writer.add_scalar("train/clean_loss.val", clean_loss_meter.val, n_iter)
            trainer._writer.add_scalar("train/adv_loss.val", adv_loss_meter.val, n_iter)
            trainer._writer.add_scalar("train/acc.val", acc_meter.val, n_iter)
            trainer._writer.add_scalar("train/acc.avg", acc_meter.avg, n_iter)
            trainer._writer.add_scalar("train/mean_acc", mean_acc, n_iter)
            trainer._writer.add_scalar("train/many_acc", many_acc, n_iter)
            trainer._writer.add_scalar("train/med_acc", med_acc, n_iter)
            trainer._writer.add_scalar("train/few_acc", few_acc, n_iter)

            end = time.time()

        trainer.sched.step()
        torch.cuda.empty_cache()

    print("Finish adversarial fine-tuning")
    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print(f"Time elapsed: {elapsed}")

    trainer.save_model(cfg.output_dir)
    trainer.test()
    trainer._writer.close()


def main(args):
    cfg_data_file = os.path.join("./configs/data", args.data + ".yaml")
    cfg_model_file = os.path.join("./configs/model", args.model + ".yaml")

    cfg.defrost()
    cfg.merge_from_file(cfg_data_file)
    cfg.merge_from_file(cfg_model_file)
    cfg.merge_from_list(args.opts)

    if cfg.output_dir is None:
        cfg_name = "_".join([args.data, args.model])
        opts_name = "".join(["_" + item for item in args.opts])
        cfg.output_dir = os.path.join("./output", cfg_name + "_adv" + opts_name)
    else:
        cfg.output_dir = os.path.join("./output", cfg.output_dir)

    print("Output directory: {}".format(cfg.output_dir))
    setup_logger(cfg.output_dir)

    print("** Config **")
    print(cfg)
    print("************")

    set_seed(cfg.seed)

    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    trainer = Trainer(cfg)
    train_adv(
        trainer,
        adv_eps=args.adv_eps / 255.0,
        adv_alpha=args.adv_alpha / 255.0,
        adv_steps=args.adv_steps,
        adv_lambda=args.adv_lambda,
        adv_random_start=not args.no_random_start,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="", help="data config file")
    parser.add_argument("--model", "-m", type=str, default="", help="model config file")
    parser.add_argument("--adv_eps", type=float, default=1.0, help="L_inf epsilon in /255 units")
    parser.add_argument("--adv_alpha", type=float, default=1.0, help="PGD step size in /255 units")
    parser.add_argument("--adv_steps", type=int, default=2, help="PGD steps for adversarial training")
    parser.add_argument("--adv_lambda", type=float, default=1.0, help="weight of adversarial loss")
    parser.add_argument("--no_random_start", action="store_true", help="disable PGD random start")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="modify config options using the command-line")
    args = parser.parse_args()
    main(args)
