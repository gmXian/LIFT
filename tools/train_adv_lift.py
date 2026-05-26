import os
import time
import datetime
import random
import argparse
import shutil
import numpy as np
from collections import defaultdict
from PIL import Image

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import datasets
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
    """Generate L_inf adversarial examples in raw pixel space."""
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


class LTFileDataset(Dataset):
    """Dataset backed by an LT split txt file, e.g. ImageNet_LT_val.txt."""

    def __init__(self, root, txt_path, transform=None):
        self.root = root
        self.txt_path = txt_path
        self.transform = transform
        self.img_path = []
        self.labels = []

        with open(txt_path, "r") as f:
            for line in f:
                rel_path, label = line.strip().split()[:2]
                self.img_path.append(os.path.join(root, rel_path))
                self.labels.append(int(label))

        self.cls_num_list = self.get_cls_num_list()
        self.num_classes = len(self.cls_num_list)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]
        with open(path, "rb") as f:
            image = Image.open(f).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def get_cls_num_list(self):
        counter = defaultdict(int)
        for label in self.labels:
            counter[label] += 1
        return [counter[label] for label in sorted(counter.keys())]


def build_single_crop_eval_transform(cfg, mean, std):
    resolution = cfg.resolution
    return transforms.Compose([
        transforms.Resize(resolution * 8 // 7),
        transforms.CenterCrop(resolution),
        transforms.Lambda(lambda crop: torch.stack([transforms.ToTensor()(crop)])),
        transforms.Normalize(mean.flatten().cpu().tolist(), std.flatten().cpu().tolist()),
    ])


def stratified_holdout_indices(labels, val_fraction=0.1, seed=0):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train_indices, val_indices = [], []
    for c in sorted(np.unique(labels)):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        if len(idx) <= 1 or val_fraction <= 0:
            n_val = 0
        else:
            n_val = max(1, int(round(len(idx) * val_fraction)))
            n_val = min(n_val, len(idx) - 1)
        val_indices.extend(idx[:n_val].tolist())
        train_indices.extend(idx[n_val:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def maybe_setup_cifar_holdout_val(trainer, mean, std, val_fraction, val_seed):
    cfg = trainer.cfg
    if not cfg.dataset.startswith("CIFAR100_IR"):
        return
    if val_fraction <= 0:
        print("[warning] CIFAR-LT has no official val split and cifar_val_fraction <= 0; checkpoint selection will fall back to test_loader.")
        return

    train_dataset_aug = trainer.train_loader.dataset
    if not hasattr(train_dataset_aug, "labels"):
        print("[warning] CIFAR train dataset has no labels attribute; checkpoint selection will fall back to test_loader.")
        return

    train_indices, val_indices = stratified_holdout_indices(train_dataset_aug.labels, val_fraction, val_seed)
    if len(val_indices) == 0:
        print("[warning] Empty CIFAR holdout val split; checkpoint selection will fall back to test_loader.")
        return

    # Replace the training loader with the train subset to avoid selecting checkpoints
    # on examples that are still used for AFT updates.
    trainer.train_loader = DataLoader(
        Subset(train_dataset_aug, train_indices),
        batch_size=cfg.micro_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Recompute class counts for the held-out training subset and rebuild the loss
    # so LA/CB-style losses use the actual AFT training distribution.
    train_labels = np.asarray(train_dataset_aug.labels)[train_indices]
    cls_num_list = []
    for c in range(trainer.num_classes):
        cls_num_list.append(int((train_labels == c).sum()))
    trainer.cls_num_list = cls_num_list
    trainer.build_criterion()

    # Build a deterministic eval-view dataset with the same imbalanced CIFAR subset.
    eval_transform = build_single_crop_eval_transform(cfg, mean, std)
    eval_dataset = getattr(datasets, cfg.dataset)(cfg.root, train=True, transform=eval_transform)
    trainer.checkpoint_val_loader = DataLoader(
        Subset(eval_dataset, val_indices),
        batch_size=64,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    print(
        f"Checkpoint selection uses deterministic {cfg.dataset} holdout val split: "
        f"fraction={val_fraction}, seed={val_seed}, train={len(train_indices)}, val={len(val_indices)}"
    )


def build_checkpoint_val_loader(trainer, mean, std):
    """Build validation loader for checkpoint selection."""
    cfg = trainer.cfg
    if hasattr(trainer, "checkpoint_val_loader"):
        return trainer.checkpoint_val_loader

    if cfg.dataset == "ImageNet_LT":
        val_txt = "./datasets/ImageNet_LT/ImageNet_LT_val.txt"
        if os.path.exists(val_txt):
            transform = build_single_crop_eval_transform(cfg, mean, std)
            val_dataset = LTFileDataset(cfg.root, val_txt, transform=transform)
            val_loader = DataLoader(
                val_dataset,
                batch_size=64,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )
            print(f"Checkpoint selection uses ImageNet-LT val split: {val_txt}")
            return val_loader
        print(f"[warning] ImageNet-LT val split not found at {val_txt}; falling back to test_loader.")

    print("[warning] No dedicated val split is configured for this dataset; falling back to test_loader for checkpoint selection.")
    return trainer.test_loader


@torch.no_grad()
def forward_eval_crops(model, image):
    if image.dim() == 4:
        return model(image)

    _bsz, _ncrops, _c, _h, _w = image.size()
    image_flat = image.view(_bsz * _ncrops, _c, _h, _w)
    if _ncrops <= 5:
        output = model(image_flat)
        output = output.view(_bsz, _ncrops, -1).mean(dim=1)
    else:
        outputs = []
        image_view = image.view(_bsz, _ncrops, _c, _h, _w)
        for k in range(_ncrops):
            outputs.append(model(image_view[:, k]))
        output = torch.stack(outputs).mean(dim=0)
    return output


def evaluate_checkpoint_metrics(trainer, val_loader, mean, std,
                                adv_eps, adv_alpha, adv_steps, adv_random_start=True):
    if trainer.tuner is not None:
        trainer.tuner.eval()
    if trainer.head is not None:
        trainer.head.eval()

    device = trainer.device
    clean_correct = 0
    adv_correct = 0
    total = 0

    for batch in val_loader:
        image = batch[0].to(device)
        label = batch[1].to(device)

        if image.dim() == 5:
            image_for_adv = image[:, 0]
        else:
            image_for_adv = image

        with torch.no_grad():
            output_clean = forward_eval_crops(trainer.model, image)
            clean_correct += output_clean.argmax(dim=1).eq(label).sum().item()

        x_adv = pgd_linf_attack(
            trainer.model,
            trainer.criterion,
            image_for_adv,
            label,
            mean,
            std,
            eps=adv_eps,
            alpha=adv_alpha,
            steps=adv_steps,
            random_start=adv_random_start,
        )
        with torch.no_grad():
            output_adv = trainer.model(x_adv)
            adv_correct += output_adv.argmax(dim=1).eq(label).sum().item()

        total += label.numel()

    clean_acc = 100.0 * clean_correct / max(total, 1)
    adv_acc = 100.0 * adv_correct / max(total, 1)

    if trainer.tuner is not None:
        trainer.tuner.train()
    if trainer.head is not None:
        trainer.head.train()

    return clean_acc, adv_acc


def checkpoint_state(trainer, epoch, best_clean, best_robust):
    return {
        "epoch": epoch,
        "tuner": trainer.tuner.state_dict(),
        "head": trainer.head.state_dict(),
        "best_clean": best_clean,
        "best_robust": best_robust,
        "optimizer": trainer.optim.state_dict(),
        "scheduler": trainer.sched.state_dict(),
    }


def save_checkpoint_files(trainer, epoch, best_clean, best_robust,
                          is_best_clean=False, is_best_robust=False):
    os.makedirs(trainer.cfg.output_dir, exist_ok=True)
    latest_path = os.path.join(trainer.cfg.output_dir, "checkpoint.pth.tar")
    state = checkpoint_state(trainer, epoch, best_clean, best_robust)
    torch.save(state, latest_path)

    if is_best_clean:
        clean_path = os.path.join(trainer.cfg.output_dir, "model_best_clean.pth.tar")
        shutil.copyfile(latest_path, clean_path)
        print(f"Saved best clean checkpoint to {clean_path}")

    if is_best_robust:
        robust_path = os.path.join(trainer.cfg.output_dir, "model_best_robust.pth.tar")
        shutil.copyfile(latest_path, robust_path)
        print(f"Saved best robust checkpoint to {robust_path}")


def train_adv(trainer, adv_eps, adv_alpha, adv_steps, adv_lambda,
              adv_random_start=True, val_adv_eps=None, val_adv_alpha=None,
              val_adv_steps=None):
    cfg = trainer.cfg
    device = trainer.device
    mean, std = get_mean_std(cfg, device)
    val_adv_eps = adv_eps if val_adv_eps is None else val_adv_eps
    val_adv_alpha = adv_alpha if val_adv_alpha is None else val_adv_alpha
    val_adv_steps = adv_steps if val_adv_steps is None else val_adv_steps
    val_loader = build_checkpoint_val_loader(trainer, mean, std)

    writer_dir = os.path.join(cfg.output_dir, "tensorboard")
    os.makedirs(writer_dir, exist_ok=True)
    print(f"Initialize tensorboard (log_dir={writer_dir})")
    trainer._writer = SummaryWriter(log_dir=writer_dir)

    print("Adversarial fine-tuning setting:")
    print(f"  train attack=PGD-{adv_steps}, norm=L_inf")
    print(f"  train eps={adv_eps * 255:.4f}/255, alpha={adv_alpha * 255:.4f}/255")
    print(f"  random_start={adv_random_start}, adv_lambda={adv_lambda}")
    print("Validation checkpoint selection setting:")
    print("  val clean metric: clean accuracy on val split")
    print(f"  val robust metric: PGD-{val_adv_steps}, eps={val_adv_eps * 255:.4f}/255, alpha={val_adv_alpha * 255:.4f}/255")

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter(ema=True)
    clean_loss_meter = AverageMeter(ema=True)
    adv_loss_meter = AverageMeter(ema=True)
    acc_meter = AverageMeter(ema=True)
    cls_meters = [AverageMeter(ema=True) for _ in range(trainer.num_classes)]

    best_clean = -1.0
    best_robust = -1.0
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

        val_clean, val_robust = evaluate_checkpoint_metrics(
            trainer,
            val_loader,
            mean,
            std,
            adv_eps=val_adv_eps,
            adv_alpha=val_adv_alpha,
            adv_steps=val_adv_steps,
            adv_random_start=adv_random_start,
        )
        is_best_clean = val_clean > best_clean
        is_best_robust = val_robust > best_robust
        best_clean = max(best_clean, val_clean)
        best_robust = max(best_robust, val_robust)

        trainer._writer.add_scalar("val/clean_acc", val_clean, epoch_idx + 1)
        trainer._writer.add_scalar("val/robust_acc", val_robust, epoch_idx + 1)
        print(
            f"Epoch [{epoch_idx + 1}/{num_epochs}] val_clean={val_clean:.4f} "
            f"val_robust={val_robust:.4f} best_clean={best_clean:.4f} "
            f"best_robust={best_robust:.4f}"
        )
        save_checkpoint_files(
            trainer,
            epoch=epoch_idx + 1,
            best_clean=best_clean,
            best_robust=best_robust,
            is_best_clean=is_best_clean,
            is_best_robust=is_best_robust,
        )

    print("Finish adversarial fine-tuning")
    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print(f"Time elapsed: {elapsed}")

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
    mean, std = get_mean_std(cfg, trainer.device)
    maybe_setup_cifar_holdout_val(trainer, mean, std, args.cifar_val_fraction, args.cifar_val_seed)

    train_adv(
        trainer,
        adv_eps=args.adv_eps / 255.0,
        adv_alpha=args.adv_alpha / 255.0,
        adv_steps=args.adv_steps,
        adv_lambda=args.adv_lambda,
        adv_random_start=not args.no_random_start,
        val_adv_eps=args.val_adv_eps / 255.0,
        val_adv_alpha=args.val_adv_alpha / 255.0,
        val_adv_steps=args.val_adv_steps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="", help="data config file")
    parser.add_argument("--model", "-m", type=str, default="", help="model config file")
    parser.add_argument("--adv_eps", type=float, default=1.0, help="L_inf epsilon in /255 units")
    parser.add_argument("--adv_alpha", type=float, default=1.0, help="PGD step size in /255 units")
    parser.add_argument("--adv_steps", type=int, default=2, help="PGD steps for adversarial training")
    parser.add_argument("--adv_lambda", type=float, default=1.0, help="weight of adversarial loss")
    parser.add_argument("--val_adv_eps", type=float, default=1.0, help="L_inf epsilon in /255 units for robust val checkpoint selection")
    parser.add_argument("--val_adv_alpha", type=float, default=1.0, help="PGD step size in /255 units for robust val checkpoint selection")
    parser.add_argument("--val_adv_steps", type=int, default=2, help="PGD steps for robust val checkpoint selection")
    parser.add_argument("--cifar_val_fraction", type=float, default=0.1, help="stratified holdout fraction from CIFAR-LT train set for checkpoint selection")
    parser.add_argument("--cifar_val_seed", type=int, default=0, help="seed for deterministic CIFAR-LT holdout split")
    parser.add_argument("--no_random_start", action="store_true", help="disable PGD random start")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="modify config options using the command-line")
    args = parser.parse_args()
    main(args)
