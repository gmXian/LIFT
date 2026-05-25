import argparse
import csv
import os
import random
import sys
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

try:
    import torchattacks
except ImportError as exc:
    raise ImportError(
        "torchattacks is required for adversarial evaluation. "
        "Please install it with `pip install torchattacks`."
    ) from exc

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import datasets
from trainer import Trainer
from utils.config import _C as default_cfg


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
GENERIC_MEAN = [0.5, 0.5, 0.5]
GENERIC_STD = [0.5, 0.5, 0.5]


class NormalizeThenModel(torch.nn.Module):
    """Wrap a LIFT model so attacks are performed in [0, 1] pixel space.

    LIFT dataloaders normally normalize images before forwarding them to the
    model. To match the DBD attack protocol, this wrapper lets torchattacks see
    unnormalized [0, 1] tensors while the wrapped model still receives normalized
    inputs.
    """

    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype)
        return self.model(x)


def setup_seed(seed, deterministic=False):
    if seed is None:
        return
    print(f"Setting fixed seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def build_cfg(args):
    cfg = default_cfg.clone()
    cfg_data_file = os.path.join(_REPO_ROOT, "configs/data", args.data + ".yaml")
    cfg_model_file = os.path.join(_REPO_ROOT, "configs/model", args.model + ".yaml")

    cfg.defrost()
    cfg.merge_from_file(cfg_data_file)
    cfg.merge_from_file(cfg_model_file)
    cfg.merge_from_list(args.opts)

    if args.root is not None:
        cfg.root = args.root
    if args.gpu is not None:
        cfg.gpu = args.gpu
    if args.seed is not None:
        cfg.seed = args.seed
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
        cfg.micro_batch_size = args.batch_size
    if args.model_dir is not None:
        cfg.model_dir = args.model_dir
    if args.zero_shot:
        cfg.zero_shot = True

    return cfg


def build_plain_test_loader(cfg):
    """Build an unnormalized single-crop test loader.

    This mirrors LIFT's default test crop geometry (Resize 8/7 * resolution and
    CenterCrop resolution), but omits Normalize so that the adversarial attack is
    applied in [0, 1] pixel space, following DBD's generate_adv_images.py.
    """

    transform = transforms.Compose([
        transforms.Resize(cfg.resolution * 8 // 7, interpolation=BICUBIC),
        transforms.CenterCrop(cfg.resolution),
        transforms.ToTensor(),
    ])
    dataset = getattr(datasets, cfg.dataset)(cfg.root, train=False, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args_batch_size(cfg),
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return loader


def args_batch_size(cfg):
    # LIFT uses batch_size for effective training batch size and 64 for testing.
    # For adversarial evaluation, exposing cfg.batch_size is more convenient.
    return int(cfg.batch_size) if int(cfg.batch_size) > 0 else 64


def make_attack(args, model):
    attack_name = args.attack.lower()
    eps = args.eps / 255.0
    alpha = args.alpha / 255.0

    if attack_name == "pgd":
        return torchattacks.PGD(model, eps=eps, alpha=alpha, steps=args.steps)
    if attack_name == "fgsm":
        return torchattacks.FGSM(model, eps=eps)
    if attack_name == "autoattack":
        return torchattacks.AutoAttack(model, eps=eps, version="standard", seed=args.seed or 0)
    if attack_name == "apgd-ce":
        return torchattacks.APGD(model, eps=eps, seed=args.seed or 0, loss="ce")
    if attack_name == "square":
        return torchattacks.Square(model, eps=eps, seed=args.seed or 0)
    raise NotImplementedError(f"Unsupported attack: {args.attack}")


def init_stats(num_classes):
    return {
        "total": np.zeros(num_classes, dtype=np.int64),
        "clean_correct": np.zeros(num_classes, dtype=np.int64),
        "adv_correct": np.zeros(num_classes, dtype=np.int64),
    }


def update_stats(stats, labels, clean_preds, adv_preds):
    labels = labels.detach().cpu().numpy()
    clean_preds = clean_preds.detach().cpu().numpy()
    adv_preds = adv_preds.detach().cpu().numpy()
    for y, pc, pa in zip(labels, clean_preds, adv_preds):
        y = int(y)
        stats["total"][y] += 1
        stats["clean_correct"][y] += int(int(pc) == y)
        stats["adv_correct"][y] += int(int(pa) == y)


def compute_group_summary(stats, many_idxs, med_idxs, few_idxs):
    total = stats["total"]
    clean_correct = stats["clean_correct"]
    adv_correct = stats["adv_correct"]

    clean_acc = clean_correct / np.maximum(total, 1)
    adv_acc = adv_correct / np.maximum(total, 1)

    def micro(idxs):
        idxs = np.asarray(idxs, dtype=np.int64)
        den = total[idxs].sum()
        if den == 0:
            return 0.0, 0.0
        return float(clean_correct[idxs].sum() / den), float(adv_correct[idxs].sum() / den)

    def macro(idxs):
        idxs = np.asarray(idxs, dtype=np.int64)
        if len(idxs) == 0:
            return 0.0, 0.0
        valid = total[idxs] > 0
        idxs = idxs[valid]
        if len(idxs) == 0:
            return 0.0, 0.0
        return float(clean_acc[idxs].mean()), float(adv_acc[idxs].mean())

    all_idxs = np.arange(len(total))
    groups = OrderedDict([
        ("all", all_idxs),
        ("many", many_idxs),
        ("medium", med_idxs),
        ("few", few_idxs),
    ])

    rows = []
    for name, idxs in groups.items():
        c_micro, a_micro = micro(idxs)
        c_macro, a_macro = macro(idxs)
        idxs = np.asarray(idxs, dtype=np.int64)
        valid = total[idxs] > 0
        valid_idxs = idxs[valid]
        worst_adv = float(adv_acc[valid_idxs].min()) if len(valid_idxs) else 0.0
        rows.append({
            "split": name,
            "num_classes": int(len(valid_idxs)),
            "num_samples": int(total[valid_idxs].sum()) if len(valid_idxs) else 0,
            "clean_micro": c_micro,
            "adv_micro": a_micro,
            "clean_macro": c_macro,
            "adv_macro": a_macro,
            "gap_micro": a_micro - c_micro,
            "gap_macro": a_macro - c_macro,
            "adv_worst": worst_adv,
        })
    return rows, clean_acc, adv_acc


def save_classwise_csv(path, stats, classnames, clean_acc, adv_acc):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        fields = [
            "class_id", "class_name", "test_count",
            "clean_correct", "adv_correct", "clean_acc", "adv_acc"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for class_id in range(len(stats["total"])):
            class_name = classnames[class_id] if class_id < len(classnames) else str(class_id)
            writer.writerow({
                "class_id": class_id,
                "class_name": class_name,
                "test_count": int(stats["total"][class_id]),
                "clean_correct": int(stats["clean_correct"][class_id]),
                "adv_correct": int(stats["adv_correct"][class_id]),
                "clean_acc": float(clean_acc[class_id]),
                "adv_acc": float(adv_acc[class_id]),
            })


def save_summary_csv(path, rows, args, cfg):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "model_dir", "dataset", "backbone", "attack", "eps", "alpha", "steps",
        "split", "num_classes", "num_samples",
        "clean_micro", "adv_micro", "clean_macro", "adv_macro",
        "gap_micro", "gap_macro", "adv_worst",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {
                "model_dir": args.model_dir if args.model_dir is not None else "zero_shot",
                "dataset": cfg.dataset,
                "backbone": cfg.backbone,
                "attack": args.attack,
                "eps": args.eps,
                "alpha": args.alpha,
                "steps": args.steps,
            }
            out.update(row)
            writer.writerow(out)


def main(args):
    cfg = build_cfg(args)
    setup_seed(cfg.seed, cfg.deterministic)

    trainer = Trainer(cfg)
    if not cfg.zero_shot:
        if cfg.model_dir is None:
            raise ValueError("--model_dir is required unless --zero_shot is set.")
        trainer.load_model(cfg.model_dir)

    if trainer.tuner is not None:
        trainer.tuner.eval()
    if trainer.head is not None:
        trainer.head.eval()
    trainer.model.eval()

    mean = CLIP_MEAN if cfg.backbone.startswith("CLIP") else GENERIC_MEAN
    std = CLIP_STD if cfg.backbone.startswith("CLIP") else GENERIC_STD
    wrapped_model = NormalizeThenModel(trainer.model, mean=mean, std=std).to(trainer.device)
    wrapped_model.eval()

    loader = build_plain_test_loader(cfg)
    attack = make_attack(args, wrapped_model)
    stats = init_stats(trainer.num_classes)

    print("Adversarial evaluation setting:")
    print(f"  dataset={cfg.dataset}, root={cfg.root}")
    print(f"  backbone={cfg.backbone}, model_dir={cfg.model_dir}")
    print(f"  attack={args.attack}, eps={args.eps}/255, alpha={args.alpha}/255, steps={args.steps}")
    print(f"  num_test_samples={len(loader.dataset)}, batch_size={loader.batch_size}")

    for batch_idx, batch in enumerate(loader):
        images, labels = batch[:2]
        images = images.to(trainer.device)
        labels = labels.to(trainer.device)

        with torch.no_grad():
            clean_logits = wrapped_model(images)
            clean_preds = clean_logits.argmax(dim=1)

        adv_images = attack(images, labels)
        with torch.no_grad():
            adv_logits = wrapped_model(adv_images)
            adv_preds = adv_logits.argmax(dim=1)

        update_stats(stats, labels, clean_preds, adv_preds)

        if (batch_idx + 1) % args.print_freq == 0:
            done = min((batch_idx + 1) * loader.batch_size, len(loader.dataset))
            clean_now = stats["clean_correct"].sum() / max(stats["total"].sum(), 1)
            adv_now = stats["adv_correct"].sum() / max(stats["total"].sum(), 1)
            print(f"[{done}/{len(loader.dataset)}] clean={clean_now:.4f}, adv={adv_now:.4f}")

    rows, clean_acc, adv_acc = compute_group_summary(
        stats,
        trainer.many_idxs,
        trainer.med_idxs,
        trainer.few_idxs,
    )

    print("Summary:")
    for row in rows:
        print(row)

    save_summary_csv(args.out_csv, rows, args, cfg)
    print(f"Summary saved to {args.out_csv}")

    if args.classwise_csv:
        save_classwise_csv(args.classwise_csv, stats, trainer.classnames, clean_acc, adv_acc)
        print(f"Class-wise results saved to {args.classwise_csv}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, required=True, help="data config file name")
    parser.add_argument("--model", "-m", type=str, required=True, help="model config file name")
    parser.add_argument("--model_dir", type=str, default=None, help="directory containing checkpoint.pth.tar")
    parser.add_argument("--root", type=str, default=None, help="dataset root override")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--zero_shot", action="store_true", default=False)
    parser.add_argument("--attack", type=str, default="pgd", choices=["pgd", "fgsm", "autoattack", "apgd-ce", "square"])
    parser.add_argument("--eps", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--classwise_csv", type=str, default="")
    parser.add_argument("--print_freq", type=int, default=20)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="modify config options using the command-line")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
