import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import datetime

import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import SVDD2024
from models.model import SVDDModel
from utils import seed_worker, set_seed, compute_eer


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, use_logits=True):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.use_logits = use_logits

    def forward(self, logits, targets):
        if self.use_logits:
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        else:
            bce = F.binary_cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt)**self.gamma * bce
        return loss.mean()


def collate_fn(batch):
    # Filter out None entries
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def main(args):
    set_seed(42)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    resume_training = args.load_from is not None

    # Datasets and loaders
    train_ds = SVDD2024(args.base_dir, partition="train")
    dev_ds   = SVDD2024(args.base_dir, partition="dev")
    test_ds  = SVDD2024(args.base_dir, partition="test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, worker_init_fn=seed_worker,
                              collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, worker_init_fn=seed_worker,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, worker_init_fn=seed_worker,
                             collate_fn=collate_fn)

    # Model, optimizer, scheduler
    model = SVDDModel(device, frontend=args.encoder).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    start_epoch = 0

    # Logging
    if resume_training:
        ckpt = torch.load(os.path.join(args.load_from, "checkpoints", "model_state.pt"))
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch']
        log_dir = args.load_from
    else:
        base = os.path.join(args.log_dir, args.encoder)
        os.makedirs(base, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(base, ts)
        os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save config
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        f.write(str(vars(args)))

    criterion = BinaryFocalLoss()
    best_val_eer = 1.0

    # Epoch loop
    for epoch in range(start_epoch, args.epochs):
        # Training
        model.train()
        pos_train, neg_train = [], []
        for i, batch in enumerate(tqdm(train_loader, desc=f"Train {epoch+1}/{args.epochs}")):
            if batch is None:
                continue
            x, label, _ = batch
            x, label = x.to(device), label.to(device)
            soft = label.float() * 0.9 + 0.05

            _, pred = model(x)
            loss = criterion(pred, soft.unsqueeze(1))

            pos_train.append(pred[label==1].detach().cpu().numpy())
            neg_train.append(pred[label==0].detach().cpu().numpy())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            step = epoch * len(train_loader) + i
            writer.add_scalar("Loss/train", loss.item(), step)

        scheduler.step()
        writer.add_scalar("LR/train", scheduler.get_last_lr()[0], epoch)

        if pos_train and neg_train:
            train_eer = compute_eer(np.concatenate(pos_train), np.concatenate(neg_train))[0]
            writer.add_scalar("EER/train", train_eer, epoch)

        # Save state
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, os.path.join(ckpt_dir, "model_state.pt"))

        # Validation
        model.eval()
        val_loss = 0.0
        pos_val, neg_val = [], []
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Validation"):
                if batch is None:
                    continue
                x, label, _ = batch
                x, label = x.to(device), label.to(device)
                _, pred = model(x)
                soft = label.float() * 0.9 + 0.05
                loss = criterion(pred, soft.unsqueeze(1))
                val_loss += loss.item()

                pos_val.append(pred[label==1].detach().cpu().numpy())
                neg_val.append(pred[label==0].detach().cpu().numpy())

        val_steps = max(1, sum(1 for _ in dev_loader if _ is not None))
        val_loss /= val_steps
        writer.add_scalar("Loss/val", val_loss, epoch)
        if pos_val and neg_val:
            val_eer = compute_eer(np.concatenate(pos_val), np.concatenate(neg_val))[0]
            writer.add_scalar("EER/val", val_eer, epoch)
        else:
            val_eer = float('inf')

        if val_eer < best_val_eer:
            best_val_eer = val_eer
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pt"))

            # Testing
            pos_test, neg_test = [], []
            for batch in tqdm(test_loader, desc="Testing"):
                if batch is None:
                    continue
                x, label, _ = batch
                x, label = x.to(device), label.to(device)
                _, pred = model(x)
                pos_test.append(pred[label==1].detach().cpu().numpy())
                neg_test.append(pred[label==0].detach().cpu().numpy())

            if pos_test and neg_test:
                test_eer = compute_eer(np.concatenate(pos_test), np.concatenate(neg_test))[0]
                writer.add_scalar("EER/test", test_eer, epoch)
                with open(os.path.join(log_dir, "test_eer.txt"), "w") as f:
                    f.write(f"At epoch {epoch}: {test_eer*100:.4f}%")

        # Snapshot every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_{epoch}_EER_{val_eer:.4f}.pt"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--encoder", type=str, default="rawnet")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--load_from", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
