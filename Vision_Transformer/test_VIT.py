## PyTorch
import os
import argparse
import numpy as np
import random
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()
import ml_collections
from Transformer import VisionTransformer
from Dataloader import get_loader
from tqdm import tqdm

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    print("Saved model checkpoint to [DIR: %s]", args.output_dir)

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patche_size = 16 #patch size
    config.embed_size = 768 # the 
    config.feedforward_dim = 3072 # size for the feedforward layer
    config.num_heads = 12 # number of attention head 
    config.num_layers = 12 # number of attention layers 
    config.attention_dropout_rate = 0.0
    config.dropout_rate = 0.1
    config.classifier = 'token'
    return config


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


def valid(args, model, test_loader):
    model.eval()
    val_loss = AverageMeter()
    all_preds, all_label = [], []

    for step, batch in enumerate(tqdm(test_loader)):
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        preds, loss, _ = model.calculate_loss(batch, mode="valid")

        val_loss.update(loss.item())
        
        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = (all_preds == all_label).mean()

    return accuracy



def train_all(model, args):
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    train_loader, test_loader = get_loader(args)
    
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    T_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = optim.WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=T_total)
    else:
        scheduler = optim.WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=T_total)
    
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    optimizer.zero_grad()
    model.zero_grad()
    # lll
    for e in range(0, T_total):
        for step, batch in enumerate(tqdm(train_loader)):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            loss = model.calculate_loss(batch, mode="train")
            losses.update(loss.item())  

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward() # accumulate the loss gradient

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # update the model para
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
        print("Train loss in Epoch %d, %.3f"%(e, losses.avg))
        with torch.no_grad():
            accuracy = valid(args, model, test_loader)
            print("Validate loss and accuracy in Epoch %d, (%.3f, %.3f)"%(e, losses.avg, accuracy))
        if accuracy > best_acc:
            save_model(args, model)
            best_acc = accuracy
        losses.reset()
        

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    num_classes = 10 if args.dataset == "cifar10" else 100
    config = get_b16_config()

    NUM_PATCH  = (args.img_size//config.patch_size)*(args.img_size//config.patch_size)
    model = VisionTransformer( embed_dim=config.embed_dim, hidden_dim = config.feedforward_dim, num_channels=3, num_heads=config.num_heads, num_layers=config.num_layers, num_classes = num_classes, patch_size= config.patch_size, num_patches = NUM_PATCH, dropout = config.dropout_rate)
    #model.load_from(np.load(args.pretrained_dir))
    model.to(device)
    num_params = count_parameters(model)
    print("Number of parameters: ", num_params)




if __name__ == "__main__":
    main()