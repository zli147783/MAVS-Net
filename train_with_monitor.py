#!/usr/bin/env python3
import os
import argparse
import torch
from config import Config
from torch.utils.tensorboard import SummaryWriter
from datasets.CorresDataset import CorrespondencesDataset, collate_fn
from torch.utils.data import DataLoader
from models.mavs_net import MAVSNet
import torch.optim as optim
from utils.train_eval_utils import train_one_epoch, evaluate
from utils.tools import safe_load_weights


def parse_args():
    parser = argparse.ArgumentParser(description='MAVS-Net training with live monitoring')
    parser.add_argument('--batch_size', type=int, default=20, help='training batch size')
    parser.add_argument('--epochs', type=int, default=None, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--num_workers', type=int, default=None, help='number of dataloader workers')
    parser.add_argument('--log_dir', type=str, default=None, help='tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='checkpoint directory')
    parser.add_argument('--best_model_dir', type=str, default=None, help='best model directory')
    parser.add_argument('--resume', type=str, default=None, help='resume checkpoint file path')
    parser.add_argument('--no_tensorboard', action='store_true', help='disable tensorboard logging')
    parser.add_argument('--screen', action='store_true', help='print screen/tmux usage tips after starting')
    return parser.parse_args()


def make_dir(path):
    if path is not None and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()
    conf = Config()

    if args.epochs is not None:
        conf.epochs = args.epochs
    if args.lr is not None:
        conf.canonical_lr = args.lr
    if args.num_workers is not None:
        conf.num_workers = args.num_workers
    if args.log_dir is not None:
        conf.writer_dir = args.log_dir
    if args.checkpoint_dir is not None:
        conf.checkpoint_path = args.checkpoint_dir
    if args.best_model_dir is not None:
        conf.best_model_path = args.best_model_dir
    if args.resume is not None:
        conf.resume = args.resume

    make_dir(conf.checkpoint_path)
    make_dir(conf.best_model_path)
    make_dir(conf.writer_dir)

    if not args.no_tensorboard:
        print('Start Tensorboard with: tensorboard --logdir={} --host=0.0.0.0'.format(conf.writer_dir))
        tb_writer = SummaryWriter(log_dir=conf.writer_dir)
    else:
        tb_writer = None

    log_file_path = os.path.join(conf.writer_dir, 'train.log')
    log_file = open(log_file_path, 'a', encoding='utf-8')

    train_dataset = CorrespondencesDataset(conf.data_tr, conf)
    valid_dataset = CorrespondencesDataset(conf.data_va, conf)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=conf.num_workers,
                              collate_fn=collate_fn)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=1,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=conf.num_workers,
                              collate_fn=collate_fn)

    model = MAVSNet(conf).cuda()

    pg = [p for p in model.parameters() if p.requires_grad]
    true_lr = conf.canonical_lr * (args.batch_size / conf.canonical_bs)
    optimizer = optim.Adam(pg, lr=true_lr, weight_decay=conf.weight_decay)

    best_auc = -1
    start_epoch = 0

    if os.path.exists(conf.resume):
        print('Resuming from checkpoint:', conf.resume)
        weights_dict = torch.load(conf.resume, map_location='cuda')
        best_auc = weights_dict.get('best_auc', best_auc)
        start_epoch = weights_dict.get('epoch', 0) + 1
        safe_load_weights(model, weights_dict['state_dict'])

    print('Training configuration:')
    print('  epochs:', conf.epochs)
    print('  batch_size:', args.batch_size)
    print('  lr:', true_lr)
    print('  checkpoint_path:', conf.checkpoint_path)
    print('  best_model_path:', conf.best_model_path)
    print('  tensorboard logdir:', conf.writer_dir)
    print('  resume checkpoint:', conf.resume if os.path.exists(conf.resume) else 'None')
    if args.screen:
        print('\nTip: run this script inside screen or tmux to detach safely:')
        print('  screen -S mavs_net')
        print('  python train_with_monitor.py ...')
        print('  Ctrl-A D to detach, then screen -r mavs_net to reconnect\n')

    for epoch in range(start_epoch, conf.epochs):
        cur_global_step = epoch * len(train_dataset)

        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    conf=conf,
                                    device='cuda',
                                    epoch=epoch,
                                    cur_global_step=cur_global_step,
                                    tb_writer=tb_writer,
                                    log_file=log_file)

        aucs5, aucs10, aucs20, va_res, precisions, recalls, f_scores = evaluate(model, valid_loader, conf, epoch=epoch)

        print('[Epoch {}] train_loss={:.4f} | AUC@5={:.3f} AUC@10={:.3f} AUC@20={:.3f}'.format(
            epoch, mean_loss, aucs5, aucs10, aucs20))
        print('[Epoch {}] mAP5={:.3f} mAP10={:.3f} mAP20={:.3f}'.format(
            epoch, va_res[0] * 100, va_res[1] * 100, va_res[3] * 100))
        print('[Epoch {}] precision={:.3f} recall={:.3f} f_score={:.3f}\n'.format(
            epoch, precisions * 100, recalls * 100, f_scores * 100))

        if tb_writer is not None:
            tb_writer.add_scalar('train/loss', mean_loss, epoch)
            tb_writer.add_scalar('valid/AUC@5', aucs5, epoch)
            tb_writer.add_scalar('valid/AUC@10', aucs10, epoch)
            tb_writer.add_scalar('valid/AUC@20', aucs20, epoch)
            tb_writer.add_scalar('valid/mAP5', va_res[0] * 100, epoch)
            tb_writer.add_scalar('valid/mAP10', va_res[1] * 100, epoch)
            tb_writer.add_scalar('valid/mAP20', va_res[3] * 100, epoch)
            tb_writer.add_scalar('valid/precision', precisions * 100, epoch)
            tb_writer.add_scalar('valid/recall', recalls * 100, epoch)
            tb_writer.add_scalar('valid/f_score', f_scores * 100, epoch)
            tb_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

        log_file.write('[Epoch {}] train_loss={:.4f} AUC@5={:.3f} AUC@10={:.3f} AUC@20={:.3f}\n'.format(
            epoch, mean_loss, aucs5, aucs10, aucs20))
        log_file.write('[Epoch {}] mAP5={:.3f} mAP10={:.3f} mAP20={:.3f} precision={:.3f} recall={:.3f} f_score={:.3f}\n'.format(
            epoch, va_res[0] * 100, va_res[1] * 100, va_res[3] * 100, precisions * 100, recalls * 100, f_scores * 100))
        log_file.flush()

        if aucs5 > best_auc:
            print('Saving best model with auc5 = {:.4f}\n'.format(aucs5))
            best_auc = aucs5
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_auc': best_auc,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(conf.best_model_path, 'model_best.pth'))

        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_auc': best_auc,
            'optimizer': optimizer.state_dict(),
        }, os.path.join(conf.checkpoint_path, 'checkpoint_epoch_{}.pth'.format(epoch)))

    log_file.close()
    if tb_writer is not None:
        tb_writer.close()


if __name__ == '__main__':
    main()
