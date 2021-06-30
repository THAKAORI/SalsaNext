import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tasks.semantic.modules.utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, model, optimizer, scheduler, device, FLAGS):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.FLAGS = FLAGS
        print(self.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.FLAGS.batch_size) for i in range(self.FLAGS.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.FLAGS.temperature
        return logits, labels

    def train(self, train_loader, val_loader):

        scaler = GradScaler(enabled=self.FLAGS.fp16_precision)

        # save config file
        # save_config_file(self.writer.log_dir, self.FLAGS)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.FLAGS.epochs} epochs.")
        logging.info(f"Training with gpu: {self.FLAGS.disable_cuda}.")
        max_val_acc = 0

        for epoch_counter in range(self.FLAGS.epochs):
            torch.cuda.empty_cache()
            self.model.train()
            for images in train_loader:
                images = torch.cat(images, dim=0)

                images = images.to(self.device)

                with autocast(enabled=self.FLAGS.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.FLAGS.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    print("train:epoch:", epoch_counter, " N_iter:", n_iter," loss: ", loss.item(), " acc/top1:", top1[0].item(), " acc/top5:", top5[0].item(), " learning_rate:", self.scheduler.get_lr()[0])
                    # self.writer.add_scalar('loss', loss, global_step=n_iter)
                    # self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    # self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    # self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            torch.cuda.empty_cache()
            self.model.eval()
            lossvallist = []
            top1vallist = []
            top5vallist = []
            with torch.no_grad():
                for images in val_loader:
                    images = torch.cat(images, dim=0)

                    images = images.to(self.device)

                    with autocast(enabled=self.FLAGS.fp16_precision):
                        features = self.model(images)
                        logits, labels = self.info_nce_loss(features)
                        loss = self.criterion(logits, labels)
                        lossvallist.append(loss.item())

                    top1val, top5val = accuracy(logits, labels, topk=(1, 5))
                    top1vallist.append(top1val[0].item())
                    top5vallist.append(top5val[0].item())
                    #print("val:epoch:", epoch_counter, " N_iter:", n_iter," loss: ", loss.item(), " acc/top1:", top1val[0].item(), " acc/top5:", top5val[0].item())

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")
            lossvalavg = sum(lossvallist) / len(lossvallist)
            top1valavg = sum(top1vallist) / len(top1vallist)
            top5valavg = sum(top5vallist) / len(top5vallist)
            print("val:loss: ", loss.item(), " acc/top1:", top1val[0].item(), " acc/top5:", top5val[0].item())
            if(top1valavg > max_val_acc):
                checkpoint_name = 'checkpoint_best_val.pth'
                save_checkpoint({
                    'epoch': self.FLAGS.epochs,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=True, filename=os.path.join(self.writer.log_dir, checkpoint_name))

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.FLAGS.epochs)
        save_checkpoint({
            'epoch': self.FLAGS.epochs,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")