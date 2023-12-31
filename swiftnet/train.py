import argparse
import os
from pathlib import Path
import torch
import importlib.util
import datetime
import sys
from shutil import copy
import pickle
from time import perf_counter

from torchvision.transforms import Compose
from evaluation import evaluate_semseg, ASREvaluationObserver, StorePreds
from data.transform import *


def import_module(path):
    spec = importlib.util.spec_from_file_location("module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def store(model, store_path, name):
    with open(store_path.format(name), 'wb') as f:
        torch.save(model.state_dict(), f)


class Logger(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()


class Trainer:
    def __init__(self, conf, args, name):
        self.conf = conf
        using_hparams = hasattr(conf, 'hyperparams')
        print(f'Using hparams: {using_hparams}')
        self.hyperparams = self.conf
        self.args = args
        self.name = name
        self.model = self.conf.model
        self.optimizer = self.conf.optimizer

        self.dataset_train = self.conf.dataset_train
        self.dataset_val = self.conf.dataset_val
        self.loader_train = self.conf.loader_train
        self.loader_val = self.conf.loader_val

        if self.args.poison:
            self.dataset_val_poisoned = self.conf.dataset_val_poisoned
            self.loader_val_poisoned = self.conf.loader_val_poisoned

            if self.args.save_preds:
                self.chosen_names = [x.stem for x in self.dataset_val_poisoned.images[0:5]]

    def __enter__(self):
        self.best_iou = -1
        self.best_iou_epoch = -1
        self.validation_ious = []
        self.validation_pas = []
        self.experiment_start = datetime.datetime.now()

        if self.args.poison:
            self.best_poisoned_iou = -1
            self.best_poisoned_iou_epoch = -1
            self.validation_poisoned_ious = []
            self.validation_poisoned_pas = []

        if self.args.resume:
            self.experiment_dir = Path(self.args.resume)
            print(f'Resuming experiment from {args.resume}')
        else:
            self.experiment_dir = Path(self.args.store_dir) / (
                    self.experiment_start.strftime('%Y_%m_%d_%H_%M_%S_') + self.name)

        self.checkpoint_dir = self.experiment_dir / 'stored'
        self.store_path = str(self.checkpoint_dir / '{}.pt')

        if not self.args.dry and not self.args.resume:
            os.makedirs(str(self.experiment_dir), exist_ok=True)
            os.makedirs(str(self.checkpoint_dir), exist_ok=True)
            copy(self.args.config, str(self.experiment_dir / 'config.py'))

        if self.args.log and not self.args.dry:
            f = (self.experiment_dir / 'log.txt').open(mode='a')
            sys.stdout = Logger(sys.stdout, f)

        self.model.cuda()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.args.dry:
            store(self.model, self.store_path, 'model')
        if not self.args.dry:
            with open(f'{self.experiment_dir}/val_ious.pkl', 'wb') as f:
                pickle.dump(self.validation_ious, f)
            with open(f'{self.experiment_dir}/val_pas.pkl', 'wb') as f:
                pickle.dump(self.validation_pas, f)

            if self.args.poison:
                with open(f'{self.experiment_dir}/val_poisoned_ious.pkl', 'wb') as f:
                    pickle.dump(self.validation_poisoned_ious, f)
                with open(f'{self.experiment_dir}/val_poisoned_pas.pkl', 'wb') as f:
                    pickle.dump(self.validation_poisoned_pas, f)
                    
            dir_iou = Path(self.args.store_dir) / (f'{self.best_iou:.2f}_'.replace('.', '-') + self.name)
            os.rename(self.experiment_dir, dir_iou)

    def train(self):
        num_epochs = self.hyperparams.epochs
        start_epoch = self.hyperparams.start_epoch if hasattr(self.hyperparams, 'start_epoch') else 0
        for epoch in range(start_epoch, num_epochs):
            if hasattr(self.conf, 'epoch'):
                self.conf.epoch.value = epoch
                print(self.conf.epoch)
            self.model.train()
            try:
                self.conf.lr_scheduler.step()
                print(f'Elapsed time: {datetime.datetime.now() - self.experiment_start}')
                for group in self.optimizer.param_groups:
                    print('LR: {:.4e}'.format(group['lr']))
                eval_epoch = ((epoch % self.conf.eval_each == 0) or (epoch == num_epochs - 1))  # and (epoch > 0)
                self.model.criterion.step_counter = 0
                print(f'Epoch: {epoch} / {num_epochs - 1}')
                if eval_epoch and not self.args.dry:
                    print("Experiment dir: %s" % self.experiment_dir)
                batch_iterator = iter(enumerate(self.loader_train))
                start_t = perf_counter()
                for step, batch in batch_iterator:
                    self.optimizer.zero_grad()
                    loss = self.model.loss(batch)
                    loss.backward()
                    self.optimizer.step()
                    if step % 80 == 0 and step > 0:
                        curr_t = perf_counter()
                        print(f'{(step * self.conf.batch_size) / (curr_t - start_t):.2f}fps')
                if not self.args.dry:
                    store(self.model, self.store_path, 'model')
                    store(self.optimizer, self.store_path, 'optimizer')
                if eval_epoch and self.args.eval:
                    print('Evaluating model')
                    iou, per_class_iou, pa = evaluate_semseg(self.model, self.loader_val, self.dataset_val.class_info)
                    self.validation_ious += [iou]
                    self.validation_pas += [pa]
                    if self.args.eval_train:
                        print('Evaluating train')
                        evaluate_semseg(self.model, self.loader_train, self.dataset_train.class_info)
                    if iou > self.best_iou:
                        self.best_iou = iou
                        self.best_iou_epoch = epoch
                        if not self.args.dry:
                            copy(self.store_path.format('model'), self.store_path.format('model_best'))
                    print(f'Best mIoU: {self.best_iou:.2f}% (epoch {self.best_iou_epoch})')

                    if self.args.poison:
                        observers = [ASREvaluationObserver(f'{self.experiment_dir}/val_asrs.txt')]

                        if self.args.save_preds:
                            preds_save_path = f'{self.experiment_dir}/predictions/{epoch}'
                            if not os.path.exists(preds_save_path): 
                                os.makedirs(preds_save_path)
                            
                            to_color = ColorizeLabels(self.conf.color_info)
                            to_image = Compose([Numpy(), to_color])
                            observers += [StorePreds(preds_save_path, to_image, to_color, self.chosen_names, False)]
                        
                        print('Evaluating poisoned')
                        poisoned_iou, poisoned_per_class_iou, poisoned_pa = evaluate_semseg(self.model, self.loader_val_poisoned, self.dataset_val_poisoned.class_info, observers)
                        self.validation_poisoned_ious += [poisoned_iou]
                        self.validation_poisoned_pas += [poisoned_pa]
                        if poisoned_iou > self.best_poisoned_iou:
                            self.best_poisoned_iou = poisoned_iou
                            self.best_poisoned_iou_epoch = epoch
                        print(f'Best poisoned mIoU: {self.best_poisoned_iou:.2f}% (epoch {self.best_poisoned_iou_epoch})')

            except KeyboardInterrupt:
                break


parser = argparse.ArgumentParser(description='Detector train')
parser.add_argument('config', type=str, help='Path to configuration .py file')
parser.add_argument('--poison', dest='poison', action='store_true', help='Turn on dataset poisoning')
parser.add_argument('--save_preds', dest='save_preds', action='store_true', help='Turn on 5 poisoned predictions saving after each epoch')
parser.add_argument('--store_dir', default='saves/', type=str, help='Path to experiments directory')
parser.add_argument('--resume', default=None, type=str, help='Path to existing experiment dir')
parser.add_argument('--no-log', dest='log', action='store_false', help='Turn off logging')
parser.add_argument('--log', dest='log', action='store_true', help='Turn on train evaluation')
parser.add_argument('--no-eval-train', dest='eval_train', action='store_false', help='Turn off train evaluation')
parser.add_argument('--eval-train', dest='eval_train', action='store_true', help='Turn on train evaluation')
parser.add_argument('--no-eval', dest='eval', action='store_false', help='Turn off evaluation')
parser.add_argument('--eval', dest='eval', action='store_true', help='Turn on evaluation')
parser.add_argument('--dry-run', dest='dry', action='store_true', help='Don\'t store')
parser.set_defaults(log=True)
parser.set_defaults(eval_train=False)
parser.set_defaults(eval=True)

if __name__ == '__main__':
    args = parser.parse_args()
    conf_path = Path(args.config)
    conf = import_module(args.config)

    with Trainer(conf, args, conf_path.stem) as trainer:
        trainer.train()
