from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data

from MCVN import MCVN


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=15, verbose=False, delta=0, path='./CheckPoint/checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 15
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class Trainer:
    """This class defines the MCVN trainer.

    Attributes:
        model: The MCVN model instance.
        train_data: Train tensor dataset.
        valid_data: Valid dataset(x_imgs, x_original, y_valid)
        criterion: Loss function.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        if_cuda: if use cuda.

    """

    def __init__(self, model: MCVN, train_data: Data.TensorDataset,
                 valid_data: List[torch.Tensor],
                 criterion, optimizer, scheduler=None,
                 if_cuda: bool = False) -> None:
        if torch.cuda.is_available() and if_cuda:
            self.device = torch.device('cuda:0')
            self.if_cuda = True
        else:
            self.device = torch.device('cpu')
            self.if_cuda = False
        # init model
        self.model = model
        self.model.to(self.device)
        self.model.double()
        self.model.apply(self._weight_init)
        # set data
        self.train_data = train_data
        self.valid_data = [data.to(self.device) for data in valid_data]
        # set loss function and optimizer
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        # Conv2D Layer
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
        # BN Layer
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def set_model(self, model: MCVN):
        """Set new model.

        Arguments:
            model: The MCVN model instance.

        """
        self.model = model
        self.model.to(self.device)
        self.model.double()
        self.model.apply(self._weight_init)

    def set_data(self, train_data: Data.TensorDataset,
                 valid_data: List[torch.Tensor]):
        """Set new train valid test data.

        Arguments:
            train_data: Train tensor dataset.
            valid_data: Valid dataset(x_imgs, x_original, y_valid)
            test_data: Test dataset(x_imgs, x_original, y_test)

        """
        self.train_data = train_data
        self.valid_data = [data.to(self.device) for data in valid_data]

    def set_optimizer(self, optimizer, scheduler):
        """Set new optimizer.

        Arguments:
            optimizer: The optimizer.
            scheduler: The learning rate scheduler.

        """
        self.optimizer = optimizer
        self.scheduler = scheduler

    def set_criterion(self, criterion):
        """Set new loss function.

        Arguments:
            criterion: The loss function.

        """
        self.criterion = criterion

    def train(self, batch_size: int, earlystop_patience: int, n_epochs: int,
              model_saving_path='./CheckPoint/checkpoint.pt', trace_func=print):
        """Train the model.

        Arguments:
            batch_size: Batch size for training.
            earlystop_patience: earlystop's patience(epoch).
            n_epochs: Max epochs training.
            model_saving_path: Model saving path.
            trace_func: Logging function.


        """
        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []

        # Generate train, valid, test data.
        data_iter = Data.DataLoader(self.train_data, batch_size=batch_size,
                                    shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=0,
                                    collate_fn=None, pin_memory=False,
                                    drop_last=False, timeout=0,
                                    worker_init_fn=None,
                                    multiprocessing_context=None)
        x_val_pics, x_val_other, y_val = self.valid_data

        # initialize the early_stopping object
        early_stopping = EarlyStopping(
            patience=earlystop_patience, verbose=True, path=model_saving_path, trace_func=trace_func)
        # Training
        for epoch in range(1, n_epochs + 1):
            ###################
            # train the model #
            ###################
            self.model.train()  # prep model for training
            for x_train_pics_batch, x_train_other_batch, y_train_batch in data_iter:
                x_train_pics_batch = x_train_pics_batch.to(self.device)
                x_train_other_batch = x_train_other_batch.to(self.device)
                y_train_batch = y_train_batch.to(self.device)
                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(x_train_pics_batch, x_train_other_batch)
                # calculate the loss
                loss = self.criterion(torch.squeeze(
                    output), torch.squeeze(y_train_batch))
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                # record training loss
                train_losses.append(loss.data.item())
            ######################
            # validate the model #
            ######################
            train_loss = np.average(train_losses)
            # clear lists to track next epoch
            train_losses = []
            avg_train_losses.append(train_loss)
            self.model.eval()  # prep model for evaluation
            # Valid log
            pred_val = self.model(x_val_pics, x_val_other)
            valid_loss = self.criterion(torch.squeeze(
                pred_val.cpu()), torch.squeeze(y_val.cpu())).data.item()
            epoch_len = len(str(n_epochs))
            valid_losses.append(valid_loss)

            # Terminal logging
            trace_func(f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                       f'train_loss: {train_loss:.5f} ' +
                       f'valid_loss: {valid_loss:.5f}')

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, self.model)

            if early_stopping.early_stop:
                trace_func("Early stopping")
                break

        # load the last checkpoint with the best model
        self.model.load_state_dict(torch.load(model_saving_path))
        return train_losses, valid_losses
