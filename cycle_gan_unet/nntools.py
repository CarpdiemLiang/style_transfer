"""
Neural Network tools developed for UCSD ECE285 MLIP.

Copyright 2019. Charles Deledalle, Sneha Gupta, Anurag Paul, Inderjot Saggu.
"""

import os
import time
import torch
from torch import nn
import torch.utils.data as td
from abc import ABC, abstractmethod
import pickle, argparse, util, itertools
import itertools
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable


class NeuralNetwork(nn.Module, ABC):
    """An abstract class representing a neural network.

    All other neural network should subclass it. All subclasses should override
    ``forward``, that makes a prediction for its input argument, and
    ``criterion``, that evaluates the fit between a prediction and a desired
    output. This class inherits from ``nn.Module`` and overloads the method
    ``named_parameters`` such that only parameters that require gradient
    computation are returned. Unlike ``nn.Module``, it also provides a property
    ``device`` that returns the current device in which the network is stored
    (assuming all network parameters are stored on the same device).
    """

    def __init__(self):
        super(NeuralNetwork, self).__init__()

    @property
    def device(self):
        # This is important that this is a property and not an attribute as the
        # device may change anytime if the user do ``net.to(newdevice)``.
        return next(self.parameters()).device

    def named_parameters(self, recurse=True):
        nps = nn.Module.named_parameters(self)
        for name, param in nps:
            if not param.requires_grad:
                continue
            yield name, param

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def criterion(self, y, d):
        pass


class StatsManager(object):
    """
    A class meant to track the loss during a neural network learning experiment.

    Though not abstract, this class is meant to be overloaded to compute and
    track statistics relevant for a given task. For instance, you may want to
    overload its methods to keep track of the accuracy, top-5 accuracy,
    intersection over union, PSNR, etc, when training a classifier, an object
    detector, a denoiser, etc.
    """

    def __init__(self):
        self.init()

    def __repr__(self):
        """Pretty printer showing the class name of the stats manager. This is
        what is displayed when doing ``print(stats_manager)``.
        """
        return self.__class__.__name__

    def init(self):
        """Initialize/Reset all the statistics"""
        self.running_loss = 0
        self.number_update = 0

    def accumulate(self, loss, DA, DB, x=None, y=None, d=None):
        """Accumulate statistics

        Though the arguments x, y, d are not used in this implementation, they
        are meant to be used by any subclasses. For instance they can be used
        to compute and track top-5 accuracy when training a classifier.

        Arguments:
            loss (float): the loss obtained during the last update.
            x (Tensor): the input of the network during the last update.
            y (Tensor): the prediction of by the network during the last update.
            d (Tensor): the desired output for the last update.
        """
        self.running_loss += loss
        self.number_update += 1

    def summarize(self):
        """Compute statistics based on accumulated ones"""
        return self.running_loss / self.number_update


class Experiment(object):
    """
    A class meant to run a neural network learning experiment.

    After being instantiated, the experiment can be run using the method
    ``run``. At each epoch, a checkpoint file will be created in the directory
    ``output_dir``. Two files will be present: ``checkpoint.pth.tar`` a binary
    file containing the state of the experiment, and ``config.txt`` an ASCII
    file describing the setting of the experiment. If ``output_dir`` does not
    exist, it will be created. Otherwise, the last checkpoint will be loaded,
    except if the setting does not match (in that case an exception will be
    raised). The loaded experiment will be continued from where it stopped when
    calling the method ``run``. The experiment can be evaluated using the method
    ``evaluate``.

    Attributes/Properties:
        epoch (integer): the number of performed epochs.
        history (list): a list of statistics for each epoch.
            If ``perform_validation_during_training``=False, each element of the
            list is a statistic returned by the stats manager on training data.
            If ``perform_validation_during_training``=True, each element of the
            list is a pair. The first element of the pair is a statistic
            returned by the stats manager evaluated on the training set. The
            second element of the pair is a statistic returned by the stats
            manager evaluated on the validation set.

    Arguments:
        net (NeuralNetork): a neural network.
        train_set (Dataset): a training data set.
        val_set (Dataset): a validation data set.
        stats_manager (StatsManager): a stats manager.
        output_dir (string, optional): path where to load/save checkpoints. If
            None, ``output_dir`` is set to "experiment_TIMESTAMP" where
            TIMESTAMP is the current time stamp as returned by ``time.time()``.
            (default: None)
        batch_size (integer, optional): the size of the mini batches.
            (default: 16)
        perform_validation_during_training (boolean, optional): if False,
            statistics at each epoch are computed on the training set only.
            If True, statistics at each epoch are computed on both the training
            set and the validation set. (default: False)
    """

    def __init__(self, gnetA, gnetB, dnetA, dnetB, train_set_A, train_set_B, val_set_A, val_set_B, G_optimizer, D_A_optimizer, D_B_optimizer, 
                 stats_manager, output_dir=None, batch_size=4, perform_validation_during_training=False):

        # Define data loaders
        train_loader_A = td.DataLoader(train_set_A, batch_size=batch_size, shuffle=True,
                                     drop_last=True, pin_memory=True)
        train_loader_B = td.DataLoader(train_set_B, batch_size=batch_size, shuffle=True,
                                     drop_last=True, pin_memory=True)
        val_loader_A = td.DataLoader(val_set_A, batch_size=batch_size, shuffle=False,
                                   drop_last=True, pin_memory=True)
        val_loader_B = td.DataLoader(val_set_B, batch_size=batch_size, shuffle=False,
                                   drop_last=True, pin_memory=True)

        # Initialize history
        history = []
        
        # fake images store
        fakeA_store = util.ImagePool(50)
        fakeB_store = util.ImagePool(50)

        # Define checkpoint paths
        if output_dir is None:
            output_dir = 'experiment_{}'.format(time.time())
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, "checkpoint.pth.tar")
        config_path = os.path.join(output_dir, "config.txt")

        # Transfer all local arguments/variables into attributes
        locs = {k: v for k, v in locals().items() if k is not 'self'}
        self.__dict__.update(locs)

        # Load checkpoint and check compatibility
        if os.path.isfile(config_path):
            with open(config_path, 'r') as f:
                if f.read()[:-1] != repr(self):
                    raise ValueError(
                        "Cannot create this experiment: "
                        "I found a checkpoint conflicting with the current setting.")
            self.load()
        else:
            self.save()

    @property
    def epoch(self):
        """Returns the number of epochs already performed."""
        return len(self.history)

    def setting(self):
        """Returns the setting of the experiment."""
        return {'GNet_A': self.gnetA,
                'GNet_B' : self.gnetB,
                'DNet_A' : self.dnetA,
                'DNet_B' : self.dnetB,
                'TrainSetA': self.train_set_A,
                'TrainSetB': self.train_set_B,
                'ValSetA': self.val_set_A,
                'ValSetB': self.val_set_B,
                'GOptimizer': self.G_optimizer,
                'DAOptimizer': self.D_A_optimizer,
                'DBOptimizer': self.D_B_optimizer,
                'StatsManager': self.stats_manager,
                'BatchSize': self.batch_size,
                'PerformValidationDuringTraining': self.perform_validation_during_training}

    def __repr__(self):
        """Pretty printer showing the setting of the experiment. This is what
        is displayed when doing ``print(experiment)``. This is also what is
        saved in the ``config.txt`` file.
        """
        string = ''
        for key, val in self.setting().items():
            string += '{}({})\n'.format(key, val)
        return string

    def state_dict(self):
        """Returns the current state of the experiment."""
        return {'GNet_A': self.gnetA.state_dict(),
                'GNet_B': self.gnetB.state_dict(),
                'DNet_A': self.dnetA.state_dict(),
                'DNet_B': self.dnetB.state_dict(),
                'GOptimizer': self.G_optimizer.state_dict(),
                'DAOptimizer': self.D_A_optimizer.state_dict(),
                'DBOptimizer': self.D_B_optimizer.state_dict(),
                'FakeA': self.fakeA_store,
                'FakeB': self.fakeB_store,
                'History': self.history}

    def load_state_dict(self, checkpoint):
        """Loads the experiment from the input checkpoint."""
        self.gnetA.load_state_dict(checkpoint['GNet_A'])
        self.gnetB.load_state_dict(checkpoint['GNet_B'])
        self.dnetA.load_state_dict(checkpoint['DNet_A'])
        self.dnetB.load_state_dict(checkpoint['DNet_B'])
        self.G_optimizer.load_state_dict(checkpoint['GOptimizer'])
        self.D_A_optimizer.load_state_dict(checkpoint['DAOptimizer'])
        self.D_B_optimizer.load_state_dict(checkpoint['DBOptimizer'])
        self.fakeA_store = checkpoint['FakeA']
        self.fakeB_store = checkpoint['FakeB']
        self.history = checkpoint['History']

        # The following loops are used to fix a bug that was
        # discussed here: https://github.com/pytorch/pytorch/issues/2830
        # (it is supposed to be fixed in recent PyTorch version)
        for state in self.G_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.gnetA.device)
                    
        for state in self.D_A_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.gnetA.device)
        
        for state in self.D_B_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.gnetA.device)

    def save(self):
        """Saves the experiment on disk, i.e, create/update the last checkpoint."""
        torch.save(self.state_dict(), self.checkpoint_path)
        with open(self.config_path, 'w') as f:
            print(self, file=f)

    def load(self):
        """Loads the experiment from the last checkpoint saved on disk."""
        checkpoint = torch.load(self.checkpoint_path,
                                map_location=self.gnetA.device)
        self.load_state_dict(checkpoint)
        del checkpoint

    def run(self, num_epochs, decay_epoch, lrG, lrD, lambdaA=10, lambdaB=10, plot=None):
        """Runs the experiment, i.e., trains the network using backpropagation
        based on the optimizer and the training set. Also performs statistics at
        each epoch using the stats manager.

        Arguments:
            num_epoch (integer): the number of epoch to perform.
            plot (func, optional): if not None, should be a function taking a
                single argument being an experiment (meant to be ``self``).
                Similar to a visitor pattern, this function is meant to inspect
                the current state of the experiment and display/plot/save
                statistics. For example, if the experiment is run from a
                Jupyter notebook, ``plot`` can be used to display the evolution
                of the loss with ``matplotlib``. If the experiment is run on a
                server without display, ``plot`` can be used to show statistics
                on ``stdout`` or save statistics in a log file. (default: None)
        """
        self.gnetA.train()
        self.gnetB.train()
        self.dnetA.train()
        self.dnetB.train()
        self.stats_manager.init()
        start_epoch = self.epoch
        
        # loss
        BCE_loss = nn.BCELoss().cuda()
        MSE_loss = nn.MSELoss().cuda()
        L1_loss = nn.L1Loss().cuda()
                
        print("Start/Continue training from epoch {}".format(start_epoch))
        if plot is not None:
            plot(self)
        for epoch in range(start_epoch, num_epochs):
            s = time.time()
            self.stats_manager.init()
            if (epoch+1) > decay_epoch:
                self.D_A_optimizer.param_groups[0]['lr'] -= lrD / (num_epochs - decay_epoch)
                self.D_B_optimizer.param_groups[0]['lr'] -= lrD / (num_epochs - decay_epoch)
                self.G_optimizer.param_groups[0]['lr'] -= lrG / (num_epochs - decay_epoch)
                
            for (realA, _), (realB, _) in zip(self.train_loader_A, self.train_loader_B):
                realA, realB = realA.to(self.gnetA.device), realB.to(self.gnetB.device)
                # train generator
                self.G_optimizer.zero_grad()
                fakeB = self.gnetA(realA)
                D_A_result = self.dnetA(fakeB)
                G_A_loss = MSE_loss(D_A_result, Variable(torch.ones(D_A_result.size()).cuda()))
                recA = self.gnetB(fakeB)
                A_cycle_loss = L1_loss(recA, realA) * lambdaA
                fakeA = self.gnetB(realB)
                D_B_result = self.dnetB(fakeA)
                G_B_loss = MSE_loss(D_B_result, Variable(torch.ones(D_B_result.size()).cuda()))
                recB = self.gnetA(fakeA)
                B_cycle_loss = L1_loss(recB, realB) * lambdaB
                G_loss = G_A_loss + G_B_loss + A_cycle_loss + B_cycle_loss
                G_loss.backward()
                self.G_optimizer.step()
                
                # train discriminator D_A
                self.D_A_optimizer.zero_grad()
                D_A_real = self.dnetA(realB)
                D_A_real_loss = MSE_loss(D_A_real, Variable(torch.ones(D_A_real.size()).cuda()))
                fakeB = self.fakeB_store.query(fakeB)
                D_A_fake = self.dnetA(fakeB)
                D_A_fake_loss = MSE_loss(D_A_fake, Variable(torch.zeros(D_A_fake.size()).cuda()))
                D_A_loss = (D_A_real_loss + D_A_fake_loss) * 0.5
                D_A_loss.backward()
                self.D_A_optimizer.step()

                # train discriminator D_B
                self.D_B_optimizer.zero_grad()
                D_B_real = self.dnetB(realA)
                D_B_real_loss = MSE_loss(D_B_real, Variable(torch.ones(D_B_real.size()).cuda()))
                fakeA = self.fakeA_store.query(fakeA)
                D_B_fake = self.dnetB(fakeA)
                D_B_fake_loss = MSE_loss(D_B_fake, Variable(torch.zeros(D_B_fake.size()).cuda()))
                D_B_loss = (D_B_real_loss + D_B_fake_loss) * 0.5
                D_B_loss.backward()
                self.D_B_optimizer.step()
                
                # status manager accumulate
                with torch.no_grad():
                    self.stats_manager.accumulate(G_loss.item(), D_A_loss.item(), D_B_loss.item())
            if not self.perform_validation_during_training:
                self.history.append(self.stats_manager.summarize())
            else:
                self.history.append(
                    (self.stats_manager.summarize(), self.evaluate()))
            print("Epoch {} (Time: {:.2f}s)".format(
                self.epoch, time.time() - s))
            self.save()
            if plot is not None:
                plot(self)
        print("Finish training for {} epochs".format(num_epochs))

    def evaluate(self, lambdaA=10, lambdaB=10):
        """Evaluates the experiment, i.e., forward propagates the validation set
        through the network and returns the statistics computed by the stats
        manager.
        """
        self.stats_manager.init()
        self.gnetA.eval()
        self.gnetB.eval()
        self.dnetA.eval()
        self.dnetB.eval()
        BCE_loss = nn.BCELoss().cuda()
        MSE_loss = nn.MSELoss().cuda()
        L1_loss = nn.L1Loss().cuda()
        with torch.no_grad():
            for (realA, _), (realB, _) in zip(self.val_loader_A, self.val_loader_B):
                realA, realB = realA.to(self.gnetA.device), realB.to(self.gnetA.device)
                self.G_optimizer.zero_grad()
                fakeB = self.gnetA(realA)
                D_A_result = self.dnetA(fakeB)
                G_A_loss = MSE_loss(D_A_result, Variable(torch.ones(D_A_result.size()).cuda()))
                recA = self.gnetB(fakeB)
                A_cycle_loss = L1_loss(recA, realA) * lambdaA
                fakeA = self.gnetB(realB)
                D_B_result = self.dnetB(fakeA)
                G_B_loss = MSE_loss(D_B_result, Variable(torch.ones(D_B_result.size()).cuda()))
                recB = self.gnetA(fakeA)
                B_cycle_loss = L1_loss(recB, realB) * lambdaB
                G_loss = G_A_loss + G_B_loss + A_cycle_loss + B_cycle_loss
                
                # train discriminator D_A
                self.D_A_optimizer.zero_grad()
                D_A_real = self.dnetA(realB)
                D_A_real_loss = MSE_loss(D_A_real, Variable(torch.ones(D_A_real.size()).cuda()))
                fakeB = self.fakeB_store.query(fakeB)
                D_A_fake = self.dnetA(fakeB)
                D_A_fake_loss = MSE_loss(D_A_fake, Variable(torch.zeros(D_A_fake.size()).cuda()))
                D_A_loss = (D_A_real_loss + D_A_fake_loss) * 0.5

                # train discriminator D_B
                self.D_B_optimizer.zero_grad()
                D_B_real = self.dnetB(realA)
                D_B_real_loss = MSE_loss(D_B_real, Variable(torch.ones(D_B_real.size()).cuda()))
                fakeA = self.fakeA_store.query(fakeA)
                D_B_fake = self.dnetB(fakeA)
                D_B_fake_loss = MSE_loss(D_B_fake, Variable(torch.zeros(D_B_fake.size()).cuda()))
                D_B_loss = (D_B_real_loss + D_B_fake_loss) * 0.5
                self.stats_manager.accumulate(G_loss.item(), D_A_loss.item(), D_B_loss.item())
        self.gnetA.train()
        self.gnetB.train()
        self.dnetA.train()
        self.dnetB.train()
        return self.stats_manager.summarize()
