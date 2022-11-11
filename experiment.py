################################################################################
# CSE 151B: Programming Assignment 3
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin, Yash, James
# Fall 2022
################################################################################

import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
import math
from tqdm import tqdm
from copy import deepcopy
from nltk.tokenize import word_tokenize
import caption_utils

ROOT_STATS_DIR = './experiment_data'
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model

# coco, coco_test, vocab, train_loader, val_loader, test_loader = get_datasets(
#             config_data)
# print("coco shape", coco.shape)
# print("coco_test shape", coco_test.shape)
# print("vocab shape", vocab.shape)

# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco, self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)

        # Setup Experiment
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__early_stop = config_data['experiment']['early_stop']
        self.__patience = config_data['experiment']['patience']
        self.__batch_size = config_data['dataset']['batch_size']

        # Init Model
        self.__model = get_model(config_data, self.__vocab)
        self.__best_model = deepcopy(self.__model.state_dict())

        # criterion
        # TODO
        self.__criterion = torch.nn.CrossEntropyLoss() 

        # optimizer
        # TODO
        self.__optimzer = torch.nn.optim.Adam(self.__model.parameters(), lr=0.001)


        # LR Scheduler
        # TODO
        self.__lr_scheduler = torch.nn.optim.lr_schedular.reduceLROnPlateau(self.__optimizer, mode='min')

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

        # raise NotImplementedError()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])
        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        patience_count = 0
        min_loss = 100
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            print(f'Epoch {epoch + 1}')
            print('--------')
            start_time = datetime.now()
            self.__current_epoch = epoch
            print('Training...')
            print('-----------')
            train_loss = self.__train()
            print('Validating...')
            print('-------------')
            val_loss = self.__val()

            # save best model
            if val_loss < min_loss:
                min_loss = val_loss
                self.__best_model = deepcopy(self.__model.state_dict())

            # early stop if model starts overfitting
            if self.__early_stop:
                if epoch > 0 and val_loss > self.__val_losses[epoch - 1]:
                    patience_count += 1
                if patience_count >= self.__patience:
                    print('\nEarly stopping!')
                    break

            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()
            if self.__lr_scheduler is not None:
                self.__lr_scheduler.step()
        self.__model.load_state_dict(self.__best_model)

    def __compute_loss(self, images, captions):
        """
        Computes the loss after a forward pass through the model
        """
        # TODO
        # raise NotImplementedError()

        # use the crossEntropyLoss that combines softmax + negative log likelihood loss
        loss = self.__criterion(images.view(-1, len(self.__vocab)), captions.view(-1))

        return loss
        

    def __train(self, generation_mode):
        """
        Trains the model for one epoch using teacher forcing and minibatch stochastic gradient descent
        """
        # TODO
        # raise NotImplementedError()

        # use the train loader to get images and captions
        # use the model to get the predicted captions
        for step, (images, captions) in enumerate(self.__train_loader):
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()

            # forward pass
            outputs = self.__model(images, captions)
            if generation_mode == 'deterministic':
                # Take the maximum output at each step
                outputs = torch.argmax(outputs, dim=2)
            elif generation_mode == 'stochastic':
                # Sample from the distribution at each step
                # outputs = torch.multinomial(outputs, 1) TODO: check this, seems incorrect to me
                pass
            train_loss = self.__compute_loss(outputs, captions)

            # backward pass
            self.__optimizer.zero_grad()
            train_loss.backward()
            self.__optimizer.step()
            self.__lr_scheduler.step()

            if (step + 1) % 50 == 0:
                print(f'Step [{step + 1}/{len(self.__train_loader)}], Train loss: {train_loss.item():.4f}')

            val_loss = self.__val(generation_mode=generation_mode)
            self.__record_stats(train_loss, val_loss)

            


    def __generate_captions(self, img_id, outputs, testing):
        """
        Generate captions without teacher forcing
        Params:
            img_id: Image Id for which caption is being generated
            outputs: output from the forward pass for this img_id
            testing: whether the image_id comes from the validation or test set
        Returns:
            tuple (list of original captions, predicted caption)
        """
        # TODO
        # raise NotImplementedError()

        if testing:
            coco = self.__coco_test
        else:
            coco = self.__coco

        img_original = coco.imgToAnns[img_id] # return a dictionary of all different captions of the given img_id
        print("img_original.keys(): ", img_original.keys())
        img_captions = [ann['caption'] for ann in img_original] # TODO: check if this correctly convert it to list

        predicted_caption = self.__vocab.idx2word(outputs)

        return img_captions, predicted_caption



    def __str_captions(self, img_id, original_captions, predicted_caption):
        """
            !OPTIONAL UTILITY FUNCTION!
            Create a string for logging ground truth and predicted captions for given img_id
        """
        result_str = "Captions: Img ID: {},\nActual: {},\nPredicted: {}\n".format(
            img_id, original_captions, predicted_caption)
        return result_str

    def __val(self, generation_mode):
        """
        Validate the model for one epoch using teacher forcing
        """
        self.__model.eval()
        with torch.no_grad():
            for step, (images, captions) in enumerate(self.__val_loader):
                if torch.cuda.is_available():
                    images = images.cuda()
                    captions = captions.cuda()

                # forward pass
                outputs = self.__model(images, captions, teacher_forcing = True)

                if generation_mode == 'deterministic':
                    # Take the maximum output at each step
                    outputs = torch.argmax(outputs, dim=2)
                elif generation_mode == 'stochastic':
                    # Sample from the distribution at each step
                    # outputs = torch.multinomial(outputs, 1) TODO: check this, seems incorrect to me
                    pass

                val_loss = self.__compute_loss(outputs, captions)

                if (step + 1) % 50 == 0:
                    print(f'Step [{step + 1}/{len(self.__val_loader)}], Val loss: {val_loss.item():.4f}')

        return val_loss.item() # convert tensor to number


    def test(self, generation_mode):
        """
        Test the best model on test data. Generate captions and calculate bleu scores
        """
        self.__model.eval()
        with torch.no_grad():
            for step, (images, captions) in enumerate(self.__test_loader):
                if torch.cuda.is_available():
                    images = images.cuda()
                    captions = captions.cuda()

                # forward pass
                outputs = self.__model(images, captions, teacher_forcing = False)
                
                if generation_mode == 'deterministic':
                    # Take the maximum output at each step
                    outputs = torch.argmax(outputs, dim=2)
                elif generation_mode == 'stochastic':
                    # Sample from the distribution at each step
                    # outputs = torch.multinomial(outputs, 1) TODO: check this, seems incorrect to me
                    pass
    

                test_loss = self.__compute_loss(outputs, captions)

                if (step + 1) % 50 == 0:
                    print(f'Step [{step + 1}/{len(self.__test_loader)}], Loss: {test_loss.item():.4f}')

                bleu1 = caption_utils.bleu1(captions, outputs)
                bleu2 = caption_utils.bleu2(captions, outputs)

        


    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
