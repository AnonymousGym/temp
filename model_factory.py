################################################################################
# CSE 151B: Programming Assignment 3
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin, Yash, James
# Fall 2022
################################################################################
import torch
from torchvision.models import resnet50
import torch.nn as nn
import torch.nn.functional as F
from pycocotools.coco import COCO


class CustomCNN(nn.Module):
    '''
    A Custom CNN (Task 1) implemented using PyTorch modules based on the architecture in the PA writeup. 
    This will serve as the encoder for our Image Captioning problem.
    '''
    def __init__(self, outputs):
        '''
        Define the layers (convolutional, batchnorm, maxpool, fully connected, etc.)
        with the correct arguments
        
        Parameters:
            outputs => the number of output classes that the final fully connected layer
                       should map its input to
        '''
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=640*480*3, out_channels=64, kernel_size=11, stride=4) # TODO: check if in_channels is correct
        # self.bn = nn.BatchNorm2d(out_channels=)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels=128, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels = 256, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels = 256, out_channels=128, kernel_size=3, padding=1)
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(out_features=1024)
        self.fc2 = nn.Linear(out_features=1024)
        num_classes = 300
        self.fc3 = nn.Linear(out_features=num_classes)

    def forward(self, x):
        '''
        Pass the input through each layer defined in the __init__() function
        in order.

        Parameters:
            x => Input to the CNN
        '''
        model = self.conv1(x)
        model = nn.BatchNorm2d(out_channels=64)
        model = self.relu(model)
        model = self.maxpool1(model)
        model = self.conv2(model)
        model = nn.BatchNorm2d(out_channels = 128)
        model = self.relu(model)
        model = self.maxpool2(model)
        model = self.conv3(model)
        model = nn.BatchNorm2d(out_channels = 256)
        model = self.relu(model)
        model = self.conv4(model)
        model = nn.BatchNorm2d(out_channels = 256)
        model = self.relu(model)
        model = self.conv5(model)
        model = nn.BatchNorm2d(out_channels = 128)
        model = self.relu(model)
        model = self.maxpool3(model)
        model = self.adaptive_avgpool(model)
        model = self.fc1(model)
        model = self.relu(model)
        model = self.fc2(model)
        model = self.relu(model)
        model = self.fc3(model)
        return model


class CNN_LSTM(nn.Module):
    '''
    An encoder decoder architecture.
    Contains a reference to the CNN encoder based on model_type config value.
    Contains an LSTM implemented using PyTorch modules. This will serve as the decoder for our Image Captioning problem.
    '''
    def __init__(self, config_data, vocab):
        '''
        Initialize the embedding layer, LSTM, and anything else you might need.
        '''
        super(CNN_LSTM, self).__init__()
        self.vocab = vocab
        self.hidden_size = config_data['model']['hidden_size']
        self.embedding_size = config_data['model']['embedding_size']
        self.model_type = config_data['model']['model_type']
        self.max_length = config_data['generation']['max_length']
        self.deterministic = config_data['generation']['deterministic']
        self.temp = config_data['generation']['temperature']

        # TODO
        # raise NotImplementedError()

        self.embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=self.embedding_size)

        self.cnn = CustomCNN(self.embedding_size)
        self.resnet50 = resnet50(pretrained=True)
        self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, len(vocab))



    def forward(self, images, captions, teacher_forcing=False, CNN=True):
        '''
        Forward function for this model.
        If teacher forcing is true:
            - Pass encoded images to the LSTM at first time step.
            - Pass each encoded caption one word at a time to the LSTM at every time step after the first one.
        Else:
            - Pass encoded images to the LSTM at first time step.
            - Pass output from previous time step through the LSTM at subsequent time steps
            - Generate predicted caption from the output based on whether we are generating them deterministically or not.
        '''
        # TODO
        # raise NotImplementedError()
        # the image embedding and all the captions except the last one will be the input 
        # and all the captions will be the targets
        caption_embeddings = self.embedding(captions[:, :-1])

        if CNN:
            model1_output = self.cnn.forward(images)
        else:
            # remove the last layer of resnet50 and add a linear layer
            resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
            model1_output = resnet50.forward(images)
            model1_output = model1_output.view(model1_output.size(dim=0), -1)
            model1_output = self.fc(model1_output)
        
        if teacher_forcing:
            input_features = torch.cat((model1_output.unsqueeze(1), caption_embeddings), dim=1)
            lstm_output, (_, _) = self.lstm(input_features)
            linear_output = self.fc(lstm_output)
        else:
            input_features = model1_output.unsqueeze(1)
            # lstm_input shape = (batch_size, seq_len, input_size)
            # lstm_output shape: (batch_size, sequence_length, hidden_size)
            lstm_output, (_, _) = self.lstm(input_features)
            linear_output = self.fc(self.hidden_size, len(self.vocab))
            for i in range(self.max_length):
                lstm_output, (_, _) = self.lstm(linear_output)
                linear_output = self.fc(lstm_output, len(self.vocab))

        return linear_output





def get_model(config_data, vocab, teacher_forcing):
    '''
    Return the LSTM model
    '''
    return CNN_LSTM(config_data, vocab, teacher_forcing)
