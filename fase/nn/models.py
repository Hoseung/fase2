# Evaluation-only functions
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import List

# Import 
from .functional_plain import (avg_pooling_torch, 
                            approx_relu, 
                            conv2d_eval_torch, 
                            fully_connected,
                            batchnorm_eval_torch)

class CNN_infer:
    def __init__(self,file=None):
        """
        This class loads a pretrained CNN network trained by CIFAR10_training.ipynb,
        in a SPECIFIC structure. 
        """
        print("\nInitialized CNN")
        print("Weights from: " + str(file))
        self._weights = {}
        self._bn_params = {}
        self._conv_params =None
        self.set_conv_params([0,0],[1,1])

        try:
            self.load_torch_params(file)
        except NotImplementedError:
            print("Training is not supported. You must provide with a trained parameter file")
    
    def set_conv_params(self, pad: List[int], stride: List[int]):
        """
        set a dict of paddings and strides like {'p1':1, 'p2':2, 's1':1, 's2':1}
        parameters
        ----------
        pad: list of paddings from conv1 to convN

        stride: list of strides from conv1 to convN
        """
        assert len(pad) == len(stride), "pad count and stride count don't match"
        keys = []
        for i in range(len(pad)):
            keys.append(f'p{i+1}')
        for i in range(len(stride)):
            keys.append(f's{i+1}')
        
        self._conv_params = dict(zip(keys, pad + stride))


    def eval(self, x):
        caches = self.forward_propagate_torch(x)

        return caches["result"]

    def forward_propagate_torch(self, x):
        """Go through the forward propagation.
        
        parameters
        ----------
        x: example to perfom inference on
        """

        # Name of CNN weights - as defined in load_torch_params()
        w1,w2,w3,w4,w5 = ["cvw1", "cvw2", "fcw1", "fcw2", "fcw3"]
        b1,b2,b3,b4,b5 = ["cvb1", "cvb2", "fcb1", "fcb2", "fcb3"]
        weights = self._weights
        bn_params = self._bn_params
        caches = {}
        cvp = self._conv_params
        
        Z1, caches["Z1"] = conv2d_eval_torch(x,weights[w1], weights[b1],
                                        pad=cvp['p1'], stride=cvp['s1'])
        BN1 = batchnorm_eval_torch(Z1,
                                    bn_params['gamma_1'],
                                    bn_params['beta_1'],
                                    bn_params["running_mu_1"],
                                    bn_params["running_sigma_1"])    
        caches["A1"] = approx_relu(BN1)
        Pool1, caches["Pool1"] = avg_pooling_torch(caches["A1"],2)
        Z2, caches["Z2"] = conv2d_eval_torch(Pool1,weights[w2],weights[b2],
                                        pad=cvp['p2'], stride=cvp['s2'])
        BN2 = batchnorm_eval_torch(Z2,
                                    bn_params['gamma_2'],
                                    bn_params['beta_2'],
                                    bn_params["running_mu_2"],
                                    bn_params["running_sigma_2"])
        caches["A2"] = approx_relu(BN2)
        Pool2, caches["Pool2"] = avg_pooling_torch(caches["A2"],2)
        # Reshape
        pool2_reshape = Pool2.reshape(Pool2.shape[0],Pool2.shape[1] * Pool2.shape[2] * Pool2.shape[3])
        
        Z3, caches["Z3"] = fully_connected(pool2_reshape,weights[w3],weights[b3])
        Z3 = approx_relu(Z3)
        
        Z4, caches["Z4"] = fully_connected(Z3,weights[w4],weights[b4])
        Z4 = approx_relu(Z4)
        
        Z5, caches["Z5"] = fully_connected(Z4,weights[w5],weights[b5])
        caches["result"] = approx_relu(Z5)

        return caches

    def load_torch_params(self, filename):
        """Load weights from a pretrained pytorch model of a specific network structure.
        
        todo 
        -----
        Devise a smart, automatic naming rule.
        """
        parameters = torch.load(filename)
        weights = self._weights
        bn_params = self._bn_params

        weights["cvw1"] = np.array(parameters['conv1.weight'])
        weights["cvw2"] = np.array(parameters['conv2.weight'])
        weights["fcw1"] = np.array(parameters['fc1.weight'].T)
        weights["fcw2"] = np.array(parameters['fc2.weight'].T)
        weights["fcw3"] = np.array(parameters['fc3.weight'].T)

        weights["cvb1"] = np.array(parameters['conv1.bias'])
        weights["cvb2"] = np.array(parameters['conv2.bias'])
        weights["fcb1"] = np.array(parameters['fc1.bias'].squeeze())
        weights["fcb2"] = np.array(parameters['fc2.bias'].squeeze())
        weights["fcb3"] = np.array(parameters['fc3.bias'].squeeze())

        bn_params["running_mu_1"] = np.array(parameters['bn1.running_mean'])
        bn_params["running_sigma_1"] = np.array(parameters['bn1.running_var'])
        bn_params['gamma_1'] = np.array(parameters['bn1.weight'])
        bn_params['beta_1'] = np.array(parameters['bn1.bias'])

        bn_params["running_mu_2"] = np.array(parameters['bn2.running_mean'])
        bn_params["running_sigma_2"] = np.array(parameters['bn2.running_var'])
        bn_params['gamma_2'] = np.array(parameters['bn2.weight'])
        bn_params['beta_2'] = np.array(parameters['bn2.bias'])
            


# Creating a CNN class
class ConvNeuralNet(nn.Module):
    #  Determine what layers and their order in CNN object 
    def __init__(self, num_classes, activation=F.relu):
        super().__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3,  out_channels=16, kernel_size=3, padding="same")
        self.conv_layer2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding="same")
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(32)
        self.activation = activation
        #self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.pool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same")
        self.conv_layer5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.pool(out)
        out = self.conv_layer3(out)
        out = self.activation(self.bn1(out))
        out = self.pool(out)
        
        out = self.conv_layer4(out)
        out = self.conv_layer5(out)
        out = self.activation(self.bn2(out))
        out = self.pool(out)
        
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.activation(out)
        
        out = self.fc2(out)
        return out