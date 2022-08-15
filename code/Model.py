### YOUR CODE HERE
import numpy as np
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from Network import MyNetwork
from tqdm import tqdm

"""This script defines the training, validation and testing process.
"""

class MyModel(object):

    def __init__(self, config, chkp_path = None):
        # Enables benchmark mode in cudnn. (Fixed input size, Faster Runtime)
        cudnn.benchmark = True
        self.config = config
        self.network = MyNetwork(config)
        # Comment this line out if you want to run on CPU
        self.network = self.network.cuda()

        if chkp_path is not None:
            if os.path.exists(chkp_path):
                print("Provided checkpoint " + chkp_path + " verified")
            else:
                raise Exception("Invalid checkpoint path or path does not exit: ", chkp_path)
            self.network.load_state_dict(torch.load(chkp_path))


    def init_model(self, training_configs):
        self.network_loss = nn.CrossEntropyLoss().cuda()
        self.network_optimizer = torch.optim.SGD(self.network.parameters(),
                                            training_configs["learning_rate"],
                                            momentum=0.9,
                                            nesterov=True,
                                            weight_decay=training_configs["weight_decay"])
        self.scheduler = MultiStepLR(self.network_optimizer,
                                milestones=[60, 120, 160, 200],
                                gamma=training_configs['gamma'])


    def train(self,  train_data, test_data, training_configs):
        for epoch in range(training_configs["epochs"]):
            avg_loss = 0.
            true_val = 0.
            total = 0.

            progress_bar = tqdm(train_data)
            for i, (images, labels) in enumerate(progress_bar):
                progress_bar.set_description('Epoch ' + str(epoch))
                images, labels = images.cuda(), labels.cuda()
                
                self.network.zero_grad()
                pred = self.network(images)
                loss = self.network_loss(pred, labels)
                loss.backward()
                
                self.network_optimizer.step()
                avg_loss += loss.item()
                pred = torch.max(pred.data, 1)[1]
                total += labels.size(0)
                true_val += (pred == labels.data).sum().item()
                accuracy = true_val / total
                
                progress_bar.set_postfix(
                xentropy='%.5f' % (avg_loss / (i + 1)),
                train_acc='%.5f' % accuracy)

            test_acc = self.evaluate(test_data)
            print('[Epoch:{:d} - Train_Acc: {:.5f} / Test_Acc: {:.5f}]'.format(epoch, accuracy, test_acc))
            self.scheduler.step()

            if (epoch+1) % 10 == 0:
                fname = "checkpoint_epoch" + str(epoch +1) + ".chk"
                fpath = training_configs['save_dir']
                chkp_path = os.path.join(fpath,fname)
                if not os.path.exists(fpath):
                    os.makedirs(fpath)
                torch.save(self.network.state_dict(), chkp_path)
                print("New checkpoint saved in: " + str(chkp_path))


    def evaluate(self, test_data):
        self.network.eval()
        true_val = 0.
        total = 0.

        for images, labels in test_data:
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                pred = self.network(images)

            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            true_val += (pred == labels).sum().item()

        val_acc = true_val / total
        self.network.train()
        return val_acc

    def predict_prob(self, private_test_data):
        self.network.eval()
        pred_list = []

        for images in private_test_data:
            #images = parse_record(images)
            images = images.cuda()

            with torch.no_grad():
                pred_list.extend(self.network(images).cpu().numpy())

        pred_linear = np.asarray(pred_list)
        # softmax to get the probablities
        pred_exp = np.exp(pred_linear)
        pred = pred_exp/np.sum(pred_exp, axis=-1, keepdims=True)
        return pred
### END CODE HERE