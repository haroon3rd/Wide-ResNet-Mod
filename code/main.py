### YOUR CODE HERE
import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, train_valid_split, load_testing_images
from Configure import model_configs, training_configs, preprocess_configs

parser = argparse.ArgumentParser()
parser.add_argument("mode", help="train, test or predict")                                                  # run mode: train, test or predict
parser.add_argument("--data_dir", default = model_configs['data_dir'], help="path to the data")             # default set in model_configs in Config.py for source data set location
parser.add_argument("--save_dir", default=training_configs['save_dir'], help="path to save training")       # default set in training_configs in Config.py for saving trained models
parser.add_argument("--result_dir", default=model_configs['result_dir'], help="path to save predictions")   # default set in model_configs in Config.py for saving prediction result
# Additional arguments
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')                                  # default=0.1 for learning_rate initial
parser.add_argument("--epochs", default=training_configs['epochs'], type=int, help='epochs')                # default=250 for epochs
parser.add_argument('--depth', default=model_configs['depth'], type=int, help='depth of model')             # default=28 for depth of model
parser.add_argument('--dropout', default=model_configs['drop_rate'], type=float, help='drop_rate')          # default=0.3 for drop_rate
parser.add_argument("--checkpoint", help="path to specific checkpoint file")                                # path to your best checkpoint saved
parser.add_argument('--gamma', default=training_configs['gamma'], type=float)                               # default=0.2 for gamma
parser.add_argument('--milestones', default='60,120,160,200', type=str)                                     # default='60,120,160,200' for lr change

args = parser.parse_args()


if __name__ == '__main__':
    #model = MyModel(model_configs)
    #data_dir = '../cifar-10-batches-py'

    if args.mode == 'train':
        print("Training begins for a total of " + str(args.epochs) +" epochs")
        # Load training and testing data
        train_data, test_data = load_data(args.data_dir, preprocess_configs)
        
        # initialize the model for training
        model = MyModel(model_configs)
        model.init_model(training_configs)

        # Test data is sent to get testing accuracy even while training
        model.train(train_data, test_data, training_configs)
        model.evaluate(test_data)

    elif args.mode == 'test':
        # initialize the model for testing
        model = MyModel(model_configs, args.checkpoint)
        
        # Testing on public testing dataset
        print("Testing on provided test dataset")
        _, test_data = load_data(args.data_dir, preprocess_configs)
        test_acc = model.evaluate(test_data)
        print("Test Accuracy = ", test_acc)

    elif args.mode == 'predict':
        # initialize the model for prediction
        model = MyModel(model_configs, args.checkpoint)
        
        # Prediction on private test dataset 
        print("Predictions on private data begins..")
        private_data = load_testing_images(args.data_dir, preprocess_configs)
        predictions = model.predict_prob(private_data)
        
        # Storing the prediction outputs
        print("Saving Prediction outputs....")
        output_path = os.path.join(args.result_dir,"predictions")
        if not os.path.exists(args.result_dir):
                    os.makedirs(args.result_dir)
        np.save(output_path, predictions)
        print("Prediction outputs Saved Successfully!")
    
    else:
        print("Un suppreted execution mode '" + args.mode + "'\nSupported modes: 'train', 'test', 'predict'" )

### END CODE HERE

