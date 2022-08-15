Follow this document to run the submitted code for the final project only.

Local machine config:
Python3 Version = 3.8.10
Pytorch Version = 1.6

For 'Training' follow the next steps
------------------------------------
Please place the 'cifar-10-batches-py' folder inside the 'data' folder. Otherwise, you need to specify the source or
CIFAR10 data will be downloaded at runtime.
Then run the following command
Example: python3 main.py train
Example: python3 main.py train --data_dir ../data

In local machine with data folder as mentioned above-->
$ python3 main.py train
In local machine with custom source location-->
$ python3 main.py train --data_dir '/path/to/data'

In HPRC or Colab -->
$ !python main.py train --data_dir '/path/to/data'
In HPRC or Colab with custom source location-->
$ !python main.py mode train --data_dir '/path/to/data'


For 'Testing' follow the next steps
-----------------------------------

For testing, provide the saved checkpoint of the model in the 'saved_models' folder and store inside the 'data' folder.
Then run the following command (Will not work properly with the checkpoint file).
Example: python3 main.py test --data_dir ../data --checkpoint=../saved_models/checkpoint_epoch220.chk

In local machine -->
$ python3 main.py test  --data_dir '/path/to/data' --checkpoint ../saved_models/chkp_epoch*.pt

In HPRC or Colab -->
$ !python main.py test  --data_dir '/path/to/data' --checkpoint '/path/to/saved_models/chkp_epoch*.pt'


For 'Prediction' follow the next steps
--------------------------------------

We need the private dataset for prediction. Please provide the 'private_test_images.npy' data
in the 'data' while the checkpoint file also need to be present in the 'saved_models' folder.
Then run the following command
Example: python3 main.py predict --checkpoint=../saved_models/checkpoint_epoch220.chk --save_dir ../saved_results/

In local machine -->
$ python3 main.py predict --data_dir '/path/to/data' --checkpoint ../saved_models/chkp_epoch240.pt --save_dir ./results

In HPRC or Colab -->
$ !python main.py predict --data_dir '/path/to/data' --checkpoint '/path/to/saved_models/chkp_epoch*.pt' --save_dir '/path/to/results'


All argument list
-----------------
--data_dir", default = model_configs['data_dir'], help="path to the data")             # default set in model_configs in Config.py for source data set location
--save_dir", default=training_configs['save_dir'], help="path to save training")       # default set in training_configs in Config.py for saving trained models
--result_dir", default=model_configs['result_dir'], help="path to save predictions")   # default set in model_configs in Config.py for saving prediction result
--lr', default=0.1, type=float, help='learning_rate')                                  # default=0.1 for learning_rate initial
--epochs", default=training_configs['epochs'], type=int, help='epochs')                # default=250 for epochs
--depth', default=model_configs['depth'], type=int, help='depth of model')             # default=28 for depth of model
--dropout', default=model_configs['drop_rate'], type=float, help='drop_rate')          # default=0.3 for drop_rate
--checkpoint", help="path to specific checkpoint file")                                # path to your best checkpoint saved
--gamma', default=training_configs['gamma'], type=float)                               # default=0.2 for gamma
--milestones', default='60,120,160,200', type=str)                                     # default='60,120,160,200' for lr change

