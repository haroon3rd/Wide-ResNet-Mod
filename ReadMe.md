# Wide Residual Network with Modified Augmentation Strategies

# How to train, test & predict 
## Follow this document to run, test, predict based on the submitted code for the final project only.
Local machine config:
***Python3 Version = 3.8.10***
***Pytorch Version = 1.6***

## For 'Training' follow the next steps
Please place the 'cifar-10-batches-py' folder inside the **'data'** folder in the root directory where **code** folder exists. Otherwise, you need to specify the source or **CIFAR10 data will be downloaded at runtime**.
Then run the following command
```
$ python3 main.py train
```
or
```
$ python3 main.py train --data_dir ../data
```
Or, in local machine with data folder as mentioned above-->
```
$ python3 main.py train
```
Or, in local machine with custom source location-->
```
$ python3 main.py train --data_dir '/path/to/data'
```
Or, in HPRC or Colab -->
```
$ !python main.py train --data_dir '/path/to/data'
```
Or, in HPRC or Colab with custom source location-->
```
$ !python main.py mode train --data_dir '/path/to/data'
```

## For 'Testing' follow the next steps

For testing, provide the saved checkpoint of the model in the **'saved_models'** folder in the root directory where **code** folder exists.  Then run the following command (Will not work properly without the checkpoint file).
```
$ python3 main.py test --data_dir ../data --checkpoint=../saved_models/checkpoint_epoch220.chk
```
Or, in local machine -->
```
$ python3 main.py test  --data_dir '/path/to/data' --checkpoint ../saved_models/chkp_epoch*.pt
```
Or, in HPRC or Colab -->
```
$ !python main.py test  --data_dir '/path/to/data' --checkpoint '/path/to/saved_models/chkp_epoch*.pt'
```

## For 'Prediction' follow the next steps

We need the private dataset for prediction. Please provide the 'private_test_images_v3.npy' data
in the **'data'** while the checkpoint file also need to be present in the **'saved_models'** folder.
Then run the following command
```
$ python3 main.py predict --checkpoint=../saved_models/checkpoint_epoch220.chk --save_dir ../saved_results/
```
Or, in local machine -->
```
$ python3 main.py predict --data_dir '/path/to/data' --checkpoint ../saved_models/chkp_epoch240.pt --save_dir ./results
```
Or, in HPRC or Colab -->
```
$ !python main.py predict --data_dir '/path/to/data' --checkpoint '/path/to/saved_models/chkp_epoch*.pt' --save_dir '/path/to/results'
```

All arguments list
-----------------
**--data_dir**, default = model_configs['data_dir'], help="path to the data") 
**--save_dir**, default=training_configs['save_dir'], help="path to save training")     
**--result_dir**, default=model_configs['result_dir'], help="path to save predictions") 
**--lr**, default=0.1, type=float, help='learning_rate')   
**--epochs**, default=training_configs['epochs'], type=int, help='epochs') 
**--depth**, default=model_configs['depth'], type=int, help='depth of model')    
**--dropout**, default=model_configs['drop_rate'], type=float, help='drop_rate')     
**--checkpoint**, help="path to specific checkpoint file") 
**--gamma**, default=training_configs['gamma'], type=float)
**--milestones**, default='60,120,160,200', type=str)
