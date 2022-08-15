# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE

model_configs = {
	"name": 'MyModel',
	"mode": 'train',
	"data_dir": '../data/',
	"result_dir": '../saved_result/',
	"depth": 28,
	"num_classes": 10,
	"drop_rate": 0.3,
	"widen_factor": 10,
	# ...
}

training_configs = {
	"learning_rate": 0.1,
	"batch_size": 128,
	"epochs": 250,
	"gamma": 0.2,
	"save_dir": '../saved_models/',
	"weight_decay": 5e-4, 
	# ...
}

preprocess_configs = {
	"crop" : True,
	"crop_padding" : 4,
	"flip" : True,
 	"cutout" : True,
	"cutout_holes": 1,
	"cutout_length": 16,
	"batch_size": 128,
	# ...
}

### END CODE HERE