
model = 'Model architecture. Choices are "lstm", "cnn", "aoe", and "roberta". '
model_seed = 'Integer value of model seed for initialization (and RoBERTa dropout). '

search_model_seed = 'Seed for the search model'
eval_model_seed = 'Seed for the evaluation model'

domain = 'TOPv2 domain. Choices are "alarm", "event", "messaging", "music", "navigation", "reminder", "timer", and "weather"'
data_seed = 'The seed for the random split of data into warmstart, model-selection, pool, validation, and test sets'

batchsize = 'Batch-size for training and data acquistion. '
max_epoch = 'Maximum number of epochs for training. '
patience = 'Patience value to trigger early stopping. '
tot_acq = 'Total number of acquired data points (excluding the warmstart set)'

use_gpus = 'A comma-separated list of GPU device indices to deploy the program on, or "all" to use all available GPUs. '
workers_per_gpu = 'Number of train servers to deploy per GPU device'

anneal_factor = 'Annealing factor for the simulated annealing search. '
num_sa_samples = 'Total number of samples during the simulated annealing stage. '
num_greedy_samples = 'Total number of samples during the greedy stage. '
log_dir = 'Directory for the search log file. '

criterion = 'Selection criterion. Choices are "max-entropy" and "bald". '
gpu_idx = 'The index of GPU device to deploy the program on. '
evaluation_set = 'The set to evaluate on. Choices are "valid" and "test". '

model_seeds = 'A list of model seeds. '
criterions = 'A list of criterions. '

num_random_samples = 'Total number of random samples. '
