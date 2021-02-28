# Optimal Active Learning Behaviors

This is the code repository accompanying the AISTATS 2021 paper [_Towards Understanding the Behaviors of Optimal Deep Active Learning Algorithms_](https://arxiv.org/abs/2101.00977) by Yilun Zhou, Adithya Renduchintala, Xian Li, Sida Wang, Yashar Mehdad and Asish Ghoshal. 

There are three tasks, `object_classification`, `intent_classification`, and `named_entity_recognition`. Specific instructions are listed in `<task>/README.md` for each task. 

Before proceeding, please download the preprocessed data as a zip file from [this link](http://bit.ly/optimal-al-data), and unpack the contents of `<task>/data/` into the the currently empty `<task>/data/` folder. 

In `<task>/README.md`, the first step is to search for the optimal order, which takes several days _per search_ on 8 V100 GPUs, using the settings in the paper. Thus, we have saved the log files for each task. You can download all of them from [this link](http://bit.ly/optimal-al-logs), and unpack the contents of `<task>/logs/` into the currently empty `<task>/logs/` folder. 

All the plots will be saved in `figures/<task>/` folder, which is currently populated with the those used in the paper. 

`requirements.txt` contains a (more than) sufficient list of packages, along with their versions, to run the code. The code should run with reasonably recent versions of `pytorch`, `numpy`, `scipy`, `matplotlib`, `scikit-learn`, etc., but if there are any compatibility issues, please try again with the exact versions specified in this file. 

For any questions, please contact Yilun Zhou at yilun@mit.edu. The paper can be cited as
```
@inproceedings{zhou2021towards,
  title={Towards Understanding the Behaviors of Optimal Deep Active Learning Algorithms},
  author={Zhou, Yilun and Renduchintala, Adithya and Li, Xian and Wang, Sida and Mehdad, Yashar and Ghoshal, Asish},
  booktitle={International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year={2021},
  organization={PMLR}
}
```
