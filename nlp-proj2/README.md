# Code for HW2

## Dependencies

```
torch>=1.9.0
```

Tested with Windows 10, Python 3.6, `torch==1.9.1+cu111` and RTX 3060 Mobile GPU

## Run

```
python3 hw2.py --model <MODEL_TYPE>
```
where `<MODEL_TYPE>` should be one of `{dense, RNN, extension1, extension2}`.  

Note that `main()` in `hw2.py` does not run all 4 models sequentially. Please run them one by one using the above command.


## Extension

* Extension 1: A simple 1D-convolutional network. See `models.py` for more details.
* Extension 2: A cosine annealing scheduler with warmup, and a different stopping strategy. See `hw2.py` for more details. 
