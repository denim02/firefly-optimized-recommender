# Recommender system hyper-parameter tuning using the bio-inspired Firefly metaheuristic

<img src = "https://ecobnb.com/blog/app/uploads/sites/3/2016/06/08-all-that-glitters.jpg__1072x0_q85_upscale-870x490.jpg.webp" alt = "Fireflies"  width = "500"/>

To run this program, you must first install the following dependencies: numpy, cython, and scikit-surprise
To do so (if you have pip installed), simply run:

```sh
pip install numpy cython scikit-surprise
```
***If the command above generates an error, please try adding the ```--user``` flag.***

If pip cannot be found on your machine, first try executing:
```sh
python -m pip install numpy cython scikit-surprise
```
    
If the same error occurs, then pip should first be installed on your computer.

After that, to execute the script with its base settings, move into the /src directory and run:
```sh
python main.py
```

*To modify the algorithm parameters, including the number of fireflies, epochs, and the tuning measure used, please open main.py and find the section marked as **MODIFY HERE**:*
```py
##### MODIFY HERE #####
params = {
    'reg_all': (0.01, 0.5),
    'n_factors': (25, 300),
    'n_epochs': (20, 200),
    'lr_all': (0.001, 0.05)
}
fireflies = 2
epochs = 2
useRMSE = True      # Could set to false and use hit-rate to tune instead
autoStop = True     # Automatically stops when RMSE/Hit-rate stop improving
######################
```