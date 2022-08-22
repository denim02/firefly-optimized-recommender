# Recommender system hyper-parameter tuning using the bio-inspired Firefly metaheuristic
## !!! TESTING BRANCH (Use only to re-test tuned parameters) !!!
### For the actual FA implementation, go to the main branch.

<p align="center">
    <img src = "https://ecobnb.com/blog/app/uploads/sites/3/2016/06/08-all-that-glitters.jpg__1072x0_q85_upscale-870x490.jpg" alt = "Fireflies"  width = "500"/>
</p>

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
tuned_SVD_params = {
    'reg_all': 0.002987,
    'n_factors': 93,
    'n_epochs': 87,
    'lr_all': 0.002987
}
#######################
```