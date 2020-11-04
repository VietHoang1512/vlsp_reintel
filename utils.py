import os
import numpy as np 
import random

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf

def seed_all(seed=1512):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)

def length_plot(lengths):
    plt.figure(figsize=(12, 7))
    textstr = f' Mean: {np.mean(lengths):.2f} \u00B1 {np.std(lengths):.2f} \n Max: {np.max(lengths)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    plt.text(0, 0, textstr, fontsize=14,
             verticalalignment='top', bbox=props)
    sns.countplot(lengths)