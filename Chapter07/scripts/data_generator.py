import numpy as np
import pandas as pd

from time import time, sleep
from sklearn.datasets import make_blobs
from matplotlib import pyplot


def log(message):
    print(f"\n[+] {message}\n")


def generate_event_time_records(num_records):
    time_value = int(round(time()))
    output = pd.Series([time_value]*num_records, 
                       dtype="float64")
    
    return output


def main():
    log("Generating the DataFrame of values")
    
    X, y = make_blobs(n_samples=5000, centers=2, 
                      cluster_std=[7, 3], n_features=2, 
                      random_state=43)

    n_samples = len(X)


    r1 = np.random.randint(low=-100, high=100, 
                           size=(n_samples,)).astype(int)

    r2 = np.random.randint(low=-100, high=100, 
                           size=(n_samples,)).astype(int)


    all_df = pd.DataFrame(dict(label=y, a=X[:,0], 
                               b=X[:,1], c=r1, d=r2))
    
    log("Addng the event_time column")

    all_df['index'] = range(1, len(all_df) + 1)

    all_df["event_time"] = generate_event_time_records(
        len(all_df)
    )
    
    print(all_df)
    
    log("Generating the scatterplot")

    colors = {0:'red', 1:'blue'}
    fig, ax = pyplot.subplots()
    grouped = all_df.groupby('label')

    for key, group in grouped:
        group.plot(ax=ax, 
                   kind='scatter', 
                   x='a', 
                   y='b', 
                   label=key, 
                   color=colors[key])

    pyplot.show()
    
    return (all_df)
    
    
log("Running the main() function")
all_df = main()