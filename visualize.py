import json
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

with open('data.json') as data:
    raw_data = json.load(data)
    y = np.array(raw_data[5][0])
    x = np.arange(len(y))

    # Threshold above which the line should be red
    threshold = 100


    #creates bounds...
    lower = np.ma.masked_where(y < threshold, y)
    upper = np.ma.masked_where(y > threshold, y)

    fig, ax = plt.subplots()
    ax.plot(x, upper, color="#31D9BD", label="Non-Stressed Labeling")
    ax.plot(x, lower, color="#F03681", label="Stressed Labeling")
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='best')

    plt.xlabel("Time (s)")
    plt.ylabel("Heart Rate (beats/min)")
    # plt.title("Individual's heart rate time series (BVP)")


    plt.show()
