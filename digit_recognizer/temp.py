import os
import re
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


with open(os.path.dirname(__file__) + '/data/digit_epoch.log') as fp:
    lines = fp.readlines()
    records = []
    pattern = re.compile(r".*\'epoch\': (\d+).+\'evaluation_accuracy\': tensor\((\d+\.\d+)")

    for line in lines:
        if "evaluation_accuracy" in line:
            items = pattern.findall(line)
            if len(items) == 1 and len(items[0]) == 2:
                records.append({
                    "epoch": int(items[0][0]),
                    "accuracy": float(items[0][1])
                })

    records_df = pd.DataFrame(records)
    # records_df.to_csv(os.path.dirname(__file__) + '/data/epoch_accuracies.csv', index=False)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot(records_df['epoch'], records_df['accuracy'] * 100)

    ax.set(xlabel='epoch', ylabel='accuracy (%)')
    ax.grid()

    fig.savefig("epoch_accuracy.png")

