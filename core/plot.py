import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def dynamics(label, threshold=0.05):
    with open("modelArxivICML/"+label+"/dynamics.json", "r") as f:
        data = json.load(f)
    serieses = []
    for snapshot in data:
        reform = {(outerKey, innerKey): values for outerKey, innerDict in snapshot.iteritems()
                  for innerKey, values in innerDict.iteritems()}
        serieses.append(pd.Series(reform))
    df = pd.DataFrame(serieses)
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    sns.heatmap(df.loc[:, df.max(axis=0) > threshold].transpose())
    plt.yticks(rotation=0)
    plt.show()