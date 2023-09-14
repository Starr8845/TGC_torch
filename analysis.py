
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def draw_one_day_loss(b,path):
    edge_len = int(np.ceil(np.sqrt(b.shape[0])))
    c = np.zeros(edge_len*edge_len)
    c[:b.shape[0]] = b[:]
    c = c.reshape((edge_len, edge_len))
    print(c)

    df = pd.DataFrame(c)
    print(df)

    plt.figure(dpi=120)
    sns.heatmap(data=df)
    plt.savefig(path)
