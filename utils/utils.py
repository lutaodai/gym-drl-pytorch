import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def str2tuple(v):
    v = eval(v)
    assert isinstance(v, tuple)
    return v
        
        
def plot_scores(scores, name, window_size, save_dir):
    scores = pd.DataFrame(scores, columns=["scores"])
    scores = scores.reset_index()
    scores["scores_avg"] = scores["scores"].rolling(window=window_size).mean()
    
    sns.set_style("dark")
    sns.relplot(x = "index", y = "scores",
                data=scores, kind="line")
    plt.plot(scores["index"], scores["scores_avg"], color=sns.xkcd_rgb["amber"])
    plt.legend(["Scores", "MA(%d)" %window_size])
    plt.savefig(os.path.join(save_dir, "score_plot_" + name + ".png"))
    
def env_summary():
    pass
