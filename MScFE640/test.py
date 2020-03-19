##{
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TKAgg')
import seaborn as sns
%matplotlib qt

sns.set()
tips = sns.load_dataset("tips")
sns.relplot(x="total_bill", y="tip", col="time",
           hue="smoker", style="smoker", size="size",
           data=tips)
##}
