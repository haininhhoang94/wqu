# import_package
import os
import pickle
from functools import reduce
from operator import mul

import pandas as pd
import numpy as np

from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn import linear_model
from sklearn.decomposition import PCA

import holoviews as hv
import hvplot
import hvplot.pandas

# check_bokeh_extension
np.random.seed(42)
hv.extension("bokeh")

# import pandas_datareader
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as pdr

# Import our data from CSV
indexes = pd.read_csv("./Data/StyleIndexes.csv")

# We ensure the dates are recorded correctly and compute returns
indexes.Date = pd.to_datetime(indexes.Date)
indexes.index = indexes.Date
indexes = indexes.drop(columns=["Date"])
indexes = indexes.pct_change().dropna()

# As this is a large dataset, we will only look at the last 1000 tradings day
pivot = indexes.iloc[::-1, :].iloc[-1000:, :]


# Example for PCA to identify risk to common risk or common market fundamental
class Component_Plots:
    def __init__(self, data=pivot, transformer=PCA(2), labels=True):
        self.data = data
        self.transformer = transformer
        self.labels = labels

    def components(self, start, window):
        component_data = self.transformer.fit_transform(
            self.data.iloc[start : (start + window), :].T
        )

        if self.labels:
            data_labels = reduce(
                mul,
                pd.DataFrame(
                    component_data,
                    index=self.data.columns.tolist(),
                    columns=["Component_1", "Component_2"],
                )
                .reset_index()
                .apply(
                    lambda x: hv.Text(
                        x[1], x[2], " ".join(x[0].split()[:-1]), fontsize=8
                    ),
                    axis=1,
                )
                .tolist(),
            )
        else:
            data_labels = hv.Text(0, 0, "")

        return (
            pd.DataFrame(component_data, columns=["Component_1", "Component_2"])
            .hvplot.scatter(x="Component_1", y="Component_2")
            .redim(
                Component_2={"range": (-0.1, 0.3)}, Component_1={"range": (-0.03, 0.05)}
            )
            .redim.label(
                Component_1=f"Component 1 {self.transformer.explained_variance_ratio_[0].round(4)}%",
                Component_2=f"Component 2 {self.transformer.explained_variance_ratio_[1].round(4)}%",
            )
            .options(alpha=1)
            * data_labels
        )


# We download FRED data on 3-month Tbills,
#  NASDAQ Comp and Exchnage rate mvts.
factors = pdr.data.DataReader(
    ["DTB3", "NASDAQCOM", "DTWEXB"],
    "fred",
    start=str(pivot.index.min()),
    end=str(pivot.index.max()),
)
factors.loc[:, ["NASDAQCOM", "DTWEXB"]] = factors.loc[
    :, ["NASDAQCOM", "DTWEXB"]
].pct_change()
factors.loc[:, ["DTB3"]] = ((factors.loc[:, ["DTB3"]] + 1) ** (1 / 365)).pct_change()
factors = factors.dropna()

pd.melt(factors.add(1).cumprod().reset_index(), id_vars=["Date"]).hvplot.line(
    y="value", x="Date", by="variable"
)
