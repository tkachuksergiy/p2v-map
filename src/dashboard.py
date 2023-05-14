import re
import glob
import warnings
import numpy as np
import pandas as pd

import sklearn.manifold
import sklearn.model_selection

import plotly.graph_objs as go
from plotly.offline import iplot

class DashboardP2V(object):
    def __init__(self, path, n_steps=20):
        self.path = path
        self.n_steps = n_steps
        self.files = {
            "wi": "wi_*.npy",
        }
        self.file_df = None
        self.plot_data = {}
        self.plot_data["steps"] = []
        self.plot_data["wi"] = []
        self.plot_data["tsne"] = []
        self.raw_files = glob.glob(f"{self.path}/{self.files['wi']}")

    def plot_product_embedding(
        self,
        idx=None,
        label="wi",
        size=None,
        l2norm=True,
        transpose=True,
        reload=True,
    ):
        if reload:
            self._load_data(idx=idx, agg=None)
        data = []
        steps = []
        for i in range(self.n_steps):
            data_i = self.plot_data[label][i]
            # data
            if size is not None:
                data_i = data_i.reshape(size)
            if l2norm:
                data_i /= np.linalg.norm(data_i, axis=1)[:, np.newaxis]
            if transpose:
                data_i = data_i.T
            data.append(go.Heatmap(z=data_i, colorscale="Jet", zmin=-0.6, zmax=0.6))
            # step
            step = dict(
                method="restyle",
                label=self.plot_data["steps"][i],
                args=["visible", [False] * self.n_steps],
            )
            step["args"][1][i] = True
            steps.append(step)
        sliders = dict(
            active=0, currentvalue={"visible": False}, pad={"t": 50}, steps=steps
        )
        layout = go.Layout(
            height=700,
            width=1000,
            sliders=[sliders],
            margin=go.layout.Margin(l=50, r=50, b=150, t=20, pad=4),
            template="plotly_white",
        )
        return iplot(dict(data=data, layout=layout))

    def plot_tsne_map(self, product, config):
        if len(self.plot_data["tsne"]) < self.n_steps:
            self._tsne(product, config)
        data = []
        steps = []
        for i in range(self.n_steps):
            data_tsne_i = self.plot_data["tsne"][i]
            trace = go.Scatter(
                x=data_tsne_i["x"].values,
                y=data_tsne_i["y"].values,
                text=[
                    f"c: {c} <br> j: {c}"
                    for (c, j) in zip(
                        data_tsne_i["product"].values,
                        data_tsne_i["category"].values,
                    )
                ],
                hoverinfo="text",
                mode="markers",
                marker=dict(
                    size=12,
                    color=data_tsne_i["category"].values,
                    colorscale="Jet",
                    showscale=False,
                ),
            )
            data.append(trace)
            # step
            step = dict(
                method="restyle",
                label=self.plot_data["steps"][i],
                args=["visible", [False] * self.n_steps],
            )
            step["args"][1][i] = True
            steps.append(step)
        sliders = dict(
            active=0, currentvalue={"visible": False}, pad={"t": 50}, steps=steps
        )
        layout = go.Layout(
            height=700,
            width=800,
            sliders=[sliders],
            margin=go.layout.Margin(l=0, r=0, b=0, t=0, pad=4),
            # margin=go.layout.Margin(l=50, r=50, b=150, t=20, pad=4),
            hovermode="closest",
            template="plotly_white",
            xaxis=dict(range=[-2.2, 2.2]),
            yaxis=dict(range=[-2.2, 2.2]),
            showlegend=False,
        )
        return iplot(dict(data=data, layout=layout))

    def _tsne(self, product, config):
        config_copy = config.copy()
        for i in range(self.n_steps):
            if i == 0:
                config_copy["init"] = "pca"
            else:
                config_copy["init"] = self.plot_data["tsne"][i - 1][["x", "y"]].values
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                tsne = sklearn.manifold.TSNE(**config_copy)
                X = tsne.fit_transform(self.plot_data["wi"][i])
            X = X - X.mean(axis=0)
            X = X / X.std(axis=0)
            df = product.copy()
            df["x"] = X[:, 0]
            df["y"] = X[:, 1]
            self.plot_data["tsne"].append(df)

    def _build_file_df(self, idx):
        files = [f for f in self.raw_files if re.search(r"(\d+)_(\d+).npy", f)]
        if not files:
            return None
        df = pd.DataFrame({"file": files})
        epoch_batch = df["file"].str.extract(r"(\d+)_(\d+).npy").astype(np.int32)
        epoch_batch.rename(columns={0: "epoch", 1: "batch"}, inplace=True)
        df = pd.concat([df, epoch_batch], axis=1)
        df = df.sort_values(["epoch", "batch"]).reset_index(drop=True)
        if self.n_steps < df.shape[0]:
            if idx is None:
                rows_keep = (
                    np.linspace(0, 1, self.n_steps) ** 2 * (df.shape[0] - 1)
                ).astype(int)
            else:
                # rows_keep = [i for i in idx if i <= df.shape[0]]
                rows_keep = idx
            df = df.iloc[rows_keep]
        self.file_df = df

    def _load_data(self, idx=None, agg=None):
        if len(idx) < self.n_steps:
            self.n_steps = len(idx)
        self._build_file_df(idx=idx)
        df = self.file_df
        if df is not None:
            data = []
            for i, x in df.iterrows():
                data_i = np.load(x["file"])
                if agg is not None:
                    data_i = agg(data_i)
                data.append(data_i)
            self.plot_data["wi"] = data
            self.plot_data["steps"] = [
                f"e{x:02d}-b{y:06d}" for (x, y) in zip(df.epoch.values, df.batch.values)
            ]
        self.plot_data["tsne"] = []