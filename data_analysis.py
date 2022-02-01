import numpy as np
import pandas as pd
import plotly.express as px
import prince
from sklearn.decomposition import PCA
from corextopic import corextopic as ct
from data_processing import data_pipeline, dataset_file_path


def analysis(df: pd.DataFrame, y_labels):
    """
    Plot number samples and non-zero features for each label
    :param df:
    :param y_labels:
    :return:
    """
    idx = 0
    lab_dict = {}
    for lab in y_labels:
        tmp = df[df['labels'] == idx]
        tmp = tmp.loc[:, (tmp != 0).any(axis=0)]

        lab_dict[lab] = dict(total=len(tmp), cols=tmp.shape[1])

        print(f"Lablel '{lab}':\n"
              f"\tTotal : {len(tmp)}\n"
              f"\tNon zero columns: {tmp.shape[1]}")
        idx += 1

    lab_dict = pd.DataFrame(lab_dict)
    fig = px.bar(lab_dict.T)
    fig.show()


def plot_explained_variance(df):
    """
    Plot explained variance using PCA based on number of features
    :param df:
    :return:
    """
    pca = PCA(svd_solver="full")
    X = df.drop(labels="labels", axis=1)
    pca.fit(X)
    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

    fig = px.area(
        x=range(1, exp_var_cumul.shape[0] + 1),
        y=exp_var_cumul,
        labels={"x": "# Components", "y": "Explained Variance"}
    )
    fig.show()


def plot_pca(df):
    n_components = 4

    pca = PCA(n_components=n_components)
    X = df.drop(labels="labels", axis=1)

    components = pca.fit_transform(X)

    total_var = pca.explained_variance_ratio_.sum() * 100

    labels = {str(i): f"PC {i + 1}" for i in range(n_components)}
    labels['color'] = 'Median Price'

    fig = px.scatter_matrix(
        components,
        color=df['labels'],
        dimensions=range(n_components),
        labels=labels,
        title=f'Total Explained Variance: {total_var:.2f}%',
    )
    fig.update_traces(diagonal_visible=False)
    fig.show()


def plot_corex(df):
    X = df.drop(labels="labels", axis=1)

    topic_model = ct.Corex(n_hidden=50)
    words = X.columns.values
    docs = df["labels"].values
    # Define the number of latent (hidden) topics to use.
    topic_model.fit(X.values, words=words, docs=docs)
    a = 1


def plot_loadings(df):
    n_components = 10

    pca = PCA(n_components=n_components, svd_solver="full", whiten=True)
    X = df.drop(labels="labels", axis=1)

    components = pca.fit_transform(X)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    fig = px.scatter(components, x=0, y=1, color=df['labels'])

    for i, feature in enumerate(X):

        importance= abs(loadings[i, 0])+ abs(loadings[i, 1])

        if importance<0.05:
            continue

        fig.add_shape(
            type='line',
            x0=0, y0=0,
            x1=loadings[i, 0],
            y1=loadings[i, 1]
        )
        fig.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 1],
            ax=0, ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
        )


    fig.show()
if __name__ == '__main__':
    x_df, y_df, y_labels = data_pipeline(dataset_file_path)

    x_df['labels'] = y_df

    # analysis(x_df, y_labels)

    y_labels = {idx: y_labels[idx] for idx in range(len(y_labels))}
    x_df = x_df.replace({"labels": y_labels})
    # plot_explained_variance(x_df)
    # plot_pca(x_df)
    plot_loadings(x_df)