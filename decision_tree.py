import graphviz as graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from data_processing import data_pipeline

from sklearn.datasets import load_iris


if __name__ == '__main__':
    path2file = "/home/dizzi/Downloads/Dataset.xlsx"
    classification = True
    random_state = 42

    iris = load_iris()
    X, y = data_pipeline(path2file, classification=classification)

    y_label=y.columns[:-1]
    y_label=[x.strip("Labels") for x in y_label]
    y=y['Labels']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    attributes=dict(
        random_state=random_state
    )

    if classification:
        decision_tree = DecisionTreeClassifier(criterion="entropy",**attributes)
    else:
        decision_tree = DecisionTreeRegressor(criterion="squared_error",**attributes)

    decision_tree = decision_tree.fit(X_train, y_train)

    res = cross_val_score(decision_tree, X_test, y_test, cv=10)
    dot_data = tree.export_graphviz(decision_tree, out_file=None,
                                    feature_names=X_train.columns,
                                    class_names=y_label,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.view()
    print(res)
