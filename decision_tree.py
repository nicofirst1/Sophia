import graphviz as graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from data_processing import data_pipeline

if __name__ == '__main__':
    path2file = "/home/dizzi/Downloads/Dataset.xlsx"
    classification = True
    random_state = 42

    X, y, y_labels = data_pipeline(path2file, classification=classification)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    x_labels = X_test.columns.values

    attributes = dict(
        random_state=random_state
    )

    if classification:
        decision_tree = DecisionTreeClassifier(criterion="entropy", **attributes)
    else:
        decision_tree = DecisionTreeRegressor(criterion="squared_error", **attributes)

    decision_tree = decision_tree.fit(X_train, y_train)

    res = cross_val_score(decision_tree, X_test, y_test, cv=10)
    print(res)

    dot_data = tree.export_graphviz(decision_tree, out_file=None,
                                    feature_names=x_labels,
                                    class_names=y_labels,
                                    filled=True, rounded=True,
                                    special_characters=False)
    graph = graphviz.Source(dot_data)
    graph.view()
