import random

import graphviz as graphviz
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from data_processing import data_pipeline


def print_question(label):
    try:
        cat, answ = label.split(":")
        return f"{cat} -> {answ}?[y/n] "
    except ValueError:
        return f"{label}?[y/n] "


def print_prob(probs, y_label):
    for prob, label in zip(probs, y_label):
        print(f"\t{label} = {prob * 100:.2f}%")
    print("\n\n")


def interactive_questions(classifier, feat_lab, target_lab):
    tree = classifier.tree_
    node = 0  # Index of root node
    idx = random.choice(range(len(feat_lab)))
    point = np.zeros((1, len(feat_lab)))

    while True:
        feat, thres = tree.feature[node], tree.threshold[node]
        # print(feat_lab[feat], thres)
        v = input(print_question(feat_lab[feat]))
        if v == "n":
            node = tree.children_left[node]

        else:
            node = tree.children_right[node]
            point[0][feat] = 1

        prob = classifier.predict_proba(point)[0]
        print_prob(prob, y_label=y_labels)
        if tree.children_left[node] == tree.children_right[node]:  # Check for leaf
            label = np.argmax(tree.value[node])
            print(f"Predicted Label is: {target_lab[label]}")
            print(f"Gini : {tree.impurity[node]} (close to zero is betterr)")
            break


if __name__ == '__main__':
    path2file = "/home/dizzi/Downloads/Dataset.xlsx"
    random_state = 42

    X, y, y_labels = data_pipeline(path2file)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    x_labels = X_test.columns.values

    class_weight = {}

    for idx, yl in enumerate(y_labels):
        if "Brest" in yl:
            class_weight[idx] = 0.146
        elif "Colon" in yl:
            class_weight[idx] = 0.116
        elif "Lung" in yl:
            class_weight[idx] = 0.109
        elif "Prostate" in yl:
            class_weight[idx] = 0.096

    decision_tree = DecisionTreeClassifier(criterion="entropy", random_state=random_state,
                                           max_depth=4, class_weight=class_weight)

    decision_tree = decision_tree.fit(X_train, y_train)

    res = cross_val_score(decision_tree, X_test, y_test, cv=10)
    print(res)

    dot_data = tree.export_graphviz(decision_tree, out_file=None,
                                    feature_names=x_labels,
                                    class_names=y_labels,
                                    filled=True, rounded=True,
                                    special_characters=False)
    graph = graphviz.Source(dot_data)
    graph.save()
    graph.view()
    print("\n\n\n\n")
    interactive_questions(decision_tree, x_labels, y_labels)
