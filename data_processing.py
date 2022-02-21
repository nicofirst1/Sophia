import pandas as pd

dataset_file_path = "/home/dizzi/Downloads/Dataset.xlsx"


def read_data(path2file):
    """
    Read data from xlsx file and arrange with pandas dataframe
    :param path2file: str, path to file
    :return: pd.DataFrame
    """
    df = pd.read_excel(path2file, sheet_name=1)

    x_labels = []
    cur_label = ""
    label_idx = -1
    for index, row in df.iterrows():
        if not pd.isna(row['Domande']):
            cur_label = row['Domande']
            if "Labels" in row['Domande']:
                label_idx = index
        if not pd.isna(row['Risposte']):
            x_labels.append(f"{cur_label}:{row['Risposte']}")
        else:
            x_labels.append(f"{cur_label}")

    y_labels = [elem for elem in x_labels if "Labels" in elem]

    x_labels = [elem for elem in x_labels if "Labels" not in elem]

    df = df.fillna(0)
    df = df.iloc[:, 2:]
    y_data = df[label_idx:]
    x_data = df[:label_idx]

    y_data = y_data.transpose()
    x_data = x_data.transpose()

    x_data = pd.DataFrame(x_data.values, columns=x_labels)
    y_data = pd.DataFrame(y_data.values, columns=y_labels)

    df = pd.concat([x_data, y_data], axis=1)
    return df


def expand_all_key2(dataframe):
    """
    Some of the questions have multiple answers, such answers have also 'all' as a possibility.
    Exapand all to all possibilities
    :param dataframe:
    :return:
    """
    for all_coll in dataframe.columns.values:
        if "all" not in all_coll:
            continue

        # get parent name
        parent_label = all_coll.strip(":all")

        # get rows where all_coll is 1
        all_rows = dataframe[dataframe[all_coll] == 1]

        # drop all_coll
        all_rows = all_rows.drop(all_coll, axis=1)

        new_rows = []
        for idx, row in all_rows.iterrows():

            row_indices = [elem for elem in row.index.values if parent_label in elem]

            for r_idx in row_indices:
                n_row = row.copy()
                n_row[r_idx] = 1
                new_rows.append(n_row)

        # drop indices where all=1
        dataframe = dataframe.drop(all_rows.index)
        dataframe = dataframe.drop(all_coll, axis=1)

        new_rows = pd.concat(new_rows, axis=1).T
        dataframe = pd.concat([dataframe, new_rows], ignore_index=True)

    return dataframe


def expand_all_key(dataframe, key=""):
    """
    Some of the questions have multiple answers, such answers have also 'all' as a possibility.
    Exapand all to all possibilities
    :param dataframe:
    :return:
    """
    for all_coll in dataframe.columns.values:
        if "all" not in all_coll:
            continue

        if key!="" and key not in all_coll:
            dataframe = dataframe.drop(all_coll, axis=1)
            continue

        # get parent name
        parent_label = all_coll.strip(":all")

        # get rows where all_coll is 1
        all_rows = dataframe[dataframe[all_coll] == 1]

        # drop all_coll
        all_rows = all_rows.drop(all_coll, axis=1)

        new_rows = []
        for idx, row in all_rows.iterrows():

            row_indices = [elem for elem in row.index.values if parent_label in elem]

            if "Age" in parent_label:
                min_ = sorted([x for x in row_indices if "<" in x])[-1]
                max_ = [x for x in row_indices if ">" in x][0]
                row_indices = [min_, max_]

            for r_idx in row_indices:
                n_row = row.copy()
                n_row[r_idx] = 1
                new_rows.append(n_row)

        # drop indices where all=1
        dataframe = dataframe.drop(all_rows.index)
        dataframe = dataframe.drop(all_coll, axis=1)

        new_rows = pd.concat(new_rows, axis=1).T
        dataframe = pd.concat([dataframe, new_rows], ignore_index=True)

    return dataframe




def drop_all_key(dataframe):
    """
    Some of the questions have multiple answers, such answers have also 'all' as a possibility.
    Exapand all to all possibilities
    :param dataframe:
    :return:
    """
    for all_coll in dataframe.columns.values:
        if "all" not in all_coll:
            continue

        dataframe= dataframe.drop(all_coll, axis=1)


    return dataframe


def categoriacl_data(df):
    """
    Transform from one hot encoder to category
    :param df:
    :return:
    """
    labels = [x for x in df.columns.values if ":" in x]
    labels = [x for x in labels if "all" not in x]

    parents_l = [x.split(":")[0] for x in labels]
    parents_l = list(set(parents_l))

    label_dict = dict()
    for p in parents_l:
        label_dict[p] = {}

        lbs = [x for x in labels if p in x]
        idx = 1
        for l in lbs:
            label_dict[p][l] = idx
            idx += 1

    for p in parents_l:

        lbs = [x for x in labels if p in x and "all" not in x]
        p_df = df[lbs]
        for l in lbs:
            p_df[p_df > 0] = 1
            p_df[l].replace({1: label_dict[p][l]}, inplace=True)

        p_df = p_df.sum(axis=1)
        df = df.drop(lbs, axis=1)
        df[p] = p_df

    return df.drop_duplicates()


def data_pipeline(path2file):
    """
    Run the entire pipeline and return x/y data
    :param path2file:
    :return:
    """
    df = read_data(path2file)
    df = drop_all_key(df)

    # df = categoriacl_data(df)

    label_indices = [x for x in df.columns if "Label" in x]

    y_data = df[label_indices]
    x_data = df.drop(label_indices, axis=1)
    y_labels = [x.strip("Labels") for x in y_data.columns]
    y_labels = [x.strip(":") for x in y_labels]

    y_data[y_data != 0] = 1
    label_encoding = {y_data.columns[idx]: idx for idx in range(len(y_data.columns))}
    y_data["Labels"] = -1
    for k, v in label_encoding.items():
        y_data["Labels"][y_data[k] == 1] = v

    y_data = y_data["Labels"]

    return x_data, y_data, y_labels


def get_labels(path2file):
    x_data, _, y_labels = data_pipeline(path2file)
    x_labels = x_data.columns.values

    x_labels = x_labels.tolist()

    return x_labels + y_labels


if __name__ == '__main__':
    x_df, y_df, y_labels = data_pipeline(dataset_file_path)

    print(x_df)
