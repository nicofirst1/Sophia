import pandas as pd


def read_data(path2file):
    """
    Read data from xlsx file and arrange with pandas dataframe
    :param path2file: str, path to file
    :return: pd.DataFrame
    """
    df = pd.read_excel(path2file)

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


def expand_all_key(dataframe):
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


def data_pipeline(path2file, classification=False):
    """
    Run the entire pipeline and return x/y data
    :param path2file:
    :return:
    """
    df = read_data(path2file)
    df = expand_all_key(df)

    label_indices = [x for x in df.columns if "Label" in x]

    y_data = df[label_indices]
    x_data = df.drop(label_indices, axis=1)

    if classification:
        y_data[y_data != 0] = 1
        label_encoding={y_data.columns[idx]:idx for idx in range(len(y_data.columns))}
        y_data["Labels"] = -1
        for k,v in label_encoding.items():
            y_data["Labels"][y_data[k]==1]=v

            a=12

    return x_data, y_data


if __name__ == '__main__':
    path2file = "/home/dizzi/Downloads/Dataset.xlsx"

    x_df, y_df = data_pipeline(path2file)

    print(x_df)
