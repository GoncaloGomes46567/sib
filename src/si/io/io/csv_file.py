import csv

def read_csv(filename, sep=",", features=True, label=True):
    with open(filename, "r", newline="") as file:
        reader = csv.reader(file, delimiter=sep)
        rows = list(reader)

    if not rows:
        return [], None, None

    feature_names = None
 
    if features:
        feature_names = rows[0]
        rows = rows[1:]

    if not rows:
        return [], [], feature_names

    if label:
        X = [row[:-1] for row in rows]
        y = [row[-1] for row in rows]
    else:
        X = rows
        y = None

    return X, y, feature_names
