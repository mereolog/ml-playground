import os
import pandas as pd

DATASETS = [
    {
        "name": "spambase",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
        "filename": "spambase.csv"
    },
    {
        "name": "titanic",
        "url": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "filename": "titanic.csv"
    },
    {
        "name": "iris",
        "url": "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv",
        "filename": "iris.csv"
    },
]

def download_and_save_csv(url, path, header="infer"):
    print(f"Pobieranie: {url}")
    df = pd.read_csv(url, header=header)
    df.to_csv(path, index=False)
    print(f"Zapisano do: {path}")

def main():
    target_folder = "datasets"
    os.makedirs(target_folder, exist_ok=True)

    for ds in DATASETS:
        save_path = os.path.join(target_folder, ds["filename"])
        # Spambase nie ma nagłówka w oryginalnym pliku
        if ds["name"] == "spambase":
            download_and_save_csv(ds["url"], save_path, header=None)
        else:
            download_and_save_csv(ds["url"], save_path)

if __name__ == "__main__":
    main()