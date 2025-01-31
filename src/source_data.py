from pathlib import Path
import shutil
import os
import json
import opendatasets as od

def set_kaggle_credentials(file_path="kaggle.json", print_credentials=True):
    """
    Reads the Kaggle API credentials from a kaggle.json file in the current directory
    and sets them as environment variables.
    
    :param file_path: Path to the kaggle.json file (default: 'kaggle.json' in the current directory)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: {file_path} not found. Please ensure the file is in the current directory.")

    try:
        with open(file_path, 'r') as f:
            credentials = json.load(f)

        os.environ['KAGGLE_USERNAME'] = credentials['username']
        os.environ['KAGGLE_KEY'] = credentials['key']

        print("Kaggle credentials set successfully.")
        if print_credentials:
            print(f"username:{credentials['username']}")
            print(f"username:{credentials['key']}")

    except (json.JSONDecodeError, KeyError) as e:
        print("ERROR")
        print("You must have a valid kaggle.json file with credentials, in current directory, to authenticate to kaggle")
        raise ValueError(f"Error reading {file_path}: {e}")



def download_source_data(overwrite:bool = False):

    data_dir = 'data'
    target_file_path = os.path.join(data_dir,'credit-data.csv')
    url = "https://www.kaggle.com/datasets/kelvinkelue/credit-card-fraud-prediction/download?datasetVersionNumber=1"
    dl_data_dir = os.path.join(data_dir,'downloaded')
    dl_target_dir = os.path.join(dl_data_dir,"credit-card-fraud-prediction")
    dl_target_file = os.path.join(dl_target_dir, "fraud test.csv")

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    isExisting = os.path.exists(target_file_path)
    if (overwrite) or (not isExisting):
        od.download(url, data_dir=dl_data_dir, force=overwrite)
        shutil.move(src=dl_target_file, dst=target_file_path)
        shutil.rmtree(dl_data_dir)





if __name__ == "__main__":
    set_kaggle_credentials()
    download_source_data(overwrite=False)
    pass
