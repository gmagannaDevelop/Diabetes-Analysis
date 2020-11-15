
import os
import shutil
import datetime
import json
from functools import reduce
from typing import List, Tuple, Dict, Union, NoReturn, Optional, Callable, Any

# Local imports :
import decorators
import preproc

@decorators.log_exception_to_mail(subject="Config parsing error")
def parse_config(config_file: Optional[str] = None) -> Dict[str,Union[str,bool]]:
    """
        Parse a config file. 
        
        args:
            config_file: path (absolute or relative) to the config file.
                         Said config file is expected to be in json format.
                         Defaults to "data_import_config.json"

        returns:
            A dict, containing the parsed config.

        Example config file contents : 
        {
            "read": "/Users/juanito/Downloads",
            "write": "/Users/juanito/data",
            "preprocess": true
        }

    """
    config_file = config_file or "data_import_config.json"
    ver_dir = os.path.split(config_file)[0] or os.path.abspath(".")
    if os.path.split(config_file)[1] not in os.listdir(ver_dir):
        raise Exception(f"Config file `{os.path.split(config_file)[1]}` not found in `{ver_dir}`")
    with open(config_file, "r") as f:
        config = json.load(f)
    return config
##


@decorators.log_exception_to_mail(subject="CSV file retrieving error")
def get_csv_files(path: str) -> List[str]:
    """
        Get a list containing all filenames
        found in `path` that contain "csv".
    """
    return [name for name in os.listdir(path) if name[-4:] == ".csv"]
##

@decorators.log_exception_to_mail(subject="CSV list generation error")
def get_csv_abspaths(data: Dict[str,Any]) -> List[str]:
    """
        Get a list containing all absolute paths
        to csv files found in directories within data['config']['read']
        which should be an abspath to a directory itself.
    """
    paths = [
        os.path.join(data['config']['read'], log) for log in data['logs']
    ]
    csv_files_by_directory = {
        path: [os.path.join(path, csv_file) for csv_file in get_csv_files(path)]
        for path in paths
    }
    abs_paths = reduce(lambda x, y: x + y, csv_files_by_directory.values())
    return abs_paths
##


@decorators.log_exception_to_mail(subject="CSV file retrieving error")
def get_directories(path: str) -> List[str]:
    """
        Get a list containing all subdirectories
        found in `path`.
    """
    return [name for name in os.listdir(path) if os.path.isdir(name)]
##

@decorators.time_log("logs/sync.jsonl")
@decorators.log_exception_to_mail(subject="Data sync error")
def main(logfile: Optional[str] = None):
    
    # Default logfile : 
    logfile = logfile or "data_sync_log.jsonl"
    config = parse_config()
    # This data entry is generated when calling the script.
    # Saved afterwards to logfile.
    data = {
        "dateUTC": str(datetime.datetime.utcnow()).split(" ")[0],
        "config": config,
        "logs": os.listdir(config["read"])
    }
    
    # Verify if we have started logging data imports :
    if logfile not in os.listdir("."):
        with open(logfile, "w") as f:
            f.write(f"{json.dumps(data)}\n")
        for csv_file in get_csv_abspaths(data):
            shutil.copy2(csv_file, config["tmp"])
        # print(f"{get_csv_abspaths(data)}")
    
    else:
        with open(logfile, "r") as f:
            last_log = json.loads(f.read().splitlines()[-1])

        if data['logs'] == last_log['logs']:
            print("No new logs")
            #print(f"{get_csv_abspaths(data)}")
        else:
            with open(logfile, "a") as f:
                f.write(f"{json.dumps(data)}\n")
            for csv_file in get_csv_abspaths(data):
                shutil.copy2(csv_file, config["tmp"])

            # print(f"{get_csv_abspaths(data)}")
            print("New logs!")
        #print(f"last log {last_log}")
    
    #print(data)


if __name__ == "__main__":
    main()

