
import os
import datetime
import json
import decorators
from typing import List, Tuple, Dict, Union, NoReturn, Optional, Callable, Any

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
        raise Exception(f"Config file {os.path.split(config_file)[1]} not found in {ver_dir}")
    with open(config_file, "r") as f:
        config = json.load(f)
    return config


def main():
    config = parse_config(config_file="lol")
    data = {
        "dateUTC": str(datetime.datetime.utcnow()).split(" ")[0],
        "config": config
    }
    print(data)


if __name__ == "__main__":
    main()

