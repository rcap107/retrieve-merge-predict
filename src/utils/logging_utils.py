"""
The paths used by the logger can be provided by the user, or can use the default values used here. 
"""

import pickle
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
from pathlib import Path
import os
import json

import pandas as pd
from abc import abstractmethod, ABC
# Setting the default path for all files. Assumes that the dir tree is already built. 
RESULTS_PATH = Path("results").resolve()
JSON_PATH = Path(RESULTS_PATH, "json")
# The run_id file should contain only a number, which is incremented automatically
# at the start of each run.
RUN_ID_PATH = Path("results/run_id").resolve()


class Logger:
    def __init__(
        self,
        file_path=None,
        run_id_path=RUN_ID_PATH,
        results_path=RESULTS_PATH,
    ):

        # All paths below are normally kept as default, but can be provided by 
        # the user. 
        self.run_id_path = Path(run_id_path)
        self.run_name = None
        self.results_path = Path(results_path)

        self.run_id = self.find_latest_run_id()
        # If no pre-existing path is provided, create a new empty logger file. 
        if file_path is None:
            self.obj = {}
            self.obj["run_id"] = self.run_id
            self.obj["status"] = "SUCCESS"
            self.obj["timestamps"] = {}
            self.obj["durations"] = {}

            self.obj["parameters"] = {}

            # Losses and actual imputation results
            self.obj["results"] = {}
            # Statistics measured on the given dataset (% missing values etc)
            self.obj["statistics"] = {}

            self.add_time("logger_creation_time")

            # Ensure that the required folders exist
            os.makedirs(self.results_path, exist_ok=True)
        else:
            self.obj = pickle.load(open(file_path, "rb"))
            self.run_id = self.obj["run_id"]

    def find_latest_run_id(self):
        """Utility function for opening the run_id file, checking for errors and 
        incrementing it by one at the start of a run.

        Raises:
            ValueError: Raise ValueError if the read run_id is not a positive integer.

        Returns:
            int: The new (incremented) run_id.
        """
        if self.run_id_path.exists():
            with open(self.run_id_path, "r") as fp:
                last_run_id = fp.read().strip()
                try:
                    run_id = int(last_run_id) + 1
                except ValueError:
                    raise ValueError(
                        f"Run ID {last_run_id} is not a positive integer. "
                    )
                if run_id < 0:
                    raise ValueError(f"Run ID {run_id} is not a positive integer. ")
            with open(self.run_id_path, "w") as fp:
                fp.write(f"{run_id}")
        else:
            run_id = 0
            with open(self.run_id_path, "w") as fp:
                fp.write(f"{run_id}")
        return run_id

    def add_dict(self, obj_name, obj):
        """
        Add a new dictionary to the logger. `obj_name` is the key that will be used to store the object.

        :param obj_name: A string that will be used as key.
        :param obj: The dictionary to be added.
        :return:
        """
        self.obj[obj_name] = dict()
        self.obj[obj_name].update(obj)

    def update_dict(self, obj_name, obj):
        """Update the given object with a new dictionary. 

        Args:
            obj_name (str): Label of the object to update.
            obj (dict): Dictionary to be added to the given obj. 
        """
        self.obj[obj_name].update(obj)

    def add_value(self, obj_name, key, value):
        """Updating a single value in a given object. 

        Args:
            obj_name (str): Label of the object to update.
            key (_type_): Key of the object to update.
            value (_type_): Value of the object to update.
        """
        self.obj[obj_name][key] = value

    def add_run_name(self, name):
        """Generic function for setting a run name provided by the user. 

        Args:
            name (str): Name to be assigned to this run. 
        """
        self.obj["parameters"]["run_name"] = name
        self.run_name = name

    def get_value(self, obj_name, key):
        """Retrieve a single value from one of the dictionaries. 

        Args:
            obj_name (str): Label of the object to query.
            key (_type_): Dict key to use to retrieve the value.

        Returns:
            _type_: Retrieved value.
        """
        return self.obj[obj_name][key]

    def add_time(self, label):
        """Add a new timestamp starting __now__, with the given label. 

        Args:
            label (str): Label to assign to the timestamp.
        """
        self.obj["timestamps"][label] = dt.datetime.now()

    def get_time(self, label):
        """Retrieve a time according to the given label.    

        Args:
            label (str): Label of the timestamp to be retrieved. 
        Returns:
            _type_: Retrieved timestamp.
        """
        return self.obj["timestamp"][label]

    def add_duration(self, label_start, label_end, label_duration):
        """Create a new duration as timedelta. The timedelta is computed on the 
        basis of the given start and end labels, and is assigned the given label.

        Args:
            label_start (str): Label of the timestamp to be used as start.
            label_end (str): Label of the timestamp to be used as end.
            label_duration (str): Label of the new timedelta object. 
        """        
        assert label_start in self.obj["timestamps"]
        assert label_end in self.obj["timestamps"]
        
        self.obj["durations"][label_duration] = (
            self.obj["timestamps"][label_end] - self.obj["timestamps"][label_start]
        ).total_seconds()

    def set_run_status(self, status):
        # self.status = status
        self.obj["status"] = status


    def save_logger(self, file_path=None):
        """Save log object in a specific file_path, if provided. Alternatively,
        save the log object in a default location.

        Args:
            file_path (str, optional): Path where to save the log file. Defaults to None.
        """
        if file_path:
            if not osp.exists(file_path):
                raise ValueError(f"File {file_path} does not exist.")
            pickle.dump(self.obj, open(file_path, "wb"))
        else:
            file_path = osp.join(self.results_path, f"run_{self.run_id}.pkl")
            pickle.dump(self.obj, open(file_path, "wb"))

    def __getitem__(self, item):
        return self.obj[item]

    @abstractmethod
    def pprint(self):
        """Abstract class, this should be implemented by the user with the proper
        format for the task at hand.
        """
        pass    
    
    def print_selected(self, selected_dict):
        pass

    def get_all_statistics(self):
        output_dict = {}
        for key, value in self.obj.items():
            if isinstance(value, dict):
                output_dict.update(value)
            else:
                output_dict[key] = value
        return output_dict


class OldLogger(Logger):
    
    def __init__(self, file_path=None, run_id_path=RUN_ID_PATH, results_path=RESULTS_PATH):
        super().__init__(file_path, run_id_path, results_path)
        
    
    def add_run_name(self):
        """Set the default run name, based on the name of the dirty dataset (note that the name includes the error 
        fraction in the given dataset.)
        """
        basename = osp.basename(self.obj["parameters"]["dirty_dataset"])
        name, ext = osp.splitext(basename)
        self.obj["parameters"]["run_name"] = name
        self.run_name = name

    
    def pprint(self):
        pass
    
    def print_summary(self):
        """Print on screen a summary of this run, with the main parameters. 
        """
        print(f"Run ID:{self.run_id}")
        print(f"Dataset: {self.obj['parameters']['dirty_dataset']}"),
        print(f"Training columns: {self.obj['parameters']['training_columns']}")
        print(f"Total epochs: {self.obj['parameters']['epochs']}")
        print(f"Architecture: {self.obj['parameters']['architecture']}")
        print(f"Loss function: {self.obj['parameters']['loss']}")
        print(f"Node features: {self.obj['statistics']['node_features']}")


def logging(parameters, results):
    logger = Logger()
    logger.add_dict("parameters", parameters)
    logger.add_dict("results", results)
