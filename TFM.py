import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

from methods_script.toroidal_filament.parameters import all_arrays

def determine_unique_probes(probe_set:list[str]) -> list[str]:
    """determine all unique probe numbers in probe_set"""
    unique_probes = np.unique(" ".join(probe_set).split(" "))
    return unique_probes

def check_missing_files(shot_path:str, unique_probes:list[str]) -> None:
    """
    raise FileNotFoundError if shot_path does not exist or missing required txt files with
    """

    #check if directory exist
    if not os.path.isdir(shot_path):
        raise FileNotFoundError(f"directory {shot_path} does not exist")

    #list out all existing files in shot_path
    existing_files = set(os.listdir(shot_path))

    #define all required files
    required_files = set(
        ["IP1.txt", "IT1.txt", "IOH1.txt", "IV2.txt"] + 
        ["GBP" + i + "T.txt" for i in unique_probes]
    )

    #determine missing files
    missing = required_files - existing_files

    #if missing is not empty raise Error
    if bool(missing):
        # sort missing by characters for easy reading
        missing_lst = list(missing)
        missing_lst.sort()
        raise FileNotFoundError(f"{missing_lst} are missing from {shot_path}")

    return


def TFM_main(shot_path: str,use_probe_set: list[str]) -> tuple[pd.DataFrame]:
    """
    Calculate plasma column position displacement with the Toroidal Filament Model
    :param shot_path: path to data directory containing
                      "IP1.txt", "IT1.txt", "IOH1.txt", "IV2.txt", and
                      "GBPXT.txt" for all X as unique probe number in probe_set
    :param use_probe_set: set of magnetic probes to use for calculation
                      (set number must exist in all_arrays)
    :return: dataframes of centroid displacement calculated from all probe_set
             along radial and vertical directions
    """
    #determine all unique magnetic probes to use
    unique_probes = determine_unique_probes(use_probe_set)

    #raise FileNotFoundError if shot_path does not exist or
    #missing required txt files
    check_missing_files(shot_path, unique_probes)

    return

if __name__ == "__main__":
    use_probes = ["1 4 7 10", "2 4 8 10"]

    import pytest

    def assert_TFM_error(shot_path, use_probes_set):
        with pytest.raises(FileNotFoundError) as errinfo:
            TFM_main(shot_path,use_probes)
        print(errinfo)

    assert TFM_main(os.path.join("resources","fullShotData","1641"),use_probes) == None
    assert_TFM_error(os.path.join("resources","fullShotData","1671"),use_probes)
    assert_TFM_error(os.path.join("resources","fullShotData","test_direc"),use_probes)

