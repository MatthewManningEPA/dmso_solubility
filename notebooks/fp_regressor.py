import os
import pickle

import dataset_creation
from descriptor_processing import get_api_descriptors


def get_toxprints(smiles, file_path):
    desc_df = get_api_descriptors(smiles, desc_path=file_path, desc_set="toxprints")
    print(desc_df)
    return desc_df


def load_smiles_data():
    stdized_smiles_path = "{}epa_query_stdizer_output.pkl".format(
        os.environ.get("DATA_DIR")
    )
    new_smiles = "{}epa_query_std_smiles.pkl".format(os.environ.get("DATA_DIR"))
    with open(new_smiles, "rb") as f:
        epa_df = pickle.load(f)
    # print(epa_df.columns)
    epa_map = {
        "cid": "DTXCID",
        "sid": "DTXSID",
        "smiles": "SMILES",
        "inchi": "INCHI",
        "inchiKey": "INCHI_KEY",
        "canonicalSmiles": "canonicalSmiles",
    }
    # epa_df.rename(columns=epa_map, inplace=True)
    # epa_df.insert(loc=0, column="SOURCE", value="EPA")
    # epa_df.set_index(keys="INCHI_KEY", drop=True, inplace=True)
    print(epa_df.columns)
    # with open(new_smiles, "wb") as f:
    #    pickle.dump(epa_df, f)
    enamine_df = dataset_creation.get_original_enamine()
    enamine_df.insert(loc=0, column="SOURCE", value="ENAMINE")
    enamine_df.set_index(keys="INCHI_KEY", drop=True, inplace=True)
    print(enamine_df.columns)
    return epa_df, enamine_df


def main():
    epa_toxprints_path = "{}epa_toxprints.pkl".format(os.environ.get("DATA_DIR"))
    enamine_toxprints_path = "{}enamine_toxprints.pkl".format(
        os.environ.get("DATA_DIR")
    )
    with open(enamine_toxprints_path, "rb") as f:
        # pickle.dump(enam_desc, f)
        enamine_toxprints = pickle.load(f)
    print(enamine_toxprints)
    print(type(enamine_toxprints))
    epa_df, enamine_df = load_smiles_data()
    epa_desc = get_toxprints(epa_df["SMILES"], epa_toxprints_path)
    # print(epa_failed)
    with open(epa_toxprints_path, "wb") as f:
        pickle.dump(epa_desc, f)
        # epa_toxprints = pickle.load(f)
    """    
    print(epa_toxprints.head())
    print(epa_toxprints.shape)
    print(epa_toxprints.columns)
    exit()
    """
    enam_desc = get_toxprints(enamine_df["SMILES"], enamine_toxprints_path)
    print(enam_desc)
    with open(enamine_toxprints_path, "wb") as f:
        pickle.dump(enam_desc, f)
        # enamine_toxprints = pickle.dump(enam_desc, f)
    """
    with open(, "wb") as f:
        pickle.dump(enam_info, f)
    """
    # return epa_toxprints, enamine_toxprints


if __name__ == "__main__":
    main()
