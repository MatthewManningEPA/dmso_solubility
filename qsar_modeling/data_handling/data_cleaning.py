import pandas as pd
from sklearn.utils import check_X_y as checker

from data.element_lists import DISALLOWED, WITH_SALTS


def duplicate_connect(inchi_key_list):
    first_blocks, clashes = dict(), list()
    for k in inchi_key_list:
        f = k.split(sep="-")[0]
        if f in first_blocks.keys():
            first_blocks[f].append(k)
        else:
            first_blocks[f] = list([k])
    for k, v in first_blocks.items():
        if len(v) > 1:
            clashes.append(v)
    return clashes


def qsar_readiness(smiles_list, allow_salts=True):
    not_ready, ready = list(), list()
    if allow_salts:
        forbidden = WITH_SALTS
    else:
        forbidden = DISALLOWED
    for smi in smiles_list:
        if not isinstance(smi, str):
            print(smi)
            raise ValueError
        for e in forbidden:
            if e in smi:
                not_ready.append(smi)
                break
        ready.append(smi)
    return ready, not_ready


def remove_duplicate_idx(df):
    # Reliably removes duplicated rows in a DataFrame.
    new_dict = dict()
    for k, v in df.items():
        if k in new_dict.keys():
            continue
        else:
            new_dict[k] = v
    if type(df) is pd.DataFrame:
        new_df = pd.DataFrame.from_dict(new_dict)
    elif type(df) is pd.Series:
        new_df = pd.Series(data=new_dict, name=df.name)
    return new_df


def rename_duplicate_features(feature_df):
    # Renames unique-valued, identically-named features in a DataFrame.
    # feature_df = feature_df.T.drop_duplicates().T
    while feature_df.columns[feature_df.columns.duplicated()].size > 0:
        dup_col = feature_df.columns[feature_df.columns.duplicated(keep="first")]
        feature_df.columns = feature_df.columns.where(
            ~feature_df.columns.duplicated(keep="first"),
            feature_df.columns[feature_df.columns.duplicated()] + "i",
        )
    assert feature_df.columns[feature_df.columns.duplicated()].size == 0
    return feature_df


def check_inchi_only(inchi_keys):
    # Check if a list of strings contains any non-INCHI key entries.
    correct_keys, bad_keys = list(), list()
    for k in inchi_keys:
        ksplit = k.split("-")
        if (
            len(ksplit) == 3
            and len(ksplit[0]) == 14
            and len(ksplit[1]) == 10
            and len(ksplit[2]) == 1
        ):
            correct_keys.append(k)
        else:
            bad_keys.append(k)
    return correct_keys, bad_keys


def clean_and_check(feature_df, labels, y_dtype=None, var_thresh=0, verbose=False):
    X_df = feature_df.copy()
    X_df = rename_duplicate_features(X_df)
    valid_idx, invalid_idx = check_inchi_only(X_df.index)
    if len(invalid_idx) > 0:
        raise UserWarning
        print("INCHI keys are not valid: {}".format(invalid_idx))
    zero_var_cols = X_df[X_df.nunique(axis=1) < var_thresh].columns
    if verbose:
        print(feature_df.shape)
        print(
            "{} features with less than {} variance removed.".format(
                feature_df.shape[1] - zero_var_cols.size, var_thresh
            )
        )
    if zero_var_cols.size == 0:
        X_df.drop(columns=zero_var_cols, inplace=True)
    feature_arr, labels_arr = checker(X_df, y=y_ser)
    Xt = pd.DataFrame(data=feature_arr, index=X_df.index, columns=X_df.columns)
    yt = pd.Series(data=labels_arr, index=y_ser.index)
    assert not Xt.empty
    assert not yt.empty
    return Xt, yt
