import itertools
import json
import os

import pandas as pd
import requests

import DescriptorRequestor
import padel_categorization


# import qsar_readiness
# from qsar_readiness import QSAR_COLUMNS


# TODO: Calculate and plot descriptive stats
# TODO: Filter compounds by MW, peptides, etc.
# TODO: Eliminate compounds/features with higher NaN values
# TODO: Feature agglomeration
# TODO: Feature selection


def get_api_descriptors(smiles_ser, desc_path, desc_set="padel"):
    # Wrapper function that gets descriptors for many compounds.
    desc_list = list()
    inchi_dict = dict()
    for v, k in smiles_ser.items():
        print(v, k)
        inchi_dict[str(k)] = str(v)
    print([inchi_dict.items()][0])
    desc_grabber = DescriptorRequestor.DescriptorGrabber(
        desc_set=desc_set, timeout=(30, 600)
    )
    with open(desc_path, "a") as f:
        for desc, smile in desc_grabber.bulk_epa_call(smiles_ser.tolist()):
            print(smile, inchi_dict[smile])
            if not isinstance(desc, dict):
                break
            else:
                desc["FOREIGN_KEY"] = inchi_dict[smile]
                desc_list.append(desc)
            f.write(json.dumps(desc))
    return pd.DataFrame.from_records(desc_list)


def map_cols(df, original, new):
    qsar_map = dict(zip(df[original], df[new]))
    return qsar_map


def extract_cached_desc(qsar_ready_df, cached_df):
    # Find SMILES_QSAR column in both dfs, check for union (recheck for different smiles column), return descriptor df
    upper_cache_cols = [
        x
        for x in cached_df.columns.tolist()
        if ("QSAR" in x.upper() or "SMILE" in x.upper())
    ]
    last_num = 0
    best_match, best_col, qsar_col = None, None, None
    for col, qsar_col in itertools.product(
        upper_cache_cols, ["SMILES_QSAR", "SMILES_CANONICAL"]
    ):
        cache_smiles = pd.Index(cached_df[col].squeeze().tolist())
        qsar_smiles = pd.Index(qsar_ready_df[qsar_col].squeeze().tolist())

        # Find QSAR column, if different from SMILES_QSAR, convert column to SMILES_QSAR

        matches = qsar_smiles.intersection(cache_smiles)
        if len(matches.tolist()) > last_num:
            last_num = len(matches.tolist())
            best_col = (col, qsar_col)
            best_match = matches
    smiles_matches = pd.Index(
        qsar_ready_df["SMILES_QSAR"].squeeze().tolist()
    ).intersection(cached_df.index)
    canon_matches = pd.Index(
        qsar_ready_df["SMILES_CANONICAL"].squeeze().tolist()
    ).intersection(cached_df.index)
    if last_num == 0:
        return pd.DataFrame([]), None
    elif len(smiles_matches) > last_num and len(canon_matches):
        return (
            cached_df.loc[smiles_matches].rename(
                index=map_cols(qsar_ready_df, "SMILES_QSAR", "INCHI_KEY"), inplace=True
            ),
            "SMILES_QSAR",
        )
    elif len(canon_matches) > last_num:
        return (
            cached_df.loc[canon_matches].rename(
                index=map_cols(qsar_ready_df, "SMILES_CANONICAL", "INCHI_KEY"),
                inplace=True,
            ),
            "SMILES_CANONICAL",
        )
    else:
        return (
            cached_df.loc[cached_df[best_col[0]].isin(best_match.tolist())].rename(
                index=map_cols(qsar_ready_df, best_col[1], "FOREIGN_KEY"), inplace=True
            ),
            best_col[1],
        )


def get_api_desc(desc_path, id_ser, d_set, verbose=False, *args, **kwargs):
    # Hits API to obtain descriptor (d_set).
    last_ind = 0
    inchi_dict = id_ser.to_dict()
    desc_dict = dict()
    ser_list = list()
    grabber = DescriptorRequestor.DescriptorGrabber(desc_set=d_set, *args, **kwargs)
    if os.path.isfile(desc_path):
        desc_mode = "a"
    else:
        desc_mode = "w"
    with open(desc_path, desc_mode) as fo:
        for response, api_input in grabber.bulk_epa_call(id_ser.tolist()[last_ind:]):
            if isinstance(response, dict) and "descriptors" in response.keys():
                if verbose:
                    print("API Input: {}".format(api_input))
                    print("Response: {}".format(response.items()))
                ikey = id_ser[id_ser == api_input].index[0]
                desc_dict[ikey] = response["descriptors"]
                fo.write("{}\n".format("\t".join([str(s) for s in response.values()])))
                temp_srs = pd.Series(
                    data=response["descriptors"], name=inchi_dict[ikey]
                )
                # temp_srs = pd.Series(data=response['descriptors'], index=desc_names, name=inchi_dict[api_input])
                ser_list.append(temp_srs)
            else:
                print("Featurizer failed for: ", api_input, response)
                fo.write("{}\n".format("\t".join([str(s) for s in response.values()])))
    # desc_df = pd.concat(ser_list, axis=1).T
    desc_df = pd.DataFrame.from_dict(desc_dict, orient="index")
    return desc_df


def main(
    data_path=None,
    qsar_df=None,
    desc_cache=None,
    qsar_ready_name=None,
    cache_path=None,
    desc_set=None,
    intermetals=False,
    salt_filter=False,
):
    """
    Runs compounds from data_path through EPA internal descriptor calculator.
    :param data_path:
    :param cache_path:
    :param qsar_ready_name:
    :param desc_cache: pd.DataFrame,
    :param qsar_df: pd.DataFrame, Contains QSAR Standardizer output
    """
    if type(desc_cache) is not pd.DataFrame and cache_path:
        # print(desc_cache)
        desc_cache = pd.read_pickle("{}{}.pkl".format(cache_path, qsar_ready_name))
    # if not type(qsar_df) is not pd.DataFrame and qsar_ready_name:
    #    print(qsar_df)
    #    qsar_df = pd.read_pickle('{}{}.pkl'.format(data_path, qsar_ready_name))
    """
    qsar_df.rename(mapper=QSAR_COLUMNS, inplace=True)
    salts, intermetallics = qsar_readiness.salts_and_intermetallics(
        qsar_df["SMILES_QSAR"].tolist(), intermetallics=intermetals
    )
    if salt_filter and len(salts) > 0:
        salt_df = qsar_df[qsar_df["SMILES_QSAR"].isin(salts)]
        salt_df.to_pickle("{}SALTS_{}.pkl".format(data_path, qsar_ready_name))
        qsar_df.drop(index=salt_df.index, inplace=True)
    if intermetallics and len(intermetallics) > 0:
        intermetallics_df = qsar_df[qsar_df["SMILES_QSAR"].isin(intermetallics)]
        intermetallics_df.to_pickle(
            "{}INTERMETALLICS_{}.pkl".format(data_path, qsar_ready_name)
        )
        qsar_df.drop(index=intermetallics_df.index, inplace=True)
    """
    if (
        desc_cache is not None
        and type(desc_cache) is pd.DataFrame
        and not desc_cache.empty
    ):
        cached_values, qsar_col_name = extract_cached_desc(qsar_df, desc_cache)
        missing_desc = qsar_df.loc[qsar_df.index.difference(cached_values.index)]
    else:
        missing_desc = qsar_df
    print(missing_desc.shape)
    api_desc = get_api_desc(
        "{}{}/{}_{}.csv".format(data_path, desc_set, desc_set.upper(), qsar_ready_name),
        missing_desc["SMILES_QSAR"].squeeze(),
        desc_set,
    )
    if desc_cache and type(desc_cache) is pd.DataFrame and desc_cache.empty:
        all_desc = pd.concat(
            [cached_values, api_desc], axis=1, verify_integrity=True, sort=True
        )
    else:
        all_desc = api_desc
    return all_desc


def epa_chem_lookup_api(comlist, batch_size=100, type="sid", prefix="DTXSID"):
    # stdizer = DescriptorRequestor.QsarStdizer(input_type="dtxsid")
    api_url = "https://ccte-cced-cheminformatics.epa.gov/api/search/download/properties"
    response_list = list()
    auth_header = {"x-api-key": os.environ.get("INTERNAL_KEY")}
    with requests.session() as r:
        req = requests.Request(method="POST", url=api_url)
        for c_batch in itertools.batched(comlist, n=batch_size):
            req_json = {"ids": [], "format": type}
            if prefix is not None:
                c_batch = [
                    c.removeprefix(prefix) for c in c_batch if prefix is not None
                ]
            for c in c_batch:
                req_json["ids"].append({"id": c, "sim": 0})
            response = r.request(
                method="POST", url=api_url, json=req_json, headers=auth_header
            )
            response_list.extend(response.json())
    response_df = pd.json_normalize(response_list)
    return response_df


def get_standardizer(comlist, input_type="smiles"):
    """

    Parameters
    ----------
    comlist: Iterable, list of inputs (default: SMILES)
    input_type: str, keyword for API call input

    Returns
    -------
    api_response: DataFrame, normalized output with MOL field removed (for compability)
    failed_df: DataFrame: response for compounds that did not return a valid Standardizer response
    """
    api_url = "https://ccte-cced-cheminformatics.epa.gov/api/stdizer"
    response_list, failed_list = list(), list()
    auth_header = {"x-api-key": os.environ.get("INTERNAL_KEY")}
    with requests.session() as r:
        for c in comlist:
            params = {"workflow": "qsar-ready", input_type: c}
            response = r.request(
                method="GET", url=api_url, params=params, headers=auth_header
            )
            if isinstance(response.json(), list) and len(response.json()) > 0:
                response_list.append(response.json()[0])
            else:  # isinstance(response, (list, str)):
                failed_list.append(response.content)
    clean_list = [x for x in response_list if "sid" in x.keys()]
    unregistered_list = [
        x for x in response_list if "sid" not in x.keys() and "smiles" in x.keys()
    ]
    # failed_list = [x for x in response_list if "smiles" not in x.keys()]
    clean_df = pd.json_normalize(clean_list)
    unregistered_df = pd.json_normalize(unregistered_list)
    response_df = pd.concat([clean_df, unregistered_df], ignore_index=True)
    api_response = response_df.drop(columns="mol")
    failed_df = pd.json_normalize(failed_list)
    return api_response, failed_df


def get_epa_descriptors(
    smi_list, desc_type="padel", input_format="smiles", long_names=False
):
    """
    Calls internal API using key and returns DFs of descriptors, full API response, and failed calls.
    Indices are API provided InChI keys
    API key is stored in os.environ.get("INTERNAL_KEY")

    Parameters
    ----------
    smi_list: Iterable[str], inputs for API calls
    desc_type: str, Descriptor set name as API keyword
    input_format: str, default: SMILES
    long_names: If "padel", whether to use short names or long or DF columns.

    Returns
    -------
    desc_df: DataFrame, sklearn-compatible input of descriptors only.
    info_df: DataFrame, normalized output from API call, without descriptors
    failed_list: list, inputs that did not return a valid result from API
    """
    if desc_type != "padel":
        raise NotImplementedError
    """
    API schema
    {
        "chemicals": ["string"],
        "options": {
            "headers": true,
            "timeout": 0,
            "compute2D": true,
            "compute3D": true,
            "computeFingerprints": true,
            "descriptorTypes": ["string"],
            "removeSalt": true,
            "standardizeNitro": true,
            "standardizeTautomers": true,
        },
    }
    """
    # TODO: Add original SMILES/identifier to info_df to link original data and descriptor data through info_df.
    api_url = "https://ccte-cced-cheminformatics.epa.gov/api/padel"
    info_list, desc_dict, failed_list = list(), dict(), list()
    with requests.session() as r:
        auth_header = {"x-api-key": os.environ.get("INTERNAL_KEY")}
        req = requests.Request(method="GET", url=api_url, headers=auth_header).prepare()
        for c in smi_list:
            params = {input_format: c, "type": desc_type}
            req.prepare_url(url=api_url, params=params)
            response = r.send(req)
            if response.status_code == 200 and len(response.json()) > 0:
                single_info_dict = dict(
                    [
                        (k, v)
                        for k, v in response.json()["chemicals"][0].items()
                        if k != "descriptors"
                    ]
                )
                single_info_dict.update([("SMILES_QSAR", c)])
                info_list.append(single_info_dict)
                desc_dict[response.json()["chemicals"][0]["inchiKey"]] = (
                    response.json()["chemicals"][0]["descriptors"]
                )
            else:
                failed_list.append(response.content)
    if desc_type == "padel":
        if long_names:
            padel_names = padel_categorization.get_full_padel_names()
        else:
            padel_names = padel_categorization.get_short_padel_names()
        desc_df = pd.DataFrame.from_dict(
            data=desc_dict, orient="index", columns=padel_names
        )
    else:
        try:
            desc_df = pd.DataFrame.from_dict(data=desc_dict)
            print(desc_df)
        except:
            desc_df = desc_dict
    info_df = pd.json_normalize(info_list)
    info_df.set_index(keys="inchiKey", inplace=True)
    # TODO: Is this drop needed or even valid?
    # info_df.drop(columns=padel_names, inplace=True, errors="ignore")

    return desc_df, info_df, failed_list
