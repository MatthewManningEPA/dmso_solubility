import itertools
import os
import pickle

import numpy as np
import pandas as pd
from skfp import fingerprints
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline

import dataset_creation
from descriptor_processing import get_api_descriptors
from FuzzyApplicator import FuzzyApplicator


def train_mlp_regressor(
    expert_predicts, gate_features, labels, layer_sizes, activation="tanh"
):
    gating_mlp = MLPClassifier(
        hidden_layer_sizes=layer_sizes, activation=activation, early_stopping=True
    )
    MLPRegressor().fit()
    gating_steps = [gating_mlp]
    gating_pipeline = Pipeline()
    gating_pipeline.fit()


def get_toxprints(epa_df, enamine_df, file_path):
    # desc_df = get_api_descriptors(smiles, desc_path=file_path, desc_set="toxprints")
    epa_toxprints_path = "{}epa_toxprints.pkl".format(os.environ.get("DATA_DIR"))
    enamine_toxprints_path = "{}enamine_toxprints.pkl".format(
        os.environ.get("DATA_DIR")
    )
    with open(enamine_toxprints_path, "rb") as f:
        # pickle.dump(enam_desc, f)
        enamine_toxprints = pickle.load(f)
    epa_toxprints = get_api_descriptors(
        enamine_df["SMILES"], desc_path=enamine_toxprints_path, desc_set="toxprints"
    )
    enamine_toxprints_toxprints = get_api_descriptors(
        enamine_df["SMILES"], desc_path=enamine_toxprints_path, desc_set="toxprints"
    )
    with open(epa_toxprints_path, "wb") as f:
        epa_toxprints = pickle.load(f)
    with open(enamine_toxprints_path, "wb") as f:
        pickle.dump(enamine_toxprints, f)
        # enamine_toxprints = pickle.dump(enam_desc, f)
    """
    with open(, "wb") as f:
        pickle.dump(enam_info, f)
    """
    return epa_toxprints, enamine_toxprints


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


def get_fingerprints(
    smiles, fp_type, batch_size=1000, fp_path=None, temp_path=None, n_jobs=-2
):
    print("Topo input shape: {}".format(smiles.shape))
    """
    mol_from_smiles = preprocessing.MolFromSmilesTransformer(
        sanitize=True, valid_only=True, n_jobs=n_jobs
    )
    """
    ttfp = fingerprints.TopologicalTorsionFingerprint(
        count=True, n_jobs=n_jobs, batch_size=batch_size
    )
    if not isinstance(smiles, pd.Series):
        smiles = pd.Series(smiles)
    if batch_size is None:
        batch_df = pd.DataFrame(
            ttfp.fit_transform(smiles.to_list()), index=smiles.index
        )
    else:
        fp_batch_list = list()
        invalid_list = list()
        for smiles_tup in itertools.batched(smiles.items(), n=batch_size):
            mols = dict(smiles_tup)
            for smi, m in mols.items():
                if m is None:
                    invalid_list.append(smi)
            if temp_path is not None:
                with open(temp_path, "a") as f:
                    for tup in list(mols.items()):
                        f.writelines(tup)
            [mols.pop(s) for s in invalid_list]
            batch_arr = ttfp.transform([a for a in mols.values()])

            iter_df = pd.DataFrame(batch_arr, index=[a[0] for a in smiles_tup])
            fp_batch_list.append(iter_df)
        batch_df = pd.concat(fp_batch_list)
    assert batch_df.shape[0] == smiles.shape[0]
    if fp_path is not None:
        with open(fp_path, "wb") as f:
            pickle.dump(batch_df, f)
    # fp_df.to_csv(tt_fp_path, mode="w")
    return batch_df


def create_gate(
    expert_preds,
    labels,
    gate_features,
    layers=None,
    activation="tanh",
    random_state=0,
    sample_weight=None,
    verbose=1,
):
    # if gate_features is None:
    #    gate_features = get_topo_torsions(smiles, batch_size=100)
    print("Gate features")
    print(gate_features)
    print("Expert Preds")
    print(expert_preds)
    print(np.shape(expert_preds))
    n_experts = np.shape(expert_preds)[1]
    X = pd.concat(
        (
            expert_preds,
            gate_features.loc[gate_features.index.intersection(expert_preds.index)],
        ),
        axis=1,
    )
    print(X)
    print(np.shape(X))
    gating = FuzzyApplicator(
        n_experts=n_experts,
        layer_sizes=layers,
        activation=activation,
        random_state=random_state,
        verbose=verbose,
    )

    gating.fit(X=X, y=labels, sample_weight=sample_weight)
    return gating


def assemble_models(final_path, base_path):
    model_list = list()
    for modname in os.scandir(base_path):
        if "model" in modname.name and modname.name.endswith(".pkl"):
            print(modname.path)
            with open(str(modname.path), "rb") as f:
                base = pickle.load(f)
                model_list.append(base)
                print("Base model loaded")
                break
    for fname in os.listdir(final_path):
        if "best_model" in fname and fname.endswith(".pkl"):
            print(fname)
            with open("{}{}".format(final_path, fname), "rb") as f:
                frozen = pickle.load(f)
                model_list.append(frozen)
                print("Expert model loaded.")
    return model_list


def get_expert_predict_proba(mod_list, X, pos_label):
    predict_list = list()
    for mod in mod_list:
        preds = pd.Series(mod.predict_proba(X)[:, pos_label], index=X.index)
        print(preds)
        print(np.shape(preds))
        predict_list.append(preds)

    # predict_proba = np.vstack(predict_list)
    predict_proba = pd.concat(predict_list, axis=1)
    return predict_proba


def train_predict(
    model_path,
    train_data,
    test_data,
    layers=None,
    batch_size=None,
    pos_label=1,
    sample_weight=None,
    fp_type="ecfp",
):
    lookup_df = get_smiles_data()

    # train_df = pd.read_pickle("{}prepped_train_df.pkl".format(model_path))
    # test_df = pd.read_pickle("{}prepped_test_df.pkl".format(model_path))
    if layers is None:
        layers = (64, 16)
    # smiles_ser = pd.concat([epa_data["SMILES"].squeeze(), enamine_data["SMILES"].squeeze()]).sort_index()
    # smiles_converter = smiles_ser.reset_index().set_index(keys="SMILES", drop=True)
    if "torsion" in fp_type:
        fp_path = "{}topo_torsion.pkl".format(os.environ.get("DATA_DIR"))

    elif "ecfp" in fp_type:
        fp_path = "{}ecfp_4.pkl".format(os.environ.get("DATA_DIR"))
    if os.path.isfile(fp_path):
        with open(fp_path, "rb") as f:
            fp_df = pickle.load(f, encoding="utf-8")
        # fp_df = pd.read_csv(fp_path)
    else:
        fp_df = get_fingerprints(smiles_ser, None, 1000, fp_path).loc[smiles_ser.index]
    """    
    short_index = fp_df.index.copy().map(lambda x: x.split("-")[0])
    fp_df.index = short_index
    # index_mapper = dict([*zip(fp_df.index, short_index)])
    train_data[0].index = train_data[0].index.map(lambda x: x.split("-")[0])
    train_data[1].index = train_data[1].index.map(lambda x: x.split("-")[0])
    test_data[0].index = test_data[0].index.map(lambda x: x.split("-")[0])
    test_data[1].index = test_data[1].index.map(lambda x: x.split("-")[0])
    train_tup = (
        train_data[0].copy().loc[train_data[0].index.intersection(fp_df.index)],
        train_data[1].copy()[train_data[1].index.intersection(fp_df.index)],
    )
    test_tup = (
        test_data[0].copy().loc[test_data[0].index.intersection(fp_df.index)],
        test_data[1].copy()[test_data[1].index.intersection(fp_df.index)],
    )
    """
    print("Torsion DF: {}:".format(fp_df.shape))
    print(fp_df)
    experts = assemble_models(
        final_path="{}final/".format(model_path),
        base_path="{}base_model/".format(model_path),
    )
    assert len(experts) > 1
    expert_probs = get_expert_predict_proba(
        experts, train_data[0].loc[smiles_ser.index], pos_label=pos_label
    )
    print("Expert probs")
    print(expert_probs.shape)
    print(expert_probs)
    fuzzy_gate = create_gate(
        expert_preds=expert_probs,
        labels=train_data[1][smiles_ser.index],
        gate_features=fp_df.loc[smiles_ser.index],
        layers=layers,
        sample_weight=sample_weight,
    )
    with open("{}gating_classifier.pkl".format(model_path), "wb") as f:
        pickle.dump(fuzzy_gate, f)
    return fuzzy_gate


def get_smiles_data():
    lookup_path = "{}enamine_chemtrack_consolidated.pkl".format(
        os.environ.get("DATA_DIR")
    )
    lookup_df = pd.read_pickle(lookup_path)
    return lookup_df


def main():
    pass
    """    
    print(epa_toxprints.head())
    print(epa_toxprints.shape)
    print(epa_toxprints.columns)
    exit()
    """


if __name__ == "__main__":
    main()
