# must load Keras and backend (Theano) before running this code

from os.path import exists
from os import makedirs
import numpy as np
import pandas as pd
from keras.models import model_from_json


def main():

### Setting parameters ###
    nn_type = "SI" # Network type, either "SD (species-dependent)" or "SI (specied-independent)" 
    pred_case = "P37CR" # Case name, "P37CR", "P60CR" or "V60"

    result_path = "./results/"
    data_path   = "./datasets/"
    scale_path  = "./scales/"
    model_path  = "./models/"

    cases = {"P37CR": {"nx": 257, "ny": 1},
             "P60CR": {"nx": 513, "ny": 1},
             "V60":   {"nx": 513, "ny": 257}}
    nx = cases[pred_case]["nx"]
    ny = cases[pred_case]["ny"]

    species = ["H2", "H", "O2", "O", "OH", "HO2", "H2O2", "H2O", "N", "NO2", "NO"] # Fixed for "P37CR", "P60CR" and "V60"
    ns = len(species)

### Making directory ###
    if not exists(result_path + pred_case): makedirs(result_path + pred_case)


### Loading quantities of features ###
    k   = np.fromfile(data_path + pred_case + "/" + pred_case + "_" + "k"   + ".dat", dtype=np.float32) # turbulence kinetic energy
    eps = np.fromfile(data_path + pred_case + "/" + pred_case + "_" + "eps" + ".dat", dtype=np.float32) # dissipation rate of turbulence kinetic energy
    rho = np.fromfile(data_path + pred_case + "/" + pred_case + "_" + "rho" + ".dat", dtype=np.float32) # density
    T   = np.fromfile(data_path + pred_case + "/" + pred_case + "_" + "T"   + ".dat", dtype=np.float32) # temperature

    Y = np.zeros((nx * ny, ns), dtype=np.float32) # mass fractions
    for s in range(ns): 
        Y[:, s] = np.fromfile(data_path + pred_case + "/" + pred_case + "_" + "Y-" + species[s] + ".dat", dtype=np.float32)

    if nn_type == "SI":
        omega0 = np.zeros((nx * ny, ns), dtype=np.float32) # 0th order approximation values of reaction rates
        for s in range(ns): 
            omega0[:, s] = np.fromfile(data_path + pred_case + "/" + pred_case + "_" + "omega0-" + species[s] + ".dat", dtype=np.float32)
        mw  = pd.read_csv(data_path + "thermo-chem/" + "molecular_weight"  + ".csv", delimiter="\s+", index_col=0, dtype={1: np.float32}).values.ravel()
        afr = pd.read_csv(data_path + "thermo-chem/" + "air-to-fuel_ratio" + ".csv", delimiter="\s+", index_col=0, dtype={1: np.float32}).values.ravel()


### Normalizing quantities using corresponding laminar flame values ###
    norm = pd.read_csv(data_path + "thermo-chem/" + "laminar_scales.csv", delimiter="\s+", index_col=0, header=None, dtype={1: np.float32})

    k_normalized   = k   / (norm.loc["sl"].values ** 2)
    eps_normalized = eps / (norm.loc["sl"].values ** 3 / norm.loc["deltath"].values)
    rho_normalized = (rho - norm.loc["rho_min"].values) / (norm.loc["rho_max"].values - norm.loc["rho_min"].values)
    T_normalized   = (T   - norm.loc["T_min"].values)   / (norm.loc["T_max"].values   - norm.loc["T_min"].values)

    Y_index = ["Y-" + s + "_max" for s in species]
    Y_normalized = Y / norm.loc[Y_index].values.ravel()

    omega_index = ["omega-" + s + "_maxabs" for s in species]
    if nn_type == "SI":
        omega0_normalized = omega0 / norm.loc[omega_index].values.ravel()


### Making inputs of features ###
    if nn_type == "SD":
        inputs_mixture = np.stack([k_normalized, eps_normalized, rho_normalized, T_normalized], axis=1)
        inputs = np.hstack([inputs_mixture, Y_normalized])
    if nn_type == "SI":
        inputs = np.stack([np.repeat(k_normalized, ns), np.repeat(eps_normalized, ns), np.repeat(rho_normalized, ns), np.repeat(T_normalized, ns),
                           np.ravel(Y_normalized, order="C"), np.ravel(omega0_normalized, order="C"), np.tile(mw, nx * ny), np.tile(afr, nx * ny)], axis=1)


### Feature scaling ###
    scales = pd.read_csv(scale_path + nn_type + "_scales.csv", delimiter="\s+", index_col=0, dtype={"mean": np.float32, "deviation": np.float32})

    if nn_type == "SD":
        features = ["k", "eps", "rho", "T"] + ["Y-" + s for s in species]
    if nn_type == "SI":
        features = ["k", "eps", "rho", "T", "Y", "omega0", "mw", "smr"]

    inputs_scaled = (inputs - scales["mean"][features].values) / scales["deviation"][features].values


### Outputting prediction results by model ###
    with open(model_path + nn_type + "-ESC_architecture.json", "r") as f:
        model = model_from_json(f.read())
    model.summary()
    model.load_weights(model_path + nn_type + "-ESC_weights" + ".h5")
    outputs_scaled = model.predict(inputs_scaled)


### Inverse scaling outputs ###
    if nn_type == "SD":
        targets = ["omega-" + s for s in species]
    if nn_type == "SI":
        targets = ["omega"]
    outputs_normalized = outputs_scaled * scales["deviation"][targets].values + scales["mean"][targets].values


### Inverse normalizing outputs ###
    if nn_type == "SI":
        outputs_normalized = outputs_normalized.reshape((nx * ny, ns), order="C")
    outputs = outputs_normalized * norm.loc[omega_index].values.ravel()


### Writting outputs ###
    xcoord = np.fromfile(data_path + pred_case + "/" + pred_case + "_" + "x" + ".dat", dtype=np.float32)
    if ny > 1: 
        ycoord = np.fromfile(data_path + pred_case + "/" + pred_case + "_" + "y" + ".dat", dtype=np.float32)

    xcoord_norm = xcoord / norm.loc["deltath"].values
    if ny > 1:
        ycoord_norm = ycoord / norm.loc["deltath"].values

    for s in range(ns):
        if ny == 1:
            output_csv = pd.DataFrame(outputs[:, s], columns=[species[s]], index=xcoord_norm)
        if ny > 1:
            output_csv = pd.DataFrame(outputs[:, s].reshape((nx, ny), order="C"), columns=ycoord_norm, index=xcoord_norm)
        output_csv.to_csv(result_path + pred_case + "/" + nn_type + "-ESC_" + pred_case + "_omega-" + species[s] + ".dat", encoding="utf-8")



if __name__ == "__main__":
    main()
