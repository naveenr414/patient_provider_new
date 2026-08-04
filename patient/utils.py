"""Small shared helpers: JSON I/O, noisy-utility sampling, and the
omniscient (perfect-information) linear program from Section 4.1 of the paper."""
import json
import os

import numpy as np
import gurobipy as gp
from gurobipy import GRB


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, cls=NumpyEncoder)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def create_random_weights(theta_hat, epsilon, rng=None):
    """Sample one noisy scenario theta = theta_hat + delta, ||delta||_inf <= epsilon
    (Section 3.1). Used both to simulate the true realized theta and, inside SAM,
    to draw scenarios theta^(s) for Algorithm 1."""
    rng = rng or np.random
    noisy = theta_hat + rng.uniform(-epsilon, epsilon, theta_hat.shape)
    return np.clip(noisy, 0.0, 1.0)


def solve_omniscient_lp(theta, capacities):
    """Perfect-information bipartite assignment LP (Eq. 1 / Section 4.1): the
    omniscient benchmark used to normalize utility in every figure.

    Arguments:
        theta: N x (M+1) utility matrix, last column is the exit option
        capacities: length-M array of provider capacities

    Returns: N x M 0/1 assignment matrix (LP relaxation is integral on this
    transportation polytope, so no rounding is needed)."""
    N, M_plus_1 = theta.shape
    M = M_plus_1 - 1

    m = gp.Model("omniscient_lp")
    m.setParam("OutputFlag", 0)
    x = m.addVars(N, M, lb=0.0, ub=1.0, name="x")

    obj = gp.quicksum(theta[i, M] for i in range(N))
    obj += gp.quicksum(
        (theta[i, j] - theta[i, M]) * x[i, j] for i in range(N) for j in range(M)
    )
    m.setObjective(obj, GRB.MAXIMIZE)

    for j in range(M):
        m.addConstr(gp.quicksum(x[i, j] for i in range(N)) <= capacities[j])
    for i in range(N):
        m.addConstr(gp.quicksum(x[i, j] for j in range(M)) <= 1)

    m.optimize()

    assignment = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            if x[i, j].X > 0.5:
                assignment[i, j] = 1
    return assignment


def parse_comorbidity_data(lines):
    """Parse the NIH Table-of-Comorbidities text files (data/{cardio,gastro,...}.txt)
    into per-age-group prevalence rates. Format: tab-separated 'count (pct)' cells,
    one row per condition, one column per age group; we combine conditions via
    1 - prod(1 - p) to get "has any of these" prevalence per age group."""
    rates = []
    for line in lines:
        if not line.strip():
            continue
        row = [float(cell.split("(")[1].replace(")", "")) for cell in line.split("\t")]
        rates.append(row)
    return 1 - np.prod(1 - np.array(rates) / 100, axis=0)


def get_age_group(age):
    """Bucket a patient age into the four groups used by the NIH comorbidity table."""
    if 18 <= age <= 53:
        return 0
    elif 54 <= age <= 64:
        return 1
    elif 65 <= age <= 73:
        return 2
    return 3


def get_zip5(z):
    """Normalize a ZIP-like value to a 5-digit string: truncate ZIP+4 codes
    (e.g. Medicare's 9-digit 'ZIP Code' column) to the first 5 digits, and
    zero-pad short codes (e.g. CT zipcode-population data stored as bare ints
    like 6001) up to 5 digits."""
    z = str(z)
    if len(z) >= 5:
        return z[:5]
    return "0" * (5 - len(z)) + z
