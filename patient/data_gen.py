"""Utility-matrix generators for the paper's experiments.

`semi_synthetic_theta` builds the Medicare/CT-calibrated environment used for
the main-body results (Section 6, EC.1: geographic proximity + comorbidity
concordance). `uniform_theta` / `normal_theta` / `latent_theta` are the three
distributions compared in EC.2.3.
"""
import functools
from pathlib import Path

import numpy as np
import pandas as pd

from patient.utils import parse_comorbidity_data, get_age_group, get_zip5

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

COMORBIDITIES = ["cardio", "gastro", "neuro", "substance", "onco"]
SPECIALTY_GROUPS = {
    "cardio": [
        "CARDIOVASCULAR DISEASE (CARDIOLOGY)",
        "ADVANCED HEART FAILURE AND TRANSPLANT CARDIOLOGY",
        "INTERVENTIONAL CARDIOLOGY",
        "CARDIAC ELECTROPHYSIOLOGY",
        "PERIPHERAL VASCULAR DISEASE",
        "CRITICAL CARE (INTENSIVISTS)",
    ],
    "gastro": ["NEPHROLOGY", "ENDOCRINOLOGY"],
    "neuro": ["CRITICAL CARE (INTENSIVISTS)", "GERIATRIC MEDICINE", "SLEEP MEDICINE"],
    "substance": ["ADDICTION MEDICINE"],
    "onco": ["CRITICAL CARE (INTENSIVISTS)", "HEMATOLOGY", "MEDICAL ONCOLOGY"],
}


@functools.lru_cache(maxsize=1)
def _comorbidity_rates():
    return {
        c: parse_comorbidity_data(open(DATA_DIR / f"{c}.txt").read().split("\n"))
        for c in COMORBIDITIES
    }


@functools.lru_cache(maxsize=1)
def _medicare_ct_providers():
    medicare = pd.read_csv(DATA_DIR / "medicare_data.csv", dtype={"ZIP Code": str})
    ct = medicare[medicare["State"] == "CT"]
    ct = ct[
        ct["sec_spec_all"].str.contains(
            "INTERNAL MEDICINE|GENERAL PRACTICE|FAMILY", case=False, na=False
        )
    ]
    for key, specialties in SPECIALTY_GROUPS.items():
        ct[key] = ct["sec_spec_all"].apply(lambda x: any(s in x for s in specialties))
    return ct


@functools.lru_cache(maxsize=1)
def _ct_zip_population():
    zdata = pd.read_csv(DATA_DIR / "connecticut_zipcode.csv")
    zipcodes = [get_zip5(z) for z in zdata["Zipcode"]]
    population = [int(str(p).replace(",", "")) for p in zdata["Population"]]
    probs = np.array(population) / sum(population)
    return zipcodes, probs


@functools.lru_cache(maxsize=1)
def _ct_age_distribution():
    age_distro = pd.read_csv(DATA_DIR / "ct_age.csv")
    ages, buckets = [], []
    for lo in range(0, 100, 5):
        mid = lo + 2.5
        if mid < 18:
            continue
        ages.append(mid)
        buckets.append(
            age_distro[
                (age_distro["Year"] == 2022) & (age_distro["ID Age"] == lo)
            ]["Total Population"].sum()
        )
    buckets = np.array(buckets, dtype=float)
    buckets /= buckets.sum()
    return ages, buckets


@functools.lru_cache(maxsize=1)
def _zip_distances():
    import json

    return json.load(open(DATA_DIR / "ct_zipcode_distance.json"))


def semi_synthetic_theta(num_patients, num_providers, average_distance=20.2, omega=0.5,
                          alpha=0.5, clip_distance_term=True, seed=None):
    """Semi-synthetic theta calibrated to Medicare CT provider data (EC.1),
    matching the paper's exact formula:

        theta_ij = alpha + (1-alpha) * [omega*beta_ij + (1-omega)*(dbar/d_ij - 1)]

    where d_ij is the patient-provider distance, dbar is the fixed average
    distance threshold (20.2 miles, EC.1), beta_ij is 1 iff patient i's
    comorbidities and provider j's specialty overlap, and alpha/omega are
    hyperparameters (alpha sets the match quality at d_ij=dbar; omega weighs
    comorbidity concordance against distance). theta is clipped to [0, 1],
    the domain the paper states for it (Section 3.1); the formula itself is
    unbounded in both directions, so both ends would otherwise escape (the
    upper end because dbar/d_ij blows up for same-ZIP pairs, the lower end
    because dbar/d_ij - 1 -> -1 for very distant ones).

    clip_distance_term controls WHERE the [0,1] clip lands, which turns out
    to matter more than any other knob here. With it False the formula is
    applied literally and only the final theta is clipped, so theta decays
    like 1/d across the whole state and a patient's viable options span a
    range several times wider than epsilon. With it True the distance term
    (dbar/d_ij - 1) is clipped to [0,1] *before* mixing, so
    theta = alpha + (1-alpha)*[omega*beta_ij + (1-omega)*clip(dbar/d_ij - 1)]
    lies in [alpha, 1] with a plateau for d_ij <= dbar/2 and a floor for
    d_ij >= dbar. That is where the reference implementation clips
    (`_legacy/patient/semi_synthetic.py` clips its `distance_utility`, not
    its utility), and it compresses a patient's viable options into a band
    narrower than epsilon -- the regime in which estimate noise scrambles
    the ranking and menus are worth having. The paper's EC.1 text does not
    say which placement it means.

    Unlike the reference implementation there is no hard distance cutoff and
    no additive noise term (neither appears in the paper's formula): theta
    varies continuously with distance for every pair. The paper states no
    numeric value for alpha; it defaults to 0.5, the constant the reference
    implementation used, and stays tunable.

    Distances are floored at 0.5 miles before dividing, so dbar/d_ij cannot
    blow up for same- or adjacent-ZIP pairs. Pairs with no distance data in
    `ct_zipcode_distance.json` (a real data completeness gap, not a modeling
    choice) get theta=0.

    Returns:
        theta: num_patients x num_providers array (no exit column)
        patients: list of dicts with 'age', 'zip', and one 0/1 flag per comorbidity
        providers: list of dicts with 'zip' and one 0/1 flag per comorbidity/specialty
    """
    rng = np.random.RandomState(seed)
    ct_providers = _medicare_ct_providers()
    providers_df = ct_providers.sample(
        n=num_providers, replace=True, random_state=rng
    ).reset_index(drop=True)

    zipcodes, zip_probs = _ct_zip_population()
    ages, age_buckets = _ct_age_distribution()
    rates = _comorbidity_rates()
    distances = _zip_distances()

    patients = []
    for _ in range(num_patients):
        age = rng.choice(ages, p=age_buckets)
        loc = str(rng.choice(zipcodes, p=zip_probs))
        age_group = get_age_group(age)
        patient = {"age": age, "zip": loc}
        for c in COMORBIDITIES:
            patient[c] = int(rng.random() < rates[c][age_group])
        patients.append(patient)

    providers = []
    for j in range(num_providers):
        row = providers_df.iloc[j]
        providers.append({"zip": get_zip5(row["ZIP Code"]), **{c: int(row[c]) for c in COMORBIDITIES}})

    theta = np.zeros((num_patients, num_providers))
    for i in range(num_patients):
        patient_comorbid = np.array([patients[i][c] for c in COMORBIDITIES])
        for j in range(num_providers):
            distance = distances.get(str((patients[i]["zip"], providers[j]["zip"])))
            if distance is None or np.isnan(distance):
                continue
            distance = max(distance, 0.5)  # floor to avoid dividing by ~0 for same/adjacent ZIP pairs
            provider_comorbid = np.array([providers[j][c] for c in COMORBIDITIES])
            beta_ij = 1.0 if patient_comorbid.dot(provider_comorbid) > 0 else 0.0
            distance_term = average_distance / distance - 1
            if clip_distance_term:
                distance_term = min(max(distance_term, 0.0), 1.0)
            theta[i, j] = np.clip(
                alpha + (1 - alpha) * (omega * beta_ij + (1 - omega) * distance_term), 0.0, 1.0
            )

    return theta, patients, providers


def rescale_spread(theta, spread):
    """Scale how far apart one patient's options are, holding that patient's
    MEAN utility fixed:

        theta_ij <- mean_j(theta_ij) + spread * (theta_ij - mean_j(theta_ij))

    spread = 1 is the distribution untouched; spread = 0 collapses every
    provider to the patient's own mean, so the patient is indifferent and
    there is nothing for a menu to be about. Centring on the patient's own
    row mean rather than a global constant is what makes this a pure spread
    knob: it redistributes value within a row instead of raising or lowering
    the level, so a policy's utility cannot improve merely because everyone
    got better options. That is the same invariant `instance_family.build`
    enforces on the toy family by construction.

    Applies to any theta matrix, whatever generator produced it, and is the
    exception to the [0,1] clip mattering: for spread <= 1 the result is a
    convex combination of theta and a row mean, so it cannot leave the range
    the input was already in. spread > 1 CAN escape [0, 1] and is clipped
    here; be careful with it, because the clip piles the top options up at
    exactly 1.0 and destroys the very spread being asked for (this is what
    makes `normal_theta` unusable as a spread base -- its mu_j ~ U(0,1)
    already puts many providers at the ceiling).
    """
    if spread == 1.0:
        return theta
    row_mean = theta.mean(axis=1, keepdims=True)
    return np.clip(row_mean + spread * (theta - row_mean), 0.0, 1.0)


def uniform_theta(num_patients, num_providers, seed=None):
    """EC.2.3: theta_ij ~ Uniform(0, 1), i.i.d."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0, 1, size=(num_patients, num_providers))


def normal_theta(num_patients, num_providers, sigma=0.1, seed=None):
    """EC.2.3: theta_ij ~ N(mu_j, sigma^2), mu_j ~ Uniform(0, 1) shared across patients
    (homogeneous preferences: every patient roughly agrees on provider quality).

    sigma defaults to 0.1, the reference implementation's value, rather than
    the 0.05 the paper's text gives -- 0.1 is what produced the EC.3 figure."""
    rng = np.random.RandomState(seed)
    mu = rng.uniform(0, 1, size=num_providers)
    theta = mu[None, :] + rng.normal(0, sigma, size=(num_patients, num_providers))
    return np.clip(theta, 0, 1)


def latent_theta(num_patients, num_providers, latent_dim=5, sigma=0.05, seed=None):
    """EC.2.3: theta_ij ~ u_i . v_j + N(0, sigma^2), heterogeneous structured
    preferences driven by latent patient/provider type vectors.

    Follows the reference implementation: the type vectors are standard
    normal and the raw scores are min-max rescaled onto [0, 1] before the
    noise is added. The rescaling matters -- it is what gives theta a spread
    comparable to the semi-synthetic environment's. Drawing u, v uniform and
    dividing by latent_dim instead (a literal reading of "u_i . v_j")
    concentrates theta near 0.25, right on top of the exit utility, which
    makes the whole EC.2.3 comparison degenerate."""
    rng = np.random.RandomState(seed)
    u = rng.randn(num_patients, latent_dim)
    v = rng.randn(num_providers, latent_dim)
    scores = u @ v.T
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    theta = scores + rng.normal(0, sigma, size=(num_patients, num_providers))
    return np.clip(theta, 0, 1)
