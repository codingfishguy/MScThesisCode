import numpy as np
import torch
from sbi.inference import NPE
from CP4SBI.CP4SBI.baycon import BayCon
from CP4SBI.CP4SBI.scores import HPDScore
from scipy.stats import gaussian_kde
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def fit_npe(ref_table): #once trained can be used again and again on new data (amortised)
    theta_train = np.array([[theta] for theta, summary, sim in ref_table])
    x_train = np.array([summary for theta, summary, sim in ref_table])

    theta_train = torch.tensor(theta_train, dtype=torch.float32)
    x_train = torch.tensor(x_train, dtype=torch.float32)

    inference = NPE() #uses default neural spline flow approach
    _ = inference.append_simulations(
        theta_train,
        x_train
    ).train()
    posterior = inference.build_posterior()
    return posterior

def generate_dataset(seed, sigma):
    rng = np.random.default_rng(seed)
    theta = rng.normal(loc=0, scale=5) #this prior will never be the misspecified component in my tests
    simulations = rng.normal(loc=0, scale=1, size=100)
    return (theta + sigma*simulations, theta) #sigma alone controls the level of misspecification

def ideal_posterior(data, sigma=1): #found analytically
    prior_mean=0
    prior_sd=5
    n = len(data)
    xbar = np.mean(data)
    prior_var = prior_sd**2
    lik_var = sigma**2 #the ideal posterior will use the true sigma used to generate the dataset

    post_var = 1 / (1 / prior_var + n / lik_var)
    post_mean = post_var * (
        prior_mean / prior_var + n * xbar / lik_var
    )
    return post_mean, post_var #can use these quantities to simulate from the ideal posterior
    #and also to get a HPD region from this posterior

def gen_ref_table(N=25000, seed=17): #multiple SBI algorithms use this and I want this to be the same for each
    ref_table = list()
    rng = np.random.default_rng(seed)
    for i in range(N):
        theta = rng.normal(loc=0, scale=5) #prior belief
        simulations = rng.normal(theta, scale=1, size=100) #always assuming sigma=1 (this is where the misspecification comes from)
        summaries = (np.mean(simulations), np.var(simulations))
        ref_table.append((theta, summaries, np.array(simulations)))
    return ref_table


def reject_abc(ref_table, data, quantile=0.01):
    emp_mean = np.mean(data)
    emp_var = np.var(data)

    # Compute distances once
    distances = np.empty(len(ref_table))
    for i, (_, summaries, _) in enumerate(ref_table):
        d_mean = summaries[0] - emp_mean
        d_var = summaries[1] - emp_var
        distances[i] = np.hypot(d_mean, d_var)  # equivalent to sqrt(x**2 + y**2)

    epsilon = np.quantile(distances, quantile)

    # Reuse computed distances
    accepted_samples = [
        entry
        for entry, dist in zip(ref_table, distances)
        if dist <= epsilon
    ]
    return accepted_samples


def fit_global_conformal_cutoff(posterior, calib_table, alpha=0.1):
    scores = []

    for theta, summary, sim in calib_table:
        theta_torch = torch.tensor([[theta]], dtype=torch.float32)
        summary_torch = torch.tensor(summary, dtype=torch.float32)

        score = -posterior.log_prob(
            theta_torch,
            x=summary_torch
        ).item()

        scores.append(score)

    B = len(scores)
    q_level = min((1.0 + 1.0 / B) * (1.0 - alpha), 1.0)

    return np.quantile(scores, q_level)
    

theta_grid = np.linspace(-20, 20, 2000)

def global_conformal(posterior,cutoff,data, theta_candidates=theta_grid):
    obs_summary = np.array([
        np.mean(data),
        np.var(data)
    ])
    x_obs = torch.tensor(
        obs_summary.reshape(1, -1),
        dtype=torch.float32
    )
    theta_torch = torch.tensor(
        theta_candidates.reshape(-1, 1),
        dtype=torch.float32
    )
    scores = -posterior.log_prob(
        theta_torch,
        x=x_obs
    ).detach().numpy()

    accepted_thetas = theta_candidates[scores <= cutoff]
    if len(accepted_thetas) == 0:
        return None
    return (
        accepted_thetas.min(),
        accepted_thetas.max()
    )

def amort_robconf(posterior, data, calib_vanilla, alpha=0.1):
    summaries_obs = (np.mean(data), np.var(data))
    summary_torch = torch.tensor(summaries_obs, dtype=torch.float32)

    material = reject_abc(
        calib_vanilla,
        data,
        quantile=0.01
    )

    theta_array = np.array([
        theta for theta, summary, dist in material
    ])

    summary_abc = np.array([
        summary for theta, summary, dist in material
    ])

    theta_abc = torch.tensor(
        theta_array.reshape(-1, 1),
        dtype=torch.float32
    )

    summaries_torch = torch.tensor(
        summary_abc,
        dtype=torch.float32
    )

    scores = -posterior.log_prob_batched(
        theta_abc,
        summaries_torch
    ).detach().numpy()

    q_level = 1.0 - alpha
    robglob_cutoff = np.quantile(scores, q_level)

    theta_candidates = np.linspace(-20, 20, 2000)

    theta_candidates = torch.tensor(
        theta_candidates.reshape(-1, 1),
        dtype=torch.float32
    )

    scores_global = -posterior.log_prob(
        theta_candidates,
        x=summary_torch
    ).detach().numpy()

    accepted_thetas = theta_candidates[
        scores_global <= robglob_cutoff
    ]

    if len(accepted_thetas) == 0:
        return None

    return (
        accepted_thetas.min().item(),
        accepted_thetas.max().item()
    )


def robconf(ref_table, data, train_prop=0.8, alpha=0.1, abc_quantile1 = 0.0025, abc_quantile2=0.01, method=1):
    train_size= round(train_prop*len(ref_table))
    train_table = ref_table[:train_size]
    calib_vanilla = ref_table[train_size:]
    ref_table2 = reject_abc(train_table, data, quantile=abc_quantile1)
    theta_train = np.array([[theta] for theta, summary, sim in ref_table2])
    x_train = np.array([summary for theta, summary, sim in ref_table2])

    theta_train = torch.tensor(theta_train, dtype=torch.float32)
    x_train = torch.tensor(x_train, dtype=torch.float32)

    inference = NPE() #uses default neural spline flow approach
    _ = inference.append_simulations(
        theta_train,
        x_train
    ).train()
    posterior = inference.build_posterior()
    material = reject_abc(calib_vanilla, data, quantile=abc_quantile2) 
    theta_array = np.array([theta for theta, summary, dist in material])
    summary_abc = np.array([summary for theta, summary, dist in material])
    theta_abc = torch.tensor(theta_array.reshape(-1,1), dtype=torch.float32)
    summaries_torch = torch.tensor(summary_abc, dtype=torch.float32)
    scores = -posterior.log_prob_batched(
        theta_abc,
        summaries_torch
    ).detach().numpy()
    q_level = 1.0-alpha
    if method==1:
        robglob_cutoff = np.quantile(scores, q_level)
    else:
        B = len(scores)
        q_level = min((1.0 + 1.0 / B) * (1.0 - alpha), 1.0)
        robglob_cutoff= np.quantile(scores, q_level)
    scores = -posterior.log_prob_batched(
        theta_train,
        x_train
    ).detach().numpy()

    scores = np.ravel(scores)
    accepted_thetas = theta_train.numpy().ravel()[scores <= robglob_cutoff]
    return (accepted_thetas.min(), accepted_thetas.max())


ref_table = gen_ref_table(N=25000, seed=2) #will be the same for all levels of misspecification
posterior_npe = fit_npe(ref_table)
#LoCart requires the ref_table to be split into a calibration set as well
train_size = round(0.8*len(ref_table)) #this size was used in the paper
train_table = ref_table[:train_size] #used to train the NPE
calib_table = ref_table[train_size:] 

#Fit NPE on train_table for conformal methods
theta_train = np.array([[theta] for theta, summary, sim in train_table])
x_train = np.array([summary for theta, summary, sim in train_table])
theta_train = torch.tensor(theta_train, dtype=torch.float32)
x_train = torch.tensor(x_train, dtype=torch.float32)

inference = NPE()
_ = inference.append_simulations(
    theta_train,
    x_train
).train()
posterior = inference.build_posterior()
global_cutoff = fit_global_conformal_cutoff(
    posterior,
    calib_table,
    alpha=0.1
)


def assess_conditional_coverage(B_sim=500, K=1000, sd=1): #B_sim=500, K=1000

    # =========================================================
    # Initialise totals
    # =========================================================

    # Conditional coverage MAE
    totmae_glob = 0

    totmae_robconf_amort = 0
    totmae_robconf_025_01 = 0
    totmae_robconf_01_01 = 0
    totmae_robconf_01_04 = 0
    totmae_robconf_005_02 = 0

    # Mean credible-region width
    totwidth_glob = 0

    totwidth_robconf_amort = 0
    totwidth_robconf_025_01 = 0
    totwidth_robconf_01_01 = 0
    totwidth_robconf_01_04 = 0
    totwidth_robconf_005_02 = 0

    # Unconditional coverage
    totcov_glob = 0

    totcov_robconf_amort = 0
    totcov_robconf_025_01 = 0
    totcov_robconf_01_01 = 0
    totcov_robconf_01_04 = 0
    totcov_robconf_005_02 = 0

    rng = np.random.default_rng(seed=1)

    # =========================================================
    # Simulate datasets
    # =========================================================

    for i in range(B_sim):

        # -----------------------------------------------------
        # Coverage lists for this dataset
        # -----------------------------------------------------

        globconf_coverage = list()

        robconf_amort_coverage = list()
        robconf_025_01_coverage = list()
        robconf_01_01_coverage = list()
        robconf_01_04_coverage = list()
        robconf_005_02_coverage = list()

        # -----------------------------------------------------
        # Generate observed dataset
        # -----------------------------------------------------

        x_i, theta_true = generate_dataset(
            seed=i + 1,
            sigma=sd
        )

        # -----------------------------------------------------
        # Global conformal
        # -----------------------------------------------------

        glob = global_conformal(
            posterior=posterior,
            cutoff=global_cutoff,
            data=x_i
        )

        # -----------------------------------------------------
        # Amortised RobConf
        # -----------------------------------------------------

        robconf_amort = amort_robconf(
            posterior=posterior,
            data=x_i,
            calib_vanilla=calib_table,
            alpha=0.1
        )

        # -----------------------------------------------------
        # RobConf (quantile1=0.0025, quantile2=0.01)
        # -----------------------------------------------------

        robconf_025_01 = robconf(
            ref_table=ref_table,
            data=x_i,
            alpha=0.1,
            abc_quantile1=0.0025,
            abc_quantile2=0.01
        )

        # -----------------------------------------------------
        # RobConf (quantile1=0.01, quantile2=0.01)
        # -----------------------------------------------------

        robconf_01_01 = robconf(
            ref_table=ref_table,
            data=x_i,
            alpha=0.1,
            abc_quantile1=0.01,
            abc_quantile2=0.01
        )

        # -----------------------------------------------------
        # RobConf (quantile1=0.01, quantile2=0.04)
        # -----------------------------------------------------

        robconf_01_04 = robconf(
            ref_table=ref_table,
            data=x_i,
            alpha=0.1,
            abc_quantile1=0.01,
            abc_quantile2=0.04
        )

        # -----------------------------------------------------
        # RobConf (quantile1=0.005, quantile2=0.02)
        # -----------------------------------------------------

        robconf_005_02 = robconf(
            ref_table=ref_table,
            data=x_i,
            alpha=0.1,
            abc_quantile1=0.005,
            abc_quantile2=0.02
        )

        # -----------------------------------------------------
        # Unpack intervals
        # -----------------------------------------------------

        if glob is not None:
            glob_min, glob_max = glob

            # Unconditional coverage
            if glob_min <= theta_true <= glob_max:
                totcov_glob += 1

            # Width
            totwidth_glob += glob_max - glob_min

        if robconf_amort is not None:
            robconf_amort_min, robconf_amort_max = robconf_amort
            if robconf_amort_min <= theta_true <= robconf_amort_max:
                totcov_robconf_amort += 1
            totwidth_robconf_amort += (
                robconf_amort_max - robconf_amort_min
            )
        robconf_025_01_min, robconf_025_01_max = robconf_025_01
        robconf_01_01_min, robconf_01_01_max = robconf_01_01
        robconf_01_04_min, robconf_01_04_max = robconf_01_04
        robconf_005_02_min, robconf_005_02_max = robconf_005_02

        # -----------------------------------------------------
        # RobConf unconditional coverage
        # -----------------------------------------------------

        if robconf_amort_min <= theta_true <= robconf_amort_max:
            totcov_robconf_amort += 1

        if robconf_025_01_min <= theta_true <= robconf_025_01_max:
            totcov_robconf_025_01 += 1

        if robconf_01_01_min <= theta_true <= robconf_01_01_max:
            totcov_robconf_01_01 += 1

        if robconf_01_04_min <= theta_true <= robconf_01_04_max:
            totcov_robconf_01_04 += 1

        if robconf_005_02_min <= theta_true <= robconf_005_02_max:
            totcov_robconf_005_02 += 1

        # -----------------------------------------------------
        # RobConf widths
        # -----------------------------------------------------

        totwidth_robconf_amort += (
            robconf_amort_max - robconf_amort_min
        )

        totwidth_robconf_025_01 += (
            robconf_025_01_max - robconf_025_01_min
        )

        totwidth_robconf_01_01 += (
            robconf_01_01_max - robconf_01_01_min
        )

        totwidth_robconf_01_04 += (
            robconf_01_04_max - robconf_01_04_min
        )

        totwidth_robconf_005_02 += (
            robconf_005_02_max - robconf_005_02_min
        )

        # =====================================================
        # Draw samples from ideal posterior
        # =====================================================

        oracle_mean, oracle_var = ideal_posterior(
            x_i,
            sigma=sd
        )

        for k in range(K):

            theta_k = (
                oracle_mean
                + np.sqrt(oracle_var)
                * rng.normal(loc=0, scale=1)
            )

            # -------------------------------------------------
            # Global
            # -------------------------------------------------

            if glob is None:
                globconf_coverage.append(0)
            else:
                if glob_min <= theta_k <= glob_max:
                    globconf_coverage.append(1)
                else:
                    globconf_coverage.append(0)

            # -------------------------------------------------
            # Amortised RobConf
            # -------------------------------------------------

            if robconf_amort is None:
                robconf_amort_coverage.append(0)
            else:
                if robconf_amort_min <= theta_k <= robconf_amort_max:
                    robconf_amort_coverage.append(1)
                else:
                    robconf_amort_coverage.append(0)

            # -------------------------------------------------
            # RobConf (0.0025, 0.01)
            # -------------------------------------------------

            if robconf_025_01_min <= theta_k <= robconf_025_01_max:
                robconf_025_01_coverage.append(1)
            else:
                robconf_025_01_coverage.append(0)

            # -------------------------------------------------
            # RobConf (0.01, 0.01)
            # -------------------------------------------------

            if robconf_01_01_min <= theta_k <= robconf_01_01_max:
                robconf_01_01_coverage.append(1)
            else:
                robconf_01_01_coverage.append(0)

            # -------------------------------------------------
            # RobConf (0.01, 0.04)
            # -------------------------------------------------

            if robconf_01_04_min <= theta_k <= robconf_01_04_max:
                robconf_01_04_coverage.append(1)
            else:
                robconf_01_04_coverage.append(0)

            # -------------------------------------------------
            # RobConf (0.005, 0.02)
            # -------------------------------------------------

            if robconf_005_02_min <= theta_k <= robconf_005_02_max:
                robconf_005_02_coverage.append(1)
            else:
                robconf_005_02_coverage.append(0)

        # =====================================================
        # Conditional coverage MAE
        # =====================================================

        totmae_glob += abs(
            np.mean(globconf_coverage) - 0.9
        )

        totmae_robconf_amort += abs(
            np.mean(robconf_amort_coverage) - 0.9
        )

        totmae_robconf_025_01 += abs(
            np.mean(robconf_025_01_coverage) - 0.9
        )

        totmae_robconf_01_01 += abs(
            np.mean(robconf_01_01_coverage) - 0.9
        )

        totmae_robconf_01_04 += abs(
            np.mean(robconf_01_04_coverage) - 0.9
        )

        totmae_robconf_005_02 += abs(
            np.mean(robconf_005_02_coverage) - 0.9
        )

    # =========================================================
    # Average over simulated datasets
    # =========================================================

    # ---------------------------------------------------------
    # Conditional coverage MAE
    # ---------------------------------------------------------

    mae_glob = totmae_glob / B_sim

    mae_robconf_amort = totmae_robconf_amort / B_sim
    mae_robconf_025_01 = totmae_robconf_025_01 / B_sim
    mae_robconf_01_01 = totmae_robconf_01_01 / B_sim
    mae_robconf_01_04 = totmae_robconf_01_04 / B_sim
    mae_robconf_005_02 = totmae_robconf_005_02 / B_sim

    # ---------------------------------------------------------
    # Mean credible-region width
    # ---------------------------------------------------------

    meanwidth_glob = totwidth_glob / B_sim

    meanwidth_robconf_amort = totwidth_robconf_amort / B_sim
    meanwidth_robconf_025_01 = totwidth_robconf_025_01 / B_sim
    meanwidth_robconf_01_01 = totwidth_robconf_01_01 / B_sim
    meanwidth_robconf_01_04 = totwidth_robconf_01_04 / B_sim
    meanwidth_robconf_005_02 = totwidth_robconf_005_02 / B_sim

    # ---------------------------------------------------------
    # Unconditional coverage
    # ---------------------------------------------------------

    cov_glob = totcov_glob / B_sim

    cov_robconf_amort = totcov_robconf_amort / B_sim
    cov_robconf_025_01 = totcov_robconf_025_01 / B_sim
    cov_robconf_01_01 = totcov_robconf_01_01 / B_sim
    cov_robconf_01_04 = totcov_robconf_01_04 / B_sim
    cov_robconf_005_02 = totcov_robconf_005_02 / B_sim

    # =========================================================
    # Return
    #
    # Order:
    # 0-5   = MAE
    # 6-11  = mean width
    # 12-17 = unconditional coverage
    # =========================================================

    return (
        # MAE
        mae_glob,
        mae_robconf_amort,
        mae_robconf_025_01,
        mae_robconf_01_01,
        mae_robconf_01_04,
        mae_robconf_005_02,

        # Mean width
        meanwidth_glob,
        meanwidth_robconf_amort,
        meanwidth_robconf_025_01,
        meanwidth_robconf_01_01,
        meanwidth_robconf_01_04,
        meanwidth_robconf_005_02,

        # Unconditional coverage
        cov_glob,
        cov_robconf_amort,
        cov_robconf_025_01,
        cov_robconf_01_01,
        cov_robconf_01_04,
        cov_robconf_005_02
    )
#results_overspec = assess_conditional_coverage(sd=0.5)
results_wellspec = assess_conditional_coverage() 
#results_underspec15 = assess_conditional_coverage(sd=1.5)
#results_underspec2 = assess_conditional_coverage(sd=2)
#results_underspec3 = assess_conditional_coverage(sd=2.5)

#print(results_overspec)
print("WELLSPEC!!!")
print(results_wellspec)
#print(results_underspec15)
#print(results_underspec2)
#print(results_underspec3)


