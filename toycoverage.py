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

def ideal_posterior2(data, misspec_sigma):
    prior_mean=0
    prior_sd=5
    n = len(data)
    xbar = np.mean(data)
    prior_var = prior_sd**2
    lik_var = misspec_sigma**2 #the ideal posterior will use the true sigma used to generate the dataset

    post_var = 1 / (1 / prior_var + n / lik_var)
    post_mean = post_var * (
        prior_mean / prior_var + n * xbar / lik_var
    )
    return post_mean, post_var 

def gen_ref_table(N=25000, seed=17): #multiple SBI algorithms use this and I want this to be the same for each
    ref_table = list()
    rng = np.random.default_rng(seed)
    for i in range(N):
        theta = rng.normal(loc=0, scale=5) #prior belief
        simulations = rng.normal(theta, scale=1, size=100) #always assuming sigma=1 (this is where the misspecification comes from)
        summaries = (np.mean(simulations), np.var(simulations))
        ref_table.append((theta, summaries, np.array(simulations)))
    return ref_table

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

def cred_region_npe(posterior, data, alpha=0.1): #takes in a NPE posterior object and returns the KDE of a credible region
    emp_summaries = np.array([np.mean(data), np.var(data)])
    emp_summaries = torch.as_tensor(emp_summaries, dtype=torch.float32)    
    posterior.set_default_x(emp_summaries)

    posterior_samples = posterior.sample((1000,)).squeeze().numpy()
    kde = gaussian_kde(posterior_samples)
    density = kde(posterior_samples)
    cutoff_hpd = np.quantile(density, alpha)
    theta_hpd = posterior_samples[density >= cutoff_hpd]
    return (theta_hpd.min(), theta_hpd.max())

def locart_cred_region(posterior, baycon, data, alpha=0.1):
    obs_summary = np.array([
        np.mean(data),
        np.var(data)
    ])
    x_obs = torch.tensor(obs_summary.reshape(1, -1), dtype=torch.float32)
    c_obs = float(np.asarray(
        baycon.predict_cutoff(obs_summary.reshape(1, -1))
    ))

    #Find LoCART for observed data
    posterior.set_default_x(x_obs)

    #will need to calculate the scores based on the posterior with observed data over
    #-the entire feature space technically. However, could just use a limited subset and would probably be fine
    theta_grid = np.linspace(-20, 20, 2000).reshape(-1, 1)
    theta_grid_torch = torch.tensor(theta_grid, dtype=torch.float32)
    log_probs = posterior.log_prob(theta_grid_torch)
    density_grid = torch.exp(log_probs).detach().numpy()
    locart_region = theta_grid.ravel()[-density_grid <= c_obs]

    if len(locart_region)>0:
        return (locart_region.min(), locart_region.max())
    else:
        return None
    
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

def abc_cred_region(ref_table, data, alpha=0.1):
    accepted_samples = reject_abc(ref_table, data)
    theta_ = np.array([theta for theta, summary_, dist in accepted_samples])
    theta = np.sort(theta_)
    n = len(theta)
    m = int(np.floor((1 - alpha) * n))
    widths = theta[m - 1:] - theta[:n - m + 1]
    i = np.argmin(widths)
    return theta[i], theta[i + m - 1]

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

#Robust Ideas:
#1. "Improve" the calibration set in LoCART
#Could do this by using ABC on the calibration set- ABC is robust to misspecification as well so could help
#Hard to wrap my brain around how this helps exactly though, but i suppose it makes the calibration set more connected to "reality"?
#Yeah because in the past the calibration set did not include enough potential candidates for theta around the observed data

def robnpe1(ref_table, data, alpha=0.1, abc_quantile=0.002):
    ref_table2 = reject_abc(ref_table, data, quantile=abc_quantile)
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
    emp_summaries = np.array([np.mean(data), np.var(data)])
    emp_summaries = torch.as_tensor(emp_summaries, dtype=torch.float32)    
    posterior.set_default_x(emp_summaries)

    posterior_samples = posterior.sample((1000,)).squeeze().numpy()
    theta = np.sort(posterior_samples)
    n = len(theta)
    m = int(np.floor((1 - alpha) * n))
    widths = theta[m - 1:] - theta[:n - m + 1]
    i = np.argmin(widths)
    return theta[i], theta[i + m - 1]

def robnpe3(ref_table, data, train_prop=0.8, alpha=0.1, abc_quantile1 = 0.0025, abc_quantile2=0.01):
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
    robglob_cutoff = np.quantile(scores, q_level)
    scores = -posterior.log_prob_batched(
        theta_train,
        x_train
    ).detach().numpy()

    scores = np.ravel(scores)
    accepted_thetas = theta_train.numpy().ravel()[scores <= robglob_cutoff]
    return (accepted_thetas.min(), accepted_thetas.max())
    
    
ref_table = gen_ref_table(N=10000, seed=2) #will be the same for all levels of misspecification
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

#Fit LoCART part on calibration table
X_calib = np.array([summary for theta, summary, sim in calib_table])
theta_calib = np.array([[theta] for theta, summary, sim in calib_table])

def assess_conditional_coverage(B_sim=500, K=1000, sd=1, alpha=0.1, baycon=None, global_cutoff=None):
    totmae_locart = 0
    totmae_npe = 0
    totmae_abc = 0
    totmae_glob = 0
    totmae_robconf = 0
    totmae_robnpe = 0

    totwidth_locart = 0
    totwidth_npe = 0
    totwidth_abc = 0
    totwidth_glob = 0
    totwidth_robconf = 0
    totwidth_robnpe = 0

    coverage_locart = []
    coverage_npe = []
    coverage_abc = []
    coverage_glob = []
    coverage_robconf = []
    coverage_robnpe = []

    totcov_locart = 0
    totcov_npe = 0
    totcov_robnpe = 0
    totcov_abc = 0
    totcov_glob = 0
    totcov_robconf = 0

    rng = np.random.default_rng(seed=1)
    for i in range(B_sim):

        loc_coverage = []
        npe_coverage = []
        abc_coverage = []
        globconf_coverage = []
        robconf_coverage = []
        robnpe_coverage = []

        x_i, theta_true = generate_dataset(seed=i+1, sigma=sd)

        # -------------------------------------------------
        # Construct regions using THIS alpha
        # -------------------------------------------------

        loc = locart_cred_region(
            posterior=posterior,
            baycon=baycon,
            data=x_i,
            alpha=alpha
        )

        glob = global_conformal(
            posterior=posterior,
            cutoff=global_cutoff,
            data=x_i
        )

        robnpe_min, robnpe_max = robnpe1(
            ref_table=ref_table,
            data=x_i,
            alpha=alpha
        )

        robconf_min, robconf_max = robnpe3(
            ref_table=ref_table,
            data=x_i,
            alpha=alpha
        )

        npe_min, npe_max = cred_region_npe(
            posterior_npe,
            data=x_i,
            alpha=alpha
        )

        abc_min, abc_max = abc_cred_region(
            ref_table=ref_table,
            data=x_i,
            alpha=alpha
        )

        # -------------------------------------------------
        # Ideal posterior
        # -------------------------------------------------

        oracle_mean, oracle_var = ideal_posterior(
            x_i,
            sigma=sd
        )

        # -------------------------------------------------
        # Record widths
        # -------------------------------------------------

        if loc is not None:
            loc_min, loc_max = loc
            if loc_min <= theta_true <= loc_max:
                totcov_locart+=1
            totwidth_locart += loc_max - loc_min

        if glob is not None:
            glob_min, glob_max = glob
            if glob_min <= theta_true <= glob_max:
                totcov_glob+=1
            totwidth_glob += glob_max - glob_min

        totwidth_robnpe += robnpe_max - robnpe_min
        totwidth_robconf += robconf_max - robconf_min
        totwidth_npe += npe_max - npe_min
        totwidth_abc += abc_max - abc_min
    
        #Frequentist Coverage Check
        if robnpe_min <= theta_true <= robnpe_max:
            totcov_robnpe +=1
        if robconf_min <= theta_true <= robconf_max:
            totcov_robconf += 1
        if abc_min <= theta_true <= abc_max:
            totcov_abc += 1
        if npe_min <= theta_true <= npe_max:
            totcov_npe += 1

        # -------------------------------------------------
        # Monte Carlo coverage
        # -------------------------------------------------

        for k in range(K):

            theta_k = (
                oracle_mean
                + np.sqrt(oracle_var) * rng.normal()
            )

            # LoCART
            if loc is None:
                loc_coverage.append(0)
            else:
                loc_min, loc_max = loc
                loc_coverage.append(
                    int(loc_min <= theta_k <= loc_max)
                )

            # Global conformal
            if glob is None:
                globconf_coverage.append(0)
            else:
                glob_min, glob_max = glob
                globconf_coverage.append(
                    int(glob_min <= theta_k <= glob_max)
                )

            # NPE
            npe_coverage.append(
                int(npe_min <= theta_k <= npe_max)
            )

            # RobNPE
            robnpe_coverage.append(
                int(robnpe_min <= theta_k <= robnpe_max)
            )

            # RobConf
            robconf_coverage.append(
                int(robconf_min <= theta_k <= robconf_max)
            )

            # ABC
            abc_coverage.append(
                int(abc_min <= theta_k <= abc_max)
            )

        # -------------------------------------------------
        # MAE relative to nominal coverage = 1 - alpha
        # -------------------------------------------------
        coverage_locart.append(np.mean(loc_coverage))
        coverage_npe.append(np.mean(npe_coverage))
        coverage_abc.append(np.mean(abc_coverage))
        coverage_glob.append(np.mean(globconf_coverage))
        coverage_robconf.append(np.mean(robconf_coverage))
        coverage_robnpe.append(np.mean(robnpe_coverage))

        nominal_coverage = 1 - alpha

        totmae_locart += abs(
            np.mean(loc_coverage) - nominal_coverage
        )

        totmae_glob += abs(
            np.mean(globconf_coverage) - nominal_coverage
        )

        totmae_npe += abs(
            np.mean(npe_coverage) - nominal_coverage
        )

        totmae_abc += abs(
            np.mean(abc_coverage) - nominal_coverage
        )

        totmae_robconf += abs(
            np.mean(robconf_coverage) - nominal_coverage
        )

        totmae_robnpe += abs(
            np.mean(robnpe_coverage) - nominal_coverage
        )

    # -----------------------------------------------------
    # Average over datasets
    # -----------------------------------------------------

    mae_locart = totmae_locart / B_sim
    mae_npe = totmae_npe / B_sim
    mae_abc = totmae_abc / B_sim
    mae_glob = totmae_glob / B_sim
    mae_robconf = totmae_robconf / B_sim
    mae_robnpe = totmae_robnpe / B_sim

    meanwidth_locart = totwidth_locart / B_sim
    meanwidth_glob = totwidth_glob / B_sim
    meanwidth_robconf = totwidth_robconf / B_sim
    meanwidth_npe = totwidth_npe / B_sim
    meanwidth_abc = totwidth_abc / B_sim
    meanwidth_robnpe = totwidth_robnpe / B_sim

    cov_locart = totcov_locart / B_sim
    cov_npe = totcov_npe / B_sim
    cov_robnpe = totcov_robnpe / B_sim
    cov_abc = totcov_abc / B_sim
    cov_glob = totcov_glob / B_sim
    cov_robconf = totcov_robconf / B_sim

    return {
        "LoCART": {
            "MAE": mae_locart,
            "Coverage": np.mean(coverage_locart),
            "FreqCov": cov_locart,
            "Width": meanwidth_locart
        },
        "NPE": {
            "MAE": mae_npe,
            "Coverage": np.mean(coverage_npe),
            "FreqCov": cov_npe,
            "Width": meanwidth_npe
        },
        "ABC": {
            "MAE": mae_abc,
            "Coverage": np.mean(coverage_abc),
            "FreqCov": cov_abc,
            "Width": meanwidth_abc
        },
        "Global Conformal": {
            "MAE": mae_glob,
            "Coverage": np.mean(coverage_glob),
            "FreqCov": cov_glob,
            "Width": meanwidth_glob
        },
        "RobConf": {
            "MAE": mae_robconf,
            "Coverage": np.mean(coverage_robconf),
            "FreqCov": cov_robconf,
            "Width": meanwidth_robconf
        },
        "RobNPE": {
            "MAE": mae_robnpe,
            "Coverage": np.mean(coverage_robnpe),
            "FreqCov": cov_robnpe,
            "Width": meanwidth_robnpe
        }
    }

alphas = np.array([
    0.01,
    0.05,
    0.1,
    0.15,
    0.2
])

sigma_values = {
    "Overspecified (sigma=0.5)": 0.5,
    "Well-specified (sigma=1.0)": 1.0,
    "Underspecified (sigma=1.5)": 1.5,
    "Underspecified (sigma=2.0)": 2.0
}

results = {}

for sigma_name, sigma in sigma_values.items():

    results[sigma_name] = {}

    for alpha in alphas:

        print(
            f"Running sigma={sigma}, "
            f"alpha={alpha}"
        )

        # Recalibrate LoCART for this alpha
        baycon_alpha = BayCon(
            sbi_score=HPDScore,
            base_inference=inference,
            is_fitted=True,
            conformal_method="local",
            alpha=alpha,
            split_calib=True,
            weighting=False
        )

        baycon_alpha.calib(
            X_calib=X_calib,
            theta_calib=theta_calib,
            min_samples_leaf=100,
            prune_tree=True
        )

        # Global conformal cutoff for this alpha
        global_cutoff_alpha = fit_global_conformal_cutoff(
            posterior,
            calib_table,
            alpha=alpha
        )

        results[sigma_name][alpha] = assess_conditional_coverage(
            B_sim=500,
            K=1000,
            sd=sigma,
            alpha=alpha,
            baycon=baycon_alpha,
            global_cutoff=global_cutoff_alpha
        )

methods = [
    "LoCART",
    "NPE",
    "ABC",
    "Global Conformal",
    "RobConf",
    "RobNPE"
]

# ---------------------------------------------------------
# Plot empirical coverage vs nominal coverage (1 - alpha)
# ---------------------------------------------------------

fig, axes = plt.subplots(
    1,
    4,
    figsize=(18, 4.5),
    sharex=True,
    sharey=True
)

nominal_coverage = 1 - alphas

for col, (sigma_name, sigma) in enumerate(sigma_values.items()):

    ax = axes[col]

    # Ideal/nominal coverage line
    ax.plot(
        nominal_coverage,
        nominal_coverage,
        linestyle="--",
        color="black",
        label="Ideal coverage"
    )

    # Empirical coverage for each method
    for method in methods:

        coverage_values = [
            results[sigma_name][alpha][method]["Coverage"]
            for alpha in alphas
        ]

        ax.plot(
            nominal_coverage,
            coverage_values,
            marker="o",
            label=method
        )

    ax.set_title(sigma_name)
    ax.set_xlabel(r"Nominal coverage $1-\alpha$")
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Empirical coverage")

# One shared legend
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper left",
    ncol=4,
    bbox_to_anchor=(0.5, 1.08)
)

fig.suptitle(
    "Empirical coverage vs nominal coverage",
    fontsize=15
)

plt.tight_layout(rect=[0, 0, 1, 0.90])

plt.savefig(
    "conditional_coverage_vs_nominal_coverage.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()
# ---------------------------------------------------------
# Plot unconditional coverage vs nominal coverage
# ---------------------------------------------------------

fig, axes = plt.subplots(
    1,
    4,
    figsize=(18, 4.5),
    sharex=True,
    sharey=True
)

nominal_coverage = 1 - alphas

for col, (sigma_name, sigma) in enumerate(sigma_values.items()):

    ax = axes[col]

    # Ideal calibration line
    ax.plot(
        nominal_coverage,
        nominal_coverage,
        linestyle="--",
        color="black",
        label="Ideal coverage"
    )

    # Unconditional coverage for each method
    for method in methods:

        coverage_values = [
            results[sigma_name][alpha][method]["FreqCov"]
            for alpha in alphas
        ]

        ax.plot(
            nominal_coverage,
            coverage_values,
            marker="o",
            label=method
        )

    ax.set_title(sigma_name)
    ax.set_xlabel(r"Nominal coverage $1-\alpha$")
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Unconditional coverage")

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper left",
    ncol=4,
    bbox_to_anchor=(0.5, 1.08)
)

fig.suptitle(
    "Unconditional coverage vs nominal coverage",
    fontsize=15
)

plt.tight_layout(rect=[0, 0, 1, 0.94])

plt.savefig(
    "unconditional_coverage_vs_nominal.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


