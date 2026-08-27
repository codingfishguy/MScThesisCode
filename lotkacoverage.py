import numpy as np
from scipy.stats import chi2
import torch
from sbi.inference import NPE

#in the Prangle paper sigma=exp(2.3), however, 
#here I let the well specified case be sigma=5 so that the effects of misspecification are slightly easier to see
#misspecified settings while (like in the Toy example) involve the simulator assuming sigma=5 when actually it is not
def simulate_lv(x10=50, x20=100, sigma = 5, T=33, max_transitions=100000, seed=2, dat=False,
               return_theta=False):
    #simulation done via the Gillespie method
    #Theta generated from the prior
    rng = np.random.default_rng(seed)
    records = []
    while len(records)<17: #conditioning on non-extinction, as in the Baragatti paper. Could choose to not do this if simulation takes too long
        if dat==True:
            theta1=1
            theta2=0.005
            theta3=0.6
            #the above lead to non extinction runs mostly
        else:
            logtheta1 = rng.uniform(-6, 2)
            logtheta2 = rng.uniform(-6, 2)
            logtheta3 = rng.uniform(-6, 2)
            theta1 = np.exp(logtheta1)
            theta2= np.exp(logtheta2)
            theta3 = np.exp(logtheta3)
        
        t = 0
        x1 = x10
        x2 = x20
    
        checks = list(range(2, T, 2))
        checkpoint=0
        rand1 = rng.normal(loc=0, scale=1)
        rand2 = rng.normal(loc=0, scale=1)
        records = [(
            x1 + sigma*rand1,
            x2 + sigma*rand2
        )]
        transition_count = 0
        while t < T and x1 > 0 and x2 > 0:
            checker = checks[checkpoint]
            
            a1 = theta1 * x1
            a2 = theta2 * x1 * x2
            a3 = theta3 * x2
    
            a0 = a1 + a2 + a3
    
            if a0 == 0:
                break
    
            # waiting time
            tau = rng.exponential(1 / a0) #uses scale, i.e inverse of rate parameter
            t += tau
            while checkpoint < len(checks) and t >= checks[checkpoint]:
                error1 = rng.normal(loc=0, scale=1)
                error2 = rng.normal(loc=0, scale=1)
                rec1 = x1+error1*sigma
                rec2 = x2+error2*sigma
                if rec1 <0:
                    rec1=int(0)
                if rec2<0:
                    rec2=int(0)
                records.append((rec1, rec2))
                checkpoint += 1
    
            if checkpoint == len(checks):
                break
            
            r = rng.random() * a0
            if r < a1:
                x1 += 1
    
            elif r < a1 + a2:
                x1 -= 1
                x2 += 1
    
            else:
                x2 -= 1
        
            transition_count += 1 
            if transition_count > max_transitions:
                return None      # reject this simulation
    if return_theta==True:
        theta=(theta1, theta2, theta3)
        return (theta, np.array(records))
    return np.array(records)

def gen_ref_table(N=10000): #upped samples when run on the cluster
    #takes around 30 seconds for 1000 entries
    ref_table = list()
    for i in range(N):
        pair = simulate_lv(sigma=5, return_theta=True, seed=i+1) #reminder: misspecification will not come from the prior here
        if pair: #sometimes max_iterations are exceeded. In this case we will have fewer than N pairs
            ref_table.append(pair) #uses a 34 dimensional summary. May suffer from curse of dimensionality here
    return ref_table

def reject_abc(ref_table, data, quantile=0.01):
    """
    ref_table: iterable of (theta, sims), where:
               theta has shape (3,)
               sims has shape (17, 2)
    data: array-like of shape (17, 2)
    quantile: quantile used to determine epsilon
    """
    data = np.asarray(data)
    sims = np.asarray([entry[1] for entry in ref_table])
    # Squared Euclidean distance for each simulation
    distances = np.sum((sims - data) ** 2, axis=(1, 2))
    # ABC threshold
    epsilon = np.quantile(distances, quantile)
    # Keep simulations below threshold
    accepted = distances <= epsilon
    return [
        (ref_table[i][0], ref_table[i][1])
        for i in np.flatnonzero(accepted)
    ]

def abc_confregion(ref_table, data, alpha=0.1): #again, a 90% credible region is the goal
    accepted = reject_abc(ref_table, data)
    #Now using these accepted samples to define a credible region
    theta_candidates = np.array([theta for theta, simulation in accepted])
    #Will use the multivariate normal ellipsoid method as in Baragatti et. al
    # Posterior mean
    mu = theta_candidates.mean(axis=0)
    # Posterior covariance
    Sigma = np.cov(theta_candidates, rowvar=False)
    threshold = chi2.ppf(1-alpha, df=3)
    return (mu, Sigma, threshold) #defines a (100*(1-alpha))% credible region under a Gaussian approximation of the posterior

def robnpe(ref_table, data, alpha=0.1, abc_quantile=0.01):
    ref_table2 = reject_abc(ref_table, data, quantile=abc_quantile)
    theta_train = np.array([theta for theta, sim in ref_table2])
    x_train = np.array([sim for theta, sim in ref_table2])
    # Flatten (17, 2) -> 34-dimensional vector
    x_train = x_train.reshape(len(x_train), -1)
    theta_train = torch.tensor(theta_train, dtype=torch.float32)
    x_train = torch.tensor(x_train, dtype=torch.float32)

    inference = NPE()  # uses default neural spline flow approach
    _ = inference.append_simulations(theta_train, x_train).train()
    posterior = inference.build_posterior()

    data_flat = np.asarray(data).reshape(-1)
    data_flat = torch.as_tensor(data_flat, dtype=torch.float32)
    posterior.set_default_x(data_flat)

    posterior_samples = posterior.sample((1000,)).numpy()  # shape (1000, 3)
    mu = posterior_samples.mean(axis=0)
    Sigma = np.cov(posterior_samples, rowvar=False)
    threshold = chi2.ppf(1-alpha, df=3)
    return (mu, Sigma, threshold) #defines a (100*(1-alpha))% credible region under a Gaussian approximation of the posterior

def npe(posterior, data, alpha=0.1):
    data_flat = np.asarray(data).reshape(-1)
    data_flat = torch.as_tensor(data_flat, dtype=torch.float32)
    posterior.set_default_x(data_flat)

    posterior_samples = posterior.sample((1000,)).numpy()  # shape (1000, 3)
    mu = posterior_samples.mean(axis=0)
    Sigma = np.cov(posterior_samples, rowvar=False)
    threshold = chi2.ppf(1-alpha, df=3)
    return (mu, Sigma, threshold) #defines a (100*(1-alpha))% credible region under a Gaussian approximation of the posterior

def is_in_ellipsoid(theta, mu, Sigma, threshold):
    theta = np.asarray(theta)
    diff = theta - mu
    Sigma_inv = np.linalg.inv(Sigma)
    mahalanobis_sq = diff @ Sigma_inv @ diff
    return mahalanobis_sq <= threshold
    
def ellipsoid_volume_3d(mu, Sigma, threshold):
    vol = (4/3) * np.pi * (threshold ** 1.5) * np.sqrt(np.linalg.det(Sigma))
    return vol

def robconf(ref_table, data, train_prop=0.8, alpha=0.1, abc_quantile1=0.0025, abc_quantile2=0.01):
    train_size = round(train_prop * len(ref_table))
    train_table = ref_table[:train_size]
    calib_vanilla = ref_table[train_size:]
    ref_table2 = reject_abc(train_table, data, quantile=abc_quantile1)
    theta_train = np.array([theta for theta, sim in ref_table2])
    x_train = np.array([sim for theta, sim in ref_table2])
    
    # Flatten (17, 2) -> 34-dimensional vector
    x_train = x_train.reshape(len(x_train), -1)
    theta_train = torch.tensor(theta_train, dtype=torch.float32)
    x_train = torch.tensor( x_train,dtype=torch.float32)

    inference = NPE()
    _ = inference.append_simulations(
        theta_train,
        x_train
    ).train()
    posterior = inference.build_posterior()

    material = reject_abc(
        calib_vanilla,
        data,
        quantile=abc_quantile2
    )
    theta_array = np.array([
        theta for theta, sim in material
    ])
    summary_abc = np.array([
        sim for theta, sim in material
    ])
    # Flatten (17, 2) -> 34-dimensional vector
    summary_abc = summary_abc.reshape(
        len(summary_abc), -1
    )
    theta_abc = torch.tensor(
        theta_array,
        dtype=torch.float32
    )
    summaries_torch = torch.tensor(
        summary_abc,
        dtype=torch.float32
    )
    # -------------------------------------------------
    # Calculate robust conformal cutoff
    # -------------------------------------------------
    scores = -posterior.log_prob_batched(
        theta_abc,
        summaries_torch
    ).detach().numpy()

    scores = np.ravel(scores)
    q_level = 1.0 - alpha
    robglob_cutoff = np.quantile( #could make this dependent on sample size
        scores,
        q_level
    )
    # -------------------------------------------------
    # Score the ABC-selected training simulations
    # -------------------------------------------------

    scores = -posterior.log_prob_batched(
        theta_train,
        x_train
    ).detach().numpy()

    scores = np.ravel(scores)

    # -------------------------------------------------
    # Construct conformal region
    # -------------------------------------------------

    accepted_thetas = theta_train.numpy()[
        scores <= robglob_cutoff
    ]

    if len(accepted_thetas) == 0:
        return None

    return (posterior, robglob_cutoff, theta_train, accepted_thetas)

def isin_robconfregion(theta, posterior, data, cutoff, theta_train_np):
    theta = np.asarray(theta).reshape(-1)
    t_min = np.asarray(theta_train_np).min(axis=0).reshape(-1)
    t_max = np.asarray(theta_train_np).max(axis=0).reshape(-1)

    if not all(float(t_min[i]) <= float(theta[i]) <= float(t_max[i]) for i in range(3)):
        return False

    theta_t = torch.tensor(theta, dtype=torch.float32)
    data_flat = np.asarray(data).reshape(-1)          # (17,2) -> (34,)
    data_t = torch.tensor(data_flat, dtype=torch.float32)

    score = -posterior.log_prob(theta_t, data_t).detach().item()
    return score <= cutoff


def robconfregion_volume(
    posterior,
    cutoff,
    data,
    theta_train,
    N=20000
):
    """
    Monte Carlo estimate of the RobConf region volume.

    The region is explicitly restricted to the box defined
    by theta_train, as in isin_robconfregion().
    """

    theta_train = np.asarray(theta_train)

    # RobConf parameter-space box
    lower = theta_train.min(axis=0)
    upper = theta_train.max(axis=0)

    # Uniform samples inside the theta_train box
    theta_candidates = np.random.uniform(
        lower,
        upper,
        size=(N, 3)
    )

    x_obs = torch.tensor(
        data.reshape(-1),
        dtype=torch.float32
    )

    theta_torch = torch.tensor(
        theta_candidates,
        dtype=torch.float32
    )

    scores = -posterior.log_prob(
        theta_torch,
        x=x_obs
    ).detach().numpy()

    Nc = np.sum(scores <= cutoff)

    box_volume = np.prod(upper - lower)

    return box_volume * Nc / N


def fit_npe(ref_table): #once trained can be used again and again on new data (amortised)
    theta_train = np.array([theta for theta, sim in ref_table])
    x_train = np.array(
        [x.reshape(-1) for theta, x in ref_table]
    )
    theta_train = torch.tensor(theta_train, dtype=torch.float32)
    x_train = torch.tensor(x_train, dtype=torch.float32)

    inference = NPE() #uses default neural spline flow approach
    _ = inference.append_simulations(
        theta_train,
        x_train
    ).train()
    posterior = inference.build_posterior()
    return posterior

def fit_global_conformal_cutoff(posterior, calib_table, alpha=0.1):
    scores = []
    for theta, sim in calib_table:
        theta_torch = torch.tensor(
            theta,
            dtype=torch.float32
        ).unsqueeze(0)
        sim_torch = torch.tensor(
            sim.reshape(-1),
            dtype=torch.float32
        ).unsqueeze(0)
        score = -posterior.log_prob(
            theta_torch,
            x=sim_torch
        ).item()
        scores.append(score)

    B = len(scores)
    q_level = min(
        (1 + 1/B)*(1-alpha),
        1.0
    )
    return np.quantile(scores, q_level)

def in_conformalregion(theta, posterior, cutoff, data):
    theta = np.asarray(theta).reshape(-1)
    theta_t = torch.tensor(theta, dtype=torch.float32)
    data_flat = np.asarray(data).reshape(-1)  # (17,2) -> (34,)
    data_t = torch.tensor(data_flat, dtype=torch.float32)
    score = -posterior.log_prob(theta_t, data_t).detach().item()
    return score <= cutoff

def conformalregion_volume(
    posterior,
    cutoff,
    data,
    N=500000
):
    """
    Monte Carlo estimate of the volume of the global
    conformal region over a range slightly smaller than the prior support.
    """

    lower = np.exp(-6) * np.ones(3)
    upper = np.exp(1) * np.ones(3) #a bit smaller than the prior region

    # Uniformly sample theta from the prior-support BOX.
    theta_candidates = np.random.uniform(
        lower,
        upper,
        size=(N, 3)
    )

    x_obs = torch.tensor(
        data.reshape(-1),
        dtype=torch.float32
    )

    theta_torch = torch.tensor(
        theta_candidates,
        dtype=torch.float32
    )

    scores = -posterior.log_prob(
        theta_torch,
        x=x_obs
    ).detach().numpy()

    Nc = np.sum(scores <= cutoff)

    box_volume = np.prod(upper - lower)

    return box_volume * Nc / N





#ref_table = gen_ref_table(20000) Used for the 3809740.pbs-7 cluster job
ref_table = gen_ref_table(10000)
train_size = round(0.8*len(ref_table))
train_table = ref_table[:train_size] #used to train the Conformal NPE
calib_table = ref_table[train_size:] 

theta_train = np.array([theta for theta, sim in ref_table])
x_train = np.array([sim for theta, sim in ref_table])
x_train = x_train.reshape(len(x_train), -1)
theta_train = torch.tensor(theta_train, dtype=torch.float32)
x_train = torch.tensor(x_train, dtype=torch.float32)
inference = NPE()  # uses default neural spline flow approach
_ = inference.append_simulations(theta_train, x_train).train()
posterior_npe = inference.build_posterior()

posterior_conformal = fit_npe(train_table)
global_cutoff = fit_global_conformal_cutoff(posterior=posterior_conformal, 
                                            calib_table=calib_table,
                                            alpha=0.1)

def assess_coverage(dat_iters = 500, alpha=0.1, sd=5):
    totcov_abc = 0
    totvolume_abc = 0
    totcov_robconf = 0
    totvolume_robconf = 0
    totcov_robnpe = 0
    totvolume_robnpe = 0 
    totcov_npe = 0
    totvolume_npe = 0
    totcov_globconf=0
    totvolume_globconf = 0
    total_experiments = 0
    for k in range(dat_iters):
        lv_output = simulate_lv(seed=k+1,sigma=sd,return_theta=True)
        if lv_output:
            theta_true, obs_data = lv_output
            total_experiments+=1
        else:
            continue
            
        #Coverage Check ABC and volume of ellipsoid
        abc_region_mu, abc_region_Sigma, abc_region_threshold = abc_confregion(
            ref_table=ref_table, data=obs_data, alpha=alpha)
        if is_in_ellipsoid(theta_true, abc_region_mu, abc_region_Sigma, abc_region_threshold):
            totcov_abc+=1
        totvolume_abc += ellipsoid_volume_3d(abc_region_mu, abc_region_Sigma, abc_region_threshold)

        #Coverage Check NPE and volume of ellipsoid
        npe_region_mu, npe_region_Sigma, npe_region_threshold = npe(posterior=posterior_npe,
                                                                    data=obs_data)
        if is_in_ellipsoid(theta_true, npe_region_mu, npe_region_Sigma, npe_region_threshold):
            totcov_npe+=1
        totvolume_npe += ellipsoid_volume_3d(npe_region_mu, npe_region_Sigma, npe_region_threshold)

        #Coverage Check RobNPE and volume of ellipsoid
        robnpe_region_mu, robnpe_region_Sigma, robnpe_region_threshold = robnpe(
            ref_table=ref_table, data=obs_data)
        if is_in_ellipsoid(theta_true, robnpe_region_mu, robnpe_region_Sigma, robnpe_region_threshold):
            totcov_robnpe+=1
        totvolume_robnpe += ellipsoid_volume_3d(robnpe_region_mu, robnpe_region_Sigma, robnpe_region_threshold)

        #Coverage Check GlobConf and approx. volume (overestimate) of conformal region
        if in_conformalregion(theta_true, posterior_conformal, global_cutoff, obs_data):
            totcov_globconf+=1
        totvolume_globconf+= conformalregion_volume(posterior=posterior_conformal, cutoff=global_cutoff, data=obs_data)

        #Coverage Check RobConf and approx. volume (overestimate) of conformal region
        robposterior, robcutoff, robthetas, robaccepted = robconf(ref_table=ref_table, data=obs_data) 
        if isin_robconfregion(theta_true, robposterior, data=obs_data, cutoff=robcutoff, theta_train_np=robthetas):
            totcov_robconf+=1
        totvolume_robconf += robconfregion_volume(
            posterior=robposterior,
            cutoff=robcutoff,
            data=obs_data,
            theta_train=robthetas,
            N=100000
        )
    cov_abc = totcov_abc/total_experiments
    meanvol_abc = totvolume_abc/total_experiments
    cov_npe = totcov_npe/total_experiments
    meanvol_npe = totvolume_npe/total_experiments
    cov_robnpe = totcov_robnpe/total_experiments
    meanvol_robnpe = totvolume_robnpe/total_experiments
    cov_robconf = totcov_robconf/total_experiments
    meanvol_robconf = totvolume_robconf/total_experiments
    cov_globconf = totcov_globconf/total_experiments
    meanvol_globconf = totvolume_globconf/total_experiments
    
    return (cov_abc, cov_robconf, cov_robnpe, cov_npe, cov_globconf,
            meanvol_abc, meanvol_robconf, meanvol_robnpe, meanvol_npe, meanvol_globconf,
           total_experiments)


wellspec = assess_coverage(dat_iters=500)

print("NEWEST WELLSPEC")
print(wellspec)
