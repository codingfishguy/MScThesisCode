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
'''
def abc_conformal(ref_table, calib_table, observed_data,
                  alpha=0.1, abc_quantile=0.01):
    scores = []
    material = reject_abc(calib_table, observed_data, quantile=abc_quantile)
    for theta, summmaries, x in material:
        accepted = reject_abc(ref_table, x, quantile=abc_quantile)
        posterior_samples = np.array([a[0] for a in accepted])
        kde = gaussian_kde(posterior_samples)
        # Negative posterior density at the true theta
        score = -kde(theta)[0]
        scores.append(score)
        
    scores = np.asarray(scores)
    q = np.quantile(scores, 1 - alpha)
    accepted = reject_abc(ref_table, observed_data,
                      quantile=abc_quantile)

    posterior_samples = np.array([a[0] for a in accepted])
    kde = gaussian_kde(posterior_samples)
    posterior_density = kde(posterior_samples)
    conformal_scores = -posterior_density
    credible_mask = conformal_scores <= q
    if not np.any(credible_mask):
        return None
    credible_thetas = np.sort(posterior_samples[credible_mask])
    return credible_thetas[0], credible_thetas[-1]
'''


#Robust Ideas:
#1. "Improve" the calibration set in LoCART
#Could do this by using ABC on the calibration set- ABC is robust to misspecification as well so could help
#Hard to wrap my brain around how this helps exactly though, but i suppose it makes the calibration set more connected to "reality"?
#Yeah because in the past the calibration set did not include enough potential candidates for theta around the observed data
'''
def robglob1(posterior, data, calib_vanilla, alpha=0.1, method=1):
    material = reject_abc(calib_vanilla, data, quantile=0.01) #I actually tried using the entire reference table here rather than only
    #i used a smaller quantile here after testing different ones
    #a smaller calibration set, but the method ended up performing worse.
    #I think in some other cases it might work, but it doesn't seem to result in much gains here
    #So maybe not like an asymptotic method that does better and better with more samples
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
    following would be commented out
    theta_candidates = np.linspace(-20, 20, 2000)
    theta_candidates = torch.tensor(theta_candidates.reshape(-1, 1), dtype=torch.float32)
    scores_global = -posterior.log_prob(theta_candidates,x=summary_torch).detach().numpy()
    accepted_thetas = theta_candidates[scores_global <= robglob_cutoff]
    end comment out here
    scores = np.ravel(scores)
    accepted_thetas = theta_array[scores<=robglob_cutoff]
    if len(accepted_thetas) == 0:
        return None
    return (accepted_thetas.min(), accepted_thetas.max())
'''
'''
def robglob2(posterior, data, train_table, calib_vanilla, alpha=0.1, method=1, abc_quantile=0.01,
            abc_quantile2=0.005):
    material = reject_abc(calib_vanilla, data, quantile=abc_quantile) 
    theta_array = np.array([theta for theta, summary, dist in material])
    summary_abc = np.array([summary for theta, summary, dist in material])
    theta_abc = torch.tensor(theta_array.reshape(-1,1), dtype=torch.float32)
    summaries_torch = torch.tensor(summary_abc, dtype=torch.float32)
    scores = -posterior.log_prob_batched(
        theta_abc,
        summaries_torch
    ).detach().numpy()
    q_level = 1.0-alpha
    scores = np.ravel(scores)
    robglob_cutoff = np.quantile(scores, q_level)
    following would be commented out
    theta_candidates = np.linspace(-20, 20, 2000)
    theta_candidates = torch.tensor(theta_candidates.reshape(-1, 1), dtype=torch.float32)
    scores_global = -posterior.log_prob(theta_candidates,x=summary_torch).detach().numpy()
    accepted_thetas = theta_candidates[scores_global <= robglob_cutoff]
    end comment out here
    accepted = reject_abc(train_table, data,
                      quantile=abc_quantile2)

    accepted_thetas = np.array([theta for theta, _, _ in accepted])
    theta_torch = torch.tensor(
        accepted_thetas.reshape(-1, 1),
        dtype=torch.float32
    )
    obs_summary = np.array([
        np.mean(data),
        np.var(data)
    ])

    summary_torch = torch.tensor(
        obs_summary,
        dtype=torch.float32
    ).unsqueeze(0)
    summaries_torch = summary_torch.repeat(len(accepted_thetas), 1)
    scores = -posterior.log_prob_batched(
        theta_torch,
        summaries_torch
    ).detach().numpy()

    scores = np.ravel(scores)
    accepted_thetas = accepted_thetas[scores <= robglob_cutoff]
    return (accepted_thetas.min(), accepted_thetas.max())
'''

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
'''
def robnpe2(ref_table, data, train_prop=0.8, alpha=0.1, abc_quantile=0.01):
    train_size= round(train_prop*len(ref_table))
    train_table = ref_table[:train_size]
    calib_vanilla = ref_table[train_size:]
    ref_table2 = reject_abc(train_table, data)
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
    material = reject_abc(calib_vanilla, data, quantile=abc_quantile) 
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
    scores = np.ravel(scores)
    accepted_thetas = theta_array[scores<=robglob_cutoff]
    if len(accepted_thetas) == 0:
        return None
    return (accepted_thetas.min(), accepted_thetas.max())
'''

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

#Fit LoCART part on calibration table
X_calib = np.array([summary for theta, summary, sim in calib_table])
theta_calib = np.array([[theta] for theta, summary, sim in calib_table])
baycon = BayCon(
    sbi_score=HPDScore,
    base_inference=inference,
    is_fitted=True,
    conformal_method="local",
    alpha=0.1,
    split_calib=True,
    weighting=False, #does not include variance estimate in the conformal procedure
    #tried including it earlier and the credible regions still suffered from the same problem
    #so it would not be a solution to misspecification anyway
)

baycon.calib(
    X_calib=X_calib,
    theta_calib=theta_calib,
    min_samples_leaf=100,
    prune_tree=True
)


global_cutoff = fit_global_conformal_cutoff(
    posterior,
    calib_table,
    alpha=0.1
)

def fit_models(sd, dataset_seed=17): #sd=1 is well-specified. Should include alpha eventually to make those plots
    obs_data, theta_true = generate_dataset(seed=dataset_seed, sigma=sd)
    npe_min, npe_max = cred_region_npe(posterior_npe, obs_data)
    abc_min, abc_max = abc_cred_region(ref_table, obs_data)
    locthing = locart_cred_region(posterior=posterior, baycon=baycon, data=obs_data)
    if locthing:
        locart_min = locthing[0]
        locart_max = locthing[1]
    else: #just so it shows up on the plots nicely enough
        locart_min=4
        locart_max=4       
    glob = global_conformal(posterior=posterior,cutoff=global_cutoff, data=obs_data)
    #robglob1_min, robglob1_max = robglob1(posterior=posterior, data=obs_data, calib_vanilla=calib_table)
    #robglob2_min, robglob2_max = robglob2(posterior=posterior, data=obs_data, train_table=train_table,
    #                                      calib_vanilla=calib_table)
    robnpe1_min, robnpe1_max = robnpe1(ref_table=ref_table, data=obs_data)
    #robnpe2_min, robnpe2_max = robnpe2(ref_table=ref_table, data=obs_data)
    robnpe3_min, robnpe3_max = robnpe3(ref_table=ref_table, data=obs_data)
    #abcconf = abc_conformal(ref_table=train_table,
    #                        calib_table = calib_table,
    #                        observed_data=obs_data)
    #if abcconf:
    #    abcconf_min = abcconf[0]
    #    abcconf_max = abcconf[1]
    #else:
    #    abcconf_min=4
    #    abcconf_max=4
    if glob is None:
        globconf_min, globconf_max = np.nan, np.nan
    else:
        globconf_min, globconf_max = glob
    post_mean, post_var = ideal_posterior(obs_data, sigma=sd)
    post_sd = np.sqrt(post_var)
    post_mean2, post_var2 = ideal_posterior2(obs_data, misspec_sigma=1)
    post_sd2 = np.sqrt(post_var2)
    z = norm.ppf(1 -  0.1 / 2)
    oracle_min = post_mean - z * post_sd
    oracle_max = post_mean + z * post_sd
    oracle_min2 = post_mean2 - z * post_sd2
    oracle_max2 = post_mean2 + z * post_sd2
    
    return dict(npe_vanilla = (npe_min, npe_max),
                reject_abc = (abc_min, abc_max),
                locart = (locart_min, locart_max),
                oracle = (post_mean, post_sd, oracle_min, oracle_max),
                oracle2 = (post_mean2, post_sd2, oracle_min2, oracle_max2),
                global_conf = (globconf_min, globconf_max),
                #abcconf = (abcconf_min, abcconf_max),
                #robustmethod1 = (robglob1_min, robglob1_max),
                #robustmethod2 = (robglob2_min, robglob2_max),
                robustmethod3 = (robnpe1_min, robnpe1_max),
                #robustmethod4 = (robnpe2_min, robnpe2_max),
                robustmethod5 = (robnpe3_min, robnpe3_max),
                theta_true=theta_true)

overspec = fit_models(sd=0.5) #simulator overspecifies true variance
wellspec = fit_models(sd=1)
underspec15 = fit_models(sd=1.5)
#underspec2 = fit_models(sd=1.8) #simulator underspecifies true variance (likely worse)
#underspec3 = fit_models(sd=2)

#Plot function different levels of misspecification

def plot_cred_regions(regions_dict, underspec_graph=False):
    post_mean = regions_dict['oracle'][0]
    post_sd = regions_dict['oracle'][1]
    oracle_min = regions_dict['oracle'][2]
    oracle_max = regions_dict['oracle'][3]
    #oracle_min2 = regions_dict['oracle2'][2]
    #oracle_max2 = regions_dict['oracle2'][3]

    npe_min = regions_dict['npe_vanilla'][0]
    npe_max = regions_dict['npe_vanilla'][1]
    abc_min = regions_dict['reject_abc'][0]
    abc_max = regions_dict['reject_abc'][1]
    #abcconf_min = regions_dict['abcconf'][0]
    #abcconf_max = regions_dict['abcconf'][1]
    locart_min = regions_dict['locart'][0]
    locart_max = regions_dict['locart'][1]
    globconf_min = regions_dict['global_conf'][0]
    globconf_max = regions_dict['global_conf'][1]
    #rob1min = regions_dict['robustmethod1'][0]
    #rob1max = regions_dict['robustmethod1'][1]
    #rob2min = regions_dict['robustmethod2'][0]
    #rob2max = regions_dict['robustmethod2'][1]
    rob3min = regions_dict['robustmethod3'][0]
    rob3max = regions_dict['robustmethod3'][1]
    #rob4min = regions_dict['robustmethod4'][0]
    #rob4max = regions_dict['robustmethod4'][1]
    rob5min = regions_dict['robustmethod5'][0]
    rob5max = regions_dict['robustmethod5'][1]


    theta_grid = np.linspace(oracle_min-1,oracle_max+1,2000)
    oracle_density = norm.pdf(theta_grid, loc=post_mean, scale=post_sd)
    ymax = oracle_density.max()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    band_height = 0.04 * ymax
    y_oracle = -1.0 * band_height
    #y_oracle2 = -2.0*band_height
    y_abc = -2.0 * band_height
    y_npe = -3.0 * band_height
    y_locart = -4.0*band_height
    y_globconf = -5.0*band_height
    #y_abcconf = -6.0*band_height
    #y_rob1 = -7.0*band_height
    #y_rob2 = -8.0*band_height
    y_rob3 = -6.0*band_height
    #y_rob4 = -9.0*band_height
    y_rob5 = -7.0*band_height

    ax.plot(theta_grid, oracle_density, color="black", label="Oracle posterior density")
    theta_true = regions_dict['theta_true']
    ax.axvline(
        theta_true,
        color="black",
        linestyle=":",
        linewidth=2,
        label=r"$\theta_{\mathrm{true}}$"
    )
    ax.broken_barh([(oracle_min, oracle_max - oracle_min)], (y_oracle, band_height),
                       facecolors="red", alpha=0.45, label="Oracle HPD")
    #ax.broken_barh([(oracle_min2, oracle_max2 - oracle_min2)], (y_oracle2, band_height),
    #                   facecolors="purple", alpha=0.45, label="Alt Oracle HPD")
    ax.broken_barh([(abc_min, abc_max - abc_min)], (y_abc, band_height),
                       facecolors="blue", alpha=0.45, label="ABC HPD")
    
    
    ax.broken_barh([(npe_min, npe_max - npe_min)], (y_npe, band_height),
                       facecolors="orange", alpha=0.45, label="NPE HPD")
    ax.broken_barh([(locart_min, locart_max - locart_min)], (y_locart, band_height),
                       facecolors="yellow", alpha=0.45, label="LoCART Cred Region")
    ax.broken_barh([(globconf_min, globconf_max - globconf_min)], (y_globconf, band_height),
                       facecolors="green", alpha=0.45, label="Global Conformal Cred Region")
    #ax.broken_barh([(abcconf_min, abcconf_max - abcconf_min)], (y_abcconf, band_height),
    #                   facecolors="cyan", alpha=0.45, label="ABCConf")
    #ax.broken_barh([(rob1min, rob1max - rob1min)], (y_rob1, band_height),
    #                   facecolors="pink", alpha=0.45, label="Robust Conformal Variation 1")
   # ax.broken_barh([(rob2min, rob2max - rob2min)], (y_rob2, band_height),
   #                    facecolors="pink", alpha=0.45, label="Rob2")
    ax.broken_barh([(rob3min, rob3max - rob3min)], (y_rob3, band_height),
                       facecolors="cyan", alpha=0.45, label="RobNPE")
    #ax.broken_barh([(rob4min, rob4max - rob4min)], (y_rob4, band_height),
    #                   facecolors="blue", alpha=0.45, label="Robust Conformal Variation 2")
    ax.broken_barh([(rob5min, rob5max - rob5min)], (y_rob5, band_height),
                       facecolors="pink", alpha=0.45, label="RobConf")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("Oracle posterior density")
    ax.set_ylim(y_rob5 - band_height, ymax * 1.1)
    if underspec_graph:
        ax.legend(loc="upper left")
    else:
        ax.legend(loc="upper right")
    plt.show()

plot_cred_regions(overspec)
plt.savefig("overspec_2.png", dpi=150)
plot_cred_regions(wellspec)
plt.savefig("wellspec_2.png", dpi=150)
plot_cred_regions(underspec15, underspec_graph=True)
plt.savefig("underspec15_2.png", dpi=150)


'''
plot_cred_regions(underspec2)
plt.savefig("underspec2.png", dpi=150)
plot_cred_regions(underspec3)
plt.savefig("underspec3.png", dpi=150)
'''






