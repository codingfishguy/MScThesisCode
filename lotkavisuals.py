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

def npe(posterior, data, alpha=0.1, trim=0.02):
    data_flat = torch.as_tensor(np.asarray(data).reshape(-1), dtype=torch.float32)
    posterior.set_default_x(data_flat)
    samples = posterior.sample((1000,)).numpy()

    # robust center, then drop the most extreme `trim` fraction by Mahalanobis distance
    med = np.median(samples, axis=0)
    mad_cov = np.cov(samples, rowvar=False) + np.eye(3) * 1e-8
    d2 = np.einsum('ij,jk,ik->i', samples - med, np.linalg.inv(mad_cov), samples - med)
    keep = d2 <= np.quantile(d2, 1 - trim)
    samples = samples[keep]

    mu = samples.mean(axis=0)
    Sigma = np.cov(samples, rowvar=False)
    threshold = chi2.ppf(1 - alpha, df=3)
    return (mu, Sigma, threshold)

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

    return (posterior, robglob_cutoff, theta_train)

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


def robconfregion_volume(posterior, data, cutoff, theta_train_np,
                          n_samples=50000, seed=None):
    theta_train_np = np.asarray(theta_train_np)   # handles torch tensor or numpy array alike
    rng = np.random.default_rng(seed)
    t_min = theta_train_np.min(axis=0)
    t_max = theta_train_np.max(axis=0)

    samples = rng.uniform(t_min, t_max, size=(n_samples, len(t_min)))
    samples_t = torch.tensor(samples, dtype=torch.float32)

    data_flat = np.asarray(data).reshape(-1)
    data_t = torch.tensor(data_flat, dtype=torch.float32)
    data_batch = data_t.unsqueeze(0).expand(n_samples, -1)

    with torch.no_grad():
        scores = -posterior.log_prob_batched(samples_t, data_batch).numpy()
        scores = np.ravel(scores)

    accepted = samples[scores <= cutoff]

    if len(accepted) < 10:
        box_volume = np.prod(t_max - t_min)
        return box_volume  # fallback: overestimate as before

    mu = accepted.mean(axis=0)
    Sigma = np.cov(accepted, rowvar=False)
    Sigma_inv = np.linalg.inv(Sigma)
    diffs = accepted - mu
    mahal_sq = np.einsum('ij,jk,ik->i', diffs, Sigma_inv, diffs)
    threshold = mahal_sq.max()  # smallest same-shaped ellipsoid enclosing all accepted points

    return ellipsoid_volume_3d(mu, Sigma, threshold)

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

def sample_prior(n, seed=None):
    rng = np.random.default_rng(seed)
    log_theta = rng.uniform(-6, 2, size=(n, 3))
    return np.exp(log_theta)

proposed_theta = sample_prior(50000, seed=123) 

def conformalregion_volume(posterior, cutoff, data, theta_candidates=proposed_theta):
    x_obs = torch.tensor(
        data.reshape(-1),
        dtype=torch.float32
    )
    theta_torch = torch.tensor(theta_candidates, dtype=torch.float32)
    scores = -posterior.log_prob(
        theta_torch,
        x=x_obs
    ).detach().numpy()

    accepted_theta = theta_candidates[scores <= cutoff]

    if len(accepted_theta) < 10:
        t_min = theta_candidates.min(axis=0)
        t_max = theta_candidates.max(axis=0)
        return np.prod(t_max - t_min)  # fallback: overestimate as before

    mu = accepted_theta.mean(axis=0)
    Sigma = np.cov(accepted_theta, rowvar=False)
    Sigma_inv = np.linalg.inv(Sigma)
    diffs = accepted_theta - mu
    mahal_sq = np.einsum('ij,jk,ik->i', diffs, Sigma_inv, diffs)
    threshold = mahal_sq.max()

    return ellipsoid_volume_3d(mu, Sigma, threshold)

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull
import numpy as np
import torch

PARAM_LABELS = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$']
PARAM_PAIRS = [(0, 1), (0, 2), (1, 2)]


def ellipse_2d_params(mu, Sigma, threshold, i, j):
    """
    Exact 2D projection (shadow) of the 3D ellipsoid
    {x : (x-mu)' Sigma^-1 (x-mu) <= threshold} onto coordinates (i, j).
    Uses the same threshold and the 2x2 marginal submatrix of Sigma.
    """
    mu2 = np.array([mu[i], mu[j]])
    Sigma2 = np.array([[Sigma[i, i], Sigma[i, j]],
                        [Sigma[j, i], Sigma[j, j]]])
    eigvals, eigvecs = np.linalg.eigh(Sigma2)
    order = np.argsort(eigvals)[::-1]        # descending: major axis first
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    widths = 2 * np.sqrt(np.clip(eigvals, 0, None) * threshold)  # [major, minor]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    return mu2, widths, angle


def draw_ellipse(ax, mu, Sigma, threshold, i, j, **kwargs):
    mu2, widths, angle = ellipse_2d_params(mu, Sigma, threshold, i, j)
    e = Ellipse(xy=mu2, width=widths[0], height=widths[1], angle=angle,
                fill=False, **kwargs)
    ax.add_patch(e)
    return e


def get_accepted_globconf(
    posterior,
    cutoff,
    theta_candidates=proposed_theta,
    batch_size=1000
):
    theta_candidates = np.asarray(theta_candidates)

    x_obs = torch.tensor(
        obs_data.reshape(-1),
        dtype=torch.float32
    )

    accepted = []

    with torch.no_grad():
        for start in range(0, len(theta_candidates), batch_size):
            stop = start + batch_size

            theta_batch = torch.tensor(
                theta_candidates[start:stop],
                dtype=torch.float32
            )

            scores = -posterior.log_prob(
                theta_batch,
                x=x_obs
            ).cpu().numpy().ravel()

            accepted.append(
                theta_candidates[start:stop][scores <= cutoff]
            )

    return np.concatenate(accepted)


def get_accepted_robconf(
    posterior,
    cutoff,
    theta_train_np,
    data,
    n_samples=2000,
    seed=0,
    batch_size=1000
):
    theta_train_np = np.asarray(theta_train_np)

    rng = np.random.default_rng(seed)

    t_min = theta_train_np.min(axis=0)
    t_max = theta_train_np.max(axis=0)

    samples = rng.uniform(
        t_min,
        t_max,
        size=(n_samples, 3)
    )

    data_t = torch.tensor(
        np.asarray(data).reshape(-1),
        dtype=torch.float32
    )

    accepted = []

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            stop = min(start + batch_size, n_samples)

            samples_batch = samples[start:stop]

            samples_t = torch.tensor(
                samples_batch,
                dtype=torch.float32
            )

            data_batch = data_t.unsqueeze(0).expand(
                len(samples_batch), -1
            )

            scores = -posterior.log_prob_batched(
                samples_t,
                data_batch
            ).cpu().numpy().ravel()

            accepted.append(
                samples_batch[scores <= cutoff]
            )

    return np.concatenate(accepted)

def draw_region_hull(ax, particles, i, j, **kwargs):
    pts = particles[:, [i, j]]
    if len(pts) == 0:
        # register an empty labeled artist so it still shows in the legend
        ax.plot([], [], **kwargs)
        return
    if len(pts) < 3:
        ax.scatter(*pts.T, s=8, **kwargs)  # keep the label now
        return
    hull = ConvexHull(pts)
    hull_pts = np.vstack([pts[hull.vertices], pts[hull.vertices[0]]])
    ax.plot(hull_pts[:, 0], hull_pts[:, 1], **kwargs)


def plot_credible_regions(theta_true, data,
                           abc_region, npe_region, robnpe_region,
                           globconf_posterior, globconf_cutoff,
                           robconf_posterior, robconf_cutoff, robconf_theta_train,
                           figsize=(15, 5)):
    """
    Each of abc_region / npe_region / robnpe_region is (mu, Sigma, threshold),
    as returned by abc_confregion / npe / robnpe.
    """
    global obs_data
    obs_data = data  # used inside get_accepted_globconf

    globconf_particles = get_accepted_globconf(globconf_posterior, globconf_cutoff)
    robconf_particles = get_accepted_robconf(robconf_posterior, robconf_cutoff,
                                              robconf_theta_train, data)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, (i, j) in zip(axes, PARAM_PAIRS):
        draw_ellipse(ax, *abc_region, i, j, edgecolor='tab:blue', label='ABC', lw=1.5)
        draw_ellipse(ax, *npe_region, i, j, edgecolor='tab:orange', label='NPE', lw=1.5)
        draw_ellipse(ax, *robnpe_region, i, j, edgecolor='tab:green', label='RobNPE', lw=1.5)
        draw_region_hull(ax, globconf_particles, i, j, color='tab:red', label='GlobConf', lw=1.5)
        draw_region_hull(ax, robconf_particles, i, j, color='tab:purple', label='RobConf', lw=1.5)

        ax.scatter([theta_true[i]], [theta_true[j]], marker='*', s=200,
                   color='black', zorder=5, label=r'$\theta_{true}$')

        ax.set_xlabel(PARAM_LABELS[i])
        ax.set_ylabel(PARAM_LABELS[j])
        ax.relim(); ax.autoscale_view()

    axes[0].legend(loc='upper left', bbox_to_anchor=(0, 1.25), ncol=6, fontsize=9)
    fig.tight_layout()
    return fig


ref_table = gen_ref_table(10000) 
train_size = round(0.8*len(ref_table))
train_table = ref_table[:train_size] #used to train the Conformal NPE
calib_table = ref_table[train_size:] 

# Train vanilla NPE on the training portion only
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
theta_true, obs_data = simulate_lv(seed=1, sigma=5, return_theta=True, dat=True)

abc_region = abc_confregion(ref_table=ref_table, data=obs_data)
npe_region = npe(posterior=posterior_npe, data=obs_data)
robnpe_region = robnpe(ref_table=ref_table, data=obs_data)

robposterior, robcutoff, robthetas = robconf(ref_table=ref_table, data=obs_data)

fig = plot_credible_regions(
    theta_true=theta_true, data=obs_data,
    abc_region=abc_region, npe_region=npe_region, robnpe_region=robnpe_region,
    globconf_posterior=posterior_conformal, globconf_cutoff=global_cutoff,
    robconf_posterior=robposterior, robconf_cutoff=robcutoff, robconf_theta_train=robthetas,
)
fig.savefig("lv_credible_regions_wellspec_new.png", dpi=150)