import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import PoissonRegressor, LogisticRegression, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from numba import njit

"""
Script and functions for extracting species-level R0 values from field data as
used in the manuscript

Steps include:

1. Fit cyclic B-splines to four different data streams
2. Use these data streams to compute time-varying, species-level R0 values
3. Compute the next generation matrix and R_{0, T}.
4. Repeat for different sites and different parameter combinations

"""

def get_prevalence_splines(prev_data, C, dt, include_species, 
                           response_variable, plot_it=True,
                           saveit=True, n_knots=[8]):
    """
    Compute prevalence splines over Julian day for species in 
    "include_species"

    Parameters
    ----------
    prev_data : dataframe
        Dataframe that contains species-specific observed prevalence data
        through times.
    C : float
        Regulatization parameter.  Smaller is more regularization
    dt : float
        Time step of predictions in days
    include_species : list
        List of species for which to get the prevalence splines
    response_variable : str
        Either 'bd_pos' (for bd infection) or 'water' (for whether an animal was in the water)
    plot_it : bool
        Plot the splines for quick visual assessment
    save_it : bool
        Save the plots
    n_knots : list
        List of knots. The procedure will use a GridSearch to identify the optimal
        knots.

    Returns
    -------
    : dict
        Prevalence predictions over 365 days for each of the species. Key words are species

        The procedure also saves the "linear" model that has been fit. 
    """

    prev_results = {}
    fitting_info = {}
    for spp in include_species:

        tdat = prev_data[(prev_data.species == spp)].reset_index(drop=True)

        # Check if there are 2 classes and more than 10 data points
        unique_classes = tdat.loc[:, response_variable].unique()
        if len(unique_classes) == 2 and len(tdat) > 10:

            # Set-up the Julian day B-spline
            x_jd = tdat.dayofyear.values[:, np.newaxis]
            period = 365 # Specify the period

            # Create knots over the *full* period
            knot_list = [np.linspace(1, period, nk).reshape(-1, 1) for nk in n_knots]
            degree = 3 # Cubic spline

            # Set-up scikit learn pipeline
            pipeline = Pipeline([
                ('transform', SplineTransformer(degree=degree, include_bias=False, extrapolation="periodic")),
                ('model', LogisticRegression(fit_intercept=True, penalty="l2")) # Model
            ])

            param_grid = {
                'transform__knots': knot_list,
                'model__C': C
            }

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring = "f1",
                cv = 5
                )

            grid_search.fit(x_jd, tdat.loc[:, response_variable].values)
            best_mod = grid_search.best_estimator_

            best_C = grid_search.best_params_['model__C']
            print("Best C: {0}".format(best_C))

            # Extract the best model based on CV
            prev_mod = best_mod[1]
            spline_jd = best_mod[0]
            X_full = spline_jd.fit_transform(x_jd)

            jd_coef = prev_mod.coef_[0, :X_full.shape[1]][:, np.newaxis]
            jd_values = np.arange(1, 366, step=dt)[:, np.newaxis]
            Xnew_jd = spline_jd.fit_transform(jd_values)
            jd_pred = Xnew_jd @ jd_coef + prev_mod.intercept_
            prev_results[spp] = pd.DataFrame({'jd': jd_values.ravel(),
                                              'jd_effect': 1 / (1 + np.exp(-jd_pred.ravel())),
                                              'species': spp})

            # Save the design matrix, the y data, and the spline transformer
            fitting_info[spp] = {"X": X_full,
                                 "x": x_jd, 
                                 "y": tdat.loc[:, response_variable].values, 
                                 "spline": spline_jd}


        else:

            # Here, you are really just assigning dummy values because 
            # we don't have enough data to fit the spline. 

            jd_values = np.arange(1, 366, step=dt)[:, np.newaxis]

            if len(unique_classes) == 2:
                rep_val =  tdat.loc[:, response_variable].mean()
            elif unique_classes[0] == 0:
                rep_val = 0.01
            else:
                if len(tdat.loc[:, response_variable]) > 10:
                    rep_val = 1 #
                else: 
                    rep_val = 0.5

            prev_results[spp] = pd.DataFrame({'jd': jd_values.ravel(),
                                              'jd_effect': np.repeat(rep_val, jd_values.shape[0]),
                                              'species': spp})

            # Save the design matrix, the y data, and the spline transformer
            fitting_info[spp] = {"X": None,
                                 "x": tdat.dayofyear.values, 
                                 "y": tdat.loc[:, response_variable].values, 
                                 "spline": None}



    if plot_it:

        # Use to confirm the model predictions are making sense

        fig, ax = plt.subplots(1, 1)
        for i, spp in enumerate(include_species):
            tres = prev_results[spp]

            ti = i % 10
            ax.plot(tres.jd, tres.jd_effect, color=sns.color_palette()[ti], label=spp)

            tdat = prev_data[(prev_data.species == spp)].reset_index(drop=True)

            month_prev = tdat.groupby(['month']).agg({response_variable : np.mean}).reset_index()

            # plt.plot(la_site_no_na.dayofyear, la_site_no_na.bd_pos, 'o', color=sns.color_palette()[i])
            ax.plot(month_prev.month.values*30, month_prev.loc[:, response_variable].values, 'o', color=sns.color_palette()[ti])

        ax.legend()
        ax.set_xlabel("Julian Day")
        ax.set_ylabel(response_variable)

        if saveit:
            fig.savefig("../results/{0}_{1}.pdf".format(prev_data.site_id.unique()[0], response_variable))
            plt.close("all")

    # Save all of the fitting info externally
    pd.to_pickle(fitting_info, "../results/all_fitting_info_{0}_{1}.pkl".format(prev_data.site_id.unique().astype(np.int64)[0], response_variable))

    return(prev_results)


def get_load_splines(load_data, alpha, dt, include_species, 
                     plot_it=True, saveit=True,
                     n_knots=[8], fixed_mean=None):
    """
    Compute Bd log load splines over Julian day for species in 
    "include_species"

    Parameters
    ----------
    load_data : DataFrame
        Expects 
    alpha : float
        Regularization parameter.  Smaller is **less** regularization
    dt : float
        Time step of predictions in days
    include_species : list
        List of species for which to get the prevalence splines
    plot_it : bool
        Plot the splines for quick visual assessment
    save_it : bool
        Save the plots
    n_knots : list
        List of knots. The procedure will use a GridSearch to identify the optimal
        knots.
    fixed_mean : float or None
        If there are not enough points to run the regression and not None, the
        model just returns the value fixed_mean

    Returns
    -------
    : dict
        Log10 load predictions over 365 days for each of the species. 
        Key words are species
    """


    load_results = {}
    fitting_info = {}
    for spp in include_species:

        # Remove the NA body temperatures
        tdat = load_data[(load_data.bd_pos == 1) &
                         (load_data.species == spp)].reset_index(drop=True)

        nsamps = tdat.shape[0]

        if nsamps > 5:

            ### Set-up the Julian day Bspline ###
            x_jd = tdat.dayofyear.values[:, np.newaxis]
            period = 365 # Specify the period

            # Create knots over the *full* period
            # Generate knot list over the full period
            knot_list = [np.linspace(1, period, nk).reshape(-1, 1) for nk in n_knots]
            degree = 3

            # Set-up sklearn pipeline

            pipeline = Pipeline([
                ('transform', SplineTransformer(degree=degree, include_bias=False, extrapolation="periodic")),
                ('model', Ridge(fit_intercept=True)) # Model
            ])

            param_grid = {
                'model__alpha': alpha,
                'transform__knots': knot_list
            }

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring = "r2",
                cv = 5)

            grid_search.fit(x_jd, tdat.log10_bd.values)
            best_mod = grid_search.best_estimator_

            best_alpha = grid_search.best_params_['model__alpha']
            print("Best alpha: {0}".format(best_alpha))

            # Extract the estimator and transformer 
            load_mod = best_mod[1]
            spline_jd = best_mod[0]
            X_full = spline_jd.fit_transform(x_jd)


            jd_coef = load_mod.coef_[:X_full.shape[1]][:, np.newaxis]
            jd_values = np.arange(1, period + 1, step=dt)[:, np.newaxis]
            Xnew_jd = spline_jd.fit_transform(jd_values)
            jd_pred = Xnew_jd @ jd_coef + load_mod.intercept_
            load_results[spp] = pd.DataFrame({'jd': jd_values.ravel(),
                                              'jd_effect': jd_pred.ravel(),
                                              'species': spp})

            # Save the design matrix, the y data, and the spline transformer
            fitting_info[spp] = {"X": X_full,
                                 "x": x_jd, 
                                 "y": tdat.log10_bd.values, 
                                 "spline": spline_jd}


        else:

            jd_values = np.arange(1, 366, step=dt)

            if nsamps > 0: 
                jd_effect = np.repeat(tdat.log10_bd.mean(), len(jd_values))
            else:
                if fixed_mean is None:
                    jd_effect = np.repeat(-np.inf, len(jd_values))
                else:
                    jd_effect = np.repeat(fixed_mean, len(jd_values))


            load_results[spp] = pd.DataFrame({'jd': jd_values,
                                              'jd_effect': jd_effect,
                                              'species': spp})

            # Save the design matrix, the y data, and the spline transformer
            fitting_info[spp] = {"X": None, 
                                 "x": None,
                                 "y": tdat.log10_bd.values, 
                                 "spline": None}


    if plot_it:
        # Confirm the the model predictions are making sense
        fig, ax = plt.subplots(1, 1)
        for i, spp in enumerate(include_species):
            tres = load_results[spp]

            ti = i % 10
            ax.plot(tres.jd, tres.jd_effect, color=sns.color_palette()[ti], label=spp)

            tdat = load_data[(load_data.bd_pos == 1) &
                                      (load_data.species == spp)].reset_index(drop=True)

            # plt.plot(la_site_no_na.dayofyear, la_site_no_na.bd_pos, 'o', color=sns.color_palette()[i])
            ax.plot(tdat.dayofyear, tdat.log10_bd, 
                    'o', color=sns.color_palette()[ti])

        ax.legend()
        ax.set_xlabel("Julian Day")
        ax.set_ylabel("log10(Bd load)")
        if saveit:
            fig.savefig("../results/{0}_load.pdf".format(load_data.site_id.unique()[0]))
            plt.close("all")

    # Save all of the fitting info externally
    pd.to_pickle(fitting_info, "../results/all_fitting_info_{0}_load.pkl".format(load_data.site_id.unique().astype(np.int64)[0]))

    return(load_results)


def get_abundance_splines(count_data, alpha, dt, include_species, plot_it=True,
                          n_knots=[8], saveit=True):
    """
    Compute abundance splines over Julian day for species in 
    "include_species"

    Parameters
    ----------
    count_data : DataFrame
    alpha : float
        Regularization parameter.  Smaller is **less** regularization
    dt : float
        Time step of predictions in days
    include_species : list
        List of species for which to get the prevalence splines
    plot_it : bool
        Plot the splines for quick visual assessment
    save_it : bool
        Save the plots
    n_knots : list
        List of knots. The procedure will use a GridSearch to identify the optimal
        knots.

    Returns
    -------
    : dict
        Count predictions over 365 days for each of the species. Key words are species
    """

    count_results = {}
    fitting_info = {}
    for spp in include_species:

        # Remove the NA body temperatures
        tdat = count_data[(count_data.species == spp)].reset_index(drop=True)

        ### Set-up the Julian day Bspline ###
        x_jd = tdat.dayofyear.values[:, np.newaxis]
        n_knots = n_knots # Number of knots
        period = 365 # Specify the period

        # Create knots over the *full* period
        knot_list = [np.linspace(1, period, nk).reshape(-1, 1) for nk in n_knots]
        degree = 3

        pipeline = Pipeline([
            ('transform', SplineTransformer(degree=degree, include_bias=False, extrapolation="periodic")),
            ('model', PoissonRegressor(fit_intercept=True)) # Model
        ])

        param_grid = {
            'transform__knots': knot_list,
            'model__alpha': alpha
        }

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring = "neg_mean_poisson_deviance",
            cv = 5
            )
        grid_search.fit(x_jd, tdat['count'].values)
        best_mod = grid_search.best_estimator_

        best_alpha = grid_search.best_params_['model__alpha']
        print("Best alpha: {0}".format(best_alpha))

        count_mod = best_mod[1]
        spline_jd = best_mod[0]
        X_full = spline_jd.fit_transform(x_jd)


        jd_coef = count_mod.coef_[:X_full.shape[1]][:, np.newaxis]
        jd_values = np.arange(1, period + 1, step=dt)[:, np.newaxis]
        Xnew_jd = spline_jd.fit_transform(jd_values)
        jd_pred = Xnew_jd @ jd_coef + count_mod.intercept_
        count_results[spp] = pd.DataFrame({'jd': jd_values.ravel(),
                                           'jd_effect': np.exp(jd_pred.ravel()),
                                            'species': spp})

        # Save the design matrix, the y data, and the spline transformer
        fitting_info[spp] = {"X": X_full,
                             "x": x_jd, 
                             "y": tdat['count'].values, 
                             "spline": spline_jd}

    if plot_it:

        # Confirm the the model predictions are making sense
        fig, ax = plt.subplots(1, 1)
        for i, spp in enumerate(include_species):
            tres = count_results[spp]

            ti = i % 10
            ax.plot(tres.jd, tres.jd_effect, color=sns.color_palette()[ti], label=spp)

            tdat = count_data[(count_data.species == spp)].reset_index(drop=True)

            # plt.plot(la_site_no_na.dayofyear, la_site_no_na.bd_pos, 'o', color=sns.color_palette()[i])
            ax.plot(tdat.dayofyear, tdat['count'], 
                    'o', color=sns.color_palette()[ti])


        ax.legend()
        ax.set_xlabel("Julian Day")
        ax.set_ylabel("Count")
        if saveit:
            fig.savefig("../results/{0}_counts.pdf".format(count_data.site_id.unique()[0]))
            plt.close("all")

    # Save all of the fitting info externally
    pd.to_pickle(fitting_info, "../results/all_fitting_info_{0}_count.pkl".format(count_data.site_id.unique().astype(np.int64)[0]))
    return(count_results)


@njit
def compute_Z(all_intensity, all_prev, all_density,
              all_space_use, dt, path_decay, 
              period=365, num_prior_periods=2):
    """
    Compute seasonally varying Zoospore values and seasonal equilibrium give 
    just observed intensity, prevalence, density, and space use. A 
    pathogen decay rate needs to be provided.

    Parameters
    ----------
    all_intensity : array
        An S X Q array of where S is the number of species and Q is the number of time steps.
        Typically S = 1 (one species at a time), but you can have S > 1. The
        values are proportional toe shedding rates.
    all_prev : array
        An S X Q array of where S is the number of species and Q is the number of time steps.
        Typically S = 1 (one species at a time), but you can have S > 1. The
        values are prevalence values.
    all_density : array
        An S X Q array of where S is the number of species and Q is the number of time steps.
        Typically S = 1 (one species at a time), but you can have S > 1. The
        values are relative density values.
    all_space_use : array
        An S X Q array of where S is the number of species and Q is the number of time steps.
        Typically S = 1 (one species at a time), but you can have S > 1. The
        values are probabilities of being in the aquatic environment.
    dt : float
        Time step
    path_decay : array
        An array of length Q giving the time varying pathogen decay rates.
    period : float
        The time period of the seasonal cycle. Default is 365 (starting at 0), 
        but could be any period.
    num_prior_periods : int
        This specifies how far back in the past you need to go.  When Z persists
        for awhile in the environment, this value will need to be larger to
        get a good estimate of Z^*_s(\tau). The function defaults to 2 -- go
        bag two periods when computing Z.

    Returns
    -------
    : A vector of length Q giving the seasonal equilibrium abundance of Z - Z^*_s(\tau)


    """
    
    past_time_start = period*num_prior_periods # Lower bound on integral
    steps_prior = np.arange(-past_time_start, 0, step=dt)
    tvals = np.arange(0, period, step=dt)
    Zvals = np.empty((all_prev.shape[0], len(tvals)))

    past_path_decay_base = path_decay.repeat(num_prior_periods).reshape((-1, num_prior_periods)).T.ravel()

    # Species loop for Z. Each species contributes to Z
    for s in range(Zvals.shape[0]):
        
        # Just specify these once per species to save computational time
        past_intensity_base = all_intensity[s, :].repeat(num_prior_periods).reshape((-1, num_prior_periods)).T.ravel()
        past_prev_base = all_prev[s, :].repeat(num_prior_periods).reshape((-1, num_prior_periods)).T.ravel()
        past_density_base = all_density[s, :].repeat(num_prior_periods).reshape((-1, num_prior_periods)).T.ravel()
        past_space_use_base = all_space_use[s, :].repeat(num_prior_periods).reshape((-1, num_prior_periods)).T.ravel()

        # Loop over all time steps in the period to calculate Z from renewal equation
        for i, t in enumerate(tvals):

            steps_current = np.arange(0, t, step=dt) - t
            all_steps = np.concatenate((steps_prior - t, steps_current))

            # Get each species' contribution
            past_intensity = np.concatenate((past_intensity_base, all_intensity[s, :len(steps_current)]))
            past_prev = np.concatenate((past_prev_base, all_prev[s, :len(steps_current)]))
            past_density = np.concatenate((past_density_base, all_density[s, :len(steps_current)]))
            past_space_use = np.concatenate((past_space_use_base, all_space_use[s, :len(steps_current)]))
            past_path_decay = np.concatenate((past_path_decay_base, path_decay[:len(steps_current)]))

            # Integral approximation
            Z = np.sum((past_intensity * past_prev * past_density * past_space_use) * np.exp(-past_path_decay * -1*all_steps) * dt)

            Zvals[s, i] = Z
        
    # Sum over Z contributions from every species
    totZ = Zvals.sum(axis=0)
    return(totZ)


def get_all_dzdt(all_intensity, all_prev, all_density, all_water,
                  path_decay, dt, period=365, num_prior_periods=2):
    """

    For all S species, get dz / d\tau.  

    Parameters
    ----------
    all_intensity : array
        An S X Q array of where S is the number of species and Q is the number of time steps.
        Values are proportional to shedding rates.
    all_prev : array
        An S X Q array of where S is the number of species and Q is the number of time steps.
        Values are prevalences.
    all_density : array
        An S X Q array of where S is the number of species and Q is the number of time steps.
        Values are relative abundances.
    all_water : array
        Same as `all_space_use` from compute_Z. An S X Q array of where S is 
        the number of species and Q is the number of time steps. Values are the
        time-varying probability of species being in the aquatic habitat.
    dt : float
        Time step
    path_decay : array
        An array of length Q giving the time varying pathogen decay rates.
    period : float
        The time period of the seasonal cycle. Default is 365 (starting at 0), 
        but could be any period.
    num_prior_periods : int
        This specifies how far back in the past you need to go.  When Z persists
        for awhile in the environment, this value will need to be larger to
        get a good estimate of Z^*_s(\tau). The function defaults to 2 -- go
        bag two periods when computing Z.

    Returns
    -------
    : S x (Q - 1) array
        Numerical approximations of dZ / d\tau for each species at seasonal
        equilibrium.

    """

    # Calculcate dZdt for each species separately
    n = all_intensity.shape[1]
    s = all_intensity.shape[0]
    all_dzdt = np.empty((s, n - 1))

    # Loop over species
    for ts in range(s):

        # Extract a single species values
        tintensity = all_intensity[ts, :].reshape((1, n))
        tprev = all_prev[ts, :].reshape((1, n))
        tdensity = all_density[ts, :].reshape((1, n))
        twater = all_water[ts, :].reshape((1, n))

        totZ = compute_Z(tintensity, tprev, tdensity, 
                         twater, dt, path_decay, 
                         period=period, 
                         num_prior_periods=num_prior_periods)
        dzdt = (totZ[1:] - totZ[:-1]) / dt
        all_dzdt[ts, :] = dzdt

    return(all_dzdt)


def get_species_level_seasonal_R0_full(all_intensity, all_prev, all_density,
                                       all_water, overlap,
                                       bvals, phi, path_decay, dzdt,
                                       include_alternative_site=False):
    """
    Compute species-level R0 from observed data.  Computational implementation
    of equation 2 from the main text.

    Parameters
    ----------
    all_intensity : array
        An S X Q array of where S is the number of species and Q is the number of time steps.
        Values are proportional to shedding rates.
    all_prev : array
        An S X Q array of where S is the number of species and Q is the number of time steps.
        Values are prevalences.
    all_density : array
        An S X Q array of where S is the number of species and Q is the number of time steps.
        Values are relative abundances.
    all_water : array
        Same as `all_space_use` from compute_Z. An S X Q array of where S is 
        the number of species and Q is the number of time steps. Values are the
        time-varying probability of species being in the aquatic habitat.
    overlap : array
        An S x S array specify habitat overlaps. Ranges from 0 to 1 with diagonals
        that are always 1.
    bvals : array
        An S X Q array of where S is the number of species and Q is the number of time steps.
        Values are loss of infection rates.
    phi : float
        1 / time step. This is how the equation is parameterized in the main text
    path_decay : array
        An array of length Q giving the time varying pathogen decay rates.
    dzdt : array
        An S x (Q - 1) array. The output from get_all_dzdt.
    include_alternative_site : bool
        If True, this assumes that actual host densities are constant and all 
        fluctuations are just because they move to some terresterial site that
        we are not sampling where limited transmission happens. Default is False.


    Returns
    -------
    : An S x (Q - 1) array that holds the time-varying species-specific R0 values
    as calculcated from equation 2 in the main text.


    """
    
    R0t_vals = np.empty((all_intensity.shape[0], all_intensity.shape[1] - 1))

    # Outer species loop
    for i in np.arange(R0t_vals.shape[0]):


        if include_alternative_site:
            numerator = 1 + (phi / bvals[i, 1:]) * (1 - (all_prev[i, :-1] / all_prev[i, 1:]))
        else:
            numerator = 1 + (phi / bvals[i, 1:]) * (1 - (all_density[i, :-1] / all_density[i, 1:])*(all_prev[i, :-1] / all_prev[i, 1:]))
        denom1 = (1 - all_prev[i, 1:])
        denom2 = 0
        denom3 = 0

        # Inner species loop
        for j in np.arange(R0t_vals.shape[0]):
            denom2 = denom2 + (all_density[j, 1:] / all_density[i, 1:])*(all_water[j, 1:] / all_water[i, 1:])*(all_prev[j, 1:] / all_prev[i, 1:])*(all_intensity[j, 1:] / all_intensity[i, 1:]) * overlap[i, j]
            denom3 = denom3 + (dzdt[j, :] / (all_density[i, 1:] * all_water[i, 1:] * all_prev[i, 1:] * all_intensity[i, 1:])) * overlap[i, j]
            
        R0t_vals[i, :] = numerator / (denom1*(denom2 - denom3))
    
    return(R0t_vals)


def get_seasonal_R_matrix(S, T, R0s, ϕ, γs, λs, bs, θs, ωs):
    """
    Calculate the R matrix, the spectral radius of which is multi-species seasonal R0
    
    Parameters
    ----------
    S : int
        Number of species
    T : int
        Number of time points in a season
    R0s : array
        Length of S*T.  It is ordered as follows: R0 for all species in time 1, then time 2, etc.
        For example, for two species and three time points: R0_{11}, R0_{21}, R0_{12}, R0_{22}, R0_{13}, R0_{23}
    ϕ : float
        1 / time step between temporal points
    γs : array
        Length T.  The pathogen decay rates at each time step. Needs to be an absolute rate
    λs : array
        Length S*T.  The *relative* shedding rates for each species at different times.  Ordered the same
        way as R0s, species by time
    bs : array
        Length S*T.  The absolute loss of infection rates.  Needs to be an absolute rate.  Ordered the same way
        as R0s, species by time
    θs : array
        Length S*T. The time-varying probability of species being found in water
    ωs : array
        S * S array.  The time-invariant spatial overlap coefficients. 
    
    Returns
    -------
    : array
        R matrix to compute seasonal R0. 
    
    Notes
    -----
    The calculation uses the approach from Arino to break apart the R0 matrix. See jupyter notebook.
     
    The ordering in the R matrix is all species at time 1, then all species at time 2, etc.
    """

    # Add on phi to loss rates
    γs_new = (γs + ϕ)

    # Get phi indices by computing clockwise distance
    T1 = np.tile(np.arange(1, T + 1), T).reshape(T, T)
    T2 = T1.T
    temp1 = np.abs(T2 - T1) 
    temp2 = T - np.abs(T2 - T1)
    temp1[np.triu_indices(T)] = temp2[np.triu_indices(T)]
    temp1[np.diag_indices(T)] = 0
    phi_power = np.array(ϕ**temp1)

    # Get the gamma indices
    ind_mat = np.empty((T, T))

    # Establish the diagonal
    for i in range(T):
        include = np.delete(np.arange(0, T), i)
        ind_mat[i, i] = np.prod(γs_new[include])

    for j in range(T): # Columns

        for i in range(T): # Rows

            if i > j and temp1[i, j] != (T - 1):

                ind_mat[i, j] = np.prod(np.delete(γs_new, np.r_[np.arange(j, i + 1)]))

            elif i < j and temp1[i, j] != (T - 1):

                ind_mat[i, j] = np.prod(γs_new[np.arange(i + 1, j)])

            elif i != j:
                ind_mat[i, j] = 1

    # Expand this for multiple species. Every value should be a 2 x 2 block matrix
    combined_mat = ind_mat*phi_power

    full_rate_mat = np.empty((T, T), dtype="object")
    for i in range(T):
        for j in range(T):
            full_rate_mat[i, j] = np.repeat(combined_mat[i, j], S*S).reshape((S, S))

    # Make a nested list for block matrix
    full_rate_list = [list(full_rate_mat[i, :]) for i in range(T)]
    rate_mat = np.array(np.block(full_rate_list))

    # Fixed rate
    fr = (1 / (np.prod(γs_new) - ϕ**T))
    rate_mat = rate_mat * fr

    M1 = np.dot(np.array(np.diag(R0s * bs * (1 / (λs*θs)) * np.repeat(γs, S))), (rate_mat))
    M2 = M1 * np.tile(λs*θs, S*T).reshape((S*T, S*T))

    # Weighting from non-complete overlap
    omega_mat = np.vstack([np.tile(ωs.reshape((S, S)), T) for i in range(T)])
    M2 = M2 * omega_mat

    # Compute the B matrix
    phi_mat = ϕ * np.eye(S)
    B = np.diag((-(bs + ϕ)))

    for i in range(T):

        if i != 0:
            B[i*S:(i*S + S), (i - 1)*S:((i - 1)*S + S)] = phi_mat

        # Add upper right-hand corner
        if i == (T - 1):
            B[0:S, (T - 1)*S:((T - 1)*S + S)] = phi_mat

    # Compute R matrix        
    R = M2 @ np.linalg.inv((-B))
    
    return((R, B, M2))


def loss_of_infection_rate(log10_load):
    """
    Loss of infection rate per day as a function of load from DeMarchi et al. (2026)
    
    Parameters
    -----------
    log10_load : float

    Return
    ------
    : float
        Loss of infection rate on the per day scale
    """
    
    ln_load = np.log(10) * log10_load
    logit = -0.577094 + -0.1884483 * ln_load # From DeMarchi et al. 2026
    loss_prob = 1 / (1 + np.exp(-logit))  # Probability of recovery per time step
    loss_rate = -1*np.log(1 - loss_prob) / 6 # The time step of the IPM is 6 days
    
    return(loss_rate)


if __name__ == '__main__':


    # Load in the swab data
    full_dat = pd.read_csv("../data/all_field_samples.csv")
    full_dat = full_dat[['date', 'species', 'svl_mm', 'weight_of_animal_bag_g', 
                         'weight_of_bag', 'site_id', 'doy', 'bd_positive', 
                         'bd_load', 'microhabitat', 'site_name']]
    full_dat['date'] = pd.to_datetime(full_dat.date) 
    full_dat['mass_g'] = full_dat.weight_of_animal_bag_g - full_dat.weight_of_bag

    # Drop one clear measurement error
    full_dat = full_dat.query("bd_load < 1e7")

    keep_sites = np.array([2, 7, 10, 15]) # Focus on four sites
    species_list = {2: ['racl', 'pscr', 'raca', 'hych', 'psfe'],
                    7: ['novi', 'racl', 'raca', 'pscr'],
                    10: ['novi', 'racl', 'hych', 'pscr'],
                    15: ['novi', 'racl', 'raca', 'pscr', 'psfe', 'hych']}

    full_dat = full_dat.assign(water = lambda x: (x.microhabitat.str.contains("water") | x.microhabitat.str.contains("edge")))
    full_dat.loc[full_dat.water.isna(), "water"] = True # Set nans to True
    full_dat = full_dat.assign(water = lambda x: x.water.astype(np.int64))

    # Load in abundance data from N-mixture models
    survey_dat = pd.read_csv("../data/nmixture_abundance_estimates.csv") 
    survey_dat = survey_dat.rename(columns={'site': 'site_id', 'abund': 'count', 'doy': 'dayofyear'})

    # Set up some colors for some quick visuals qhen plotting
    cp = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', 
          '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', 
          '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', 
          '#000075', '#808080', '#ffffff', '#000000'] + list(sns.color_palette())
    spp_colors = {'racl': cp[0],
                  'rapa': cp[1],
                  'raca': cp[2],
                  'rasp': cp[3],
                  'pscr': cp[4],
                  'novi': cp[5],
                  'gaca': cp[6],
                  'psfe': cp[7],
                  'rasy': cp[8],
                  'amma': cp[9],
                  'hych': cp[10]}

    # Specify this distinction for surface area calculations
    anurans = ['racl', 'rapa', 'pscr', 'psfe', 'gaca', 'rasy', 'hych', 'raca']
    caudates = ['novi', 'amma']

    dt = 0.5 # Time step per day to calculate splines
    zoospore_decay_rate = 1 / 4.0

    # Set-up different combinations. The first combination is the 
    # default and the other combinations perform sensitivity analyses
    # The order is
    # include_microhabitat : Do you include microhabitat use in the model
    # constant_density: Keep all species density temporally constant
    # constant_water: Keep all microhabitat use temporally constant
    # constant_intensity: Keep all infection intensity temporally constant
    # constant_prev: Keep all prevalence temporally constant
    # scale_by_surface_area: Scale shedding by surface area.
    tf_combos = [(True, False, False, False, False, True),
                 (True, True, False, False, False, True),
                 (True, False, True, False, False, True),
                 (True, False, False, True, False, True),
                 (True, False, False, False, True, True),
                   (True, True, False, False, False, True),
                   (True, True, True, False, False, True),
                   (True, True, True, True, False, True),
                   (True, True, True, True, True, True)]

    # Just the default
    # tf_combos = [(True, False, False, False, False, True)]

    # Run the model for different spatial overlap coefficients, omega
    overlap_weights = [1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05]

    # Loop through overlap weights
    for overlap_weight in overlap_weights:

        # Loop through model scenarios
        for tfc in tf_combos:

            # Set up different combinations
            include_microhabitat = tfc[0]
            constant_density = tfc[1]
            constant_water = tfc[2]
            constant_intensity = tfc[3]
            constant_prev = tfc[4]
            scale_by_surface_area = tfc[5]

            all_ratios = []
            all_cvs = []
            all_R0s = []
            all_densities = []
            check_site = []

            for site in keep_sites:

                site_dat = full_dat.query("site_id == {0}".format(site)).reset_index(drop=True)
                site_dat = (site_dat.assign(date = lambda x: pd.to_datetime(x.date))
                                    .assign(month = lambda x: x.date.dt.month, 
                                            dayofyear = lambda x: x.date.dt.dayofyear,
                                            log10_bd = lambda x: np.log10(x.bd_load),
                                            bd_pos = lambda x: (x.bd_load > 0).astype(np.int64)))

                # Get the species list
                site_spp = (site_dat.groupby(['species'])
                                    .agg({'bd_load' : len,
                                          'bd_pos': np.sum})
                                    .sort_values(by=["bd_load"], ascending=False)
                                    .rename(columns={'bd_load': 'species_count'}))

                # Get median body mass
                mass_spp = (site_dat.groupby(["species"])
                                    .agg({'mass_g': np.median})
                                    .reset_index())

                # Convert to surface areas for comparison: See DeMarchi et al. 2025 for reference for equations
                mass_spp['surface_area'] = [13.1826*mass_spp.mass_g.values[i]**0.6091 if spp in anurans else 8.42*mass_spp.mass_g.values[i]**0.694 for i, spp in enumerate(mass_spp.species.values)]
                mass_spp = mass_spp.set_index("species")

                include_species = species_list[site] 

                if len(include_species) > 1:

                    ## Fit the prevalence splines, P^*_s(\tau)

                    plot_it = True # Plot the splines to visualize the relationships
                    Cvals = [0.5, 1, 2] # Inverse regularization. Explore a range
                    n_knots = [4, 5, 6] # Explore range of knots
                    
                    if constant_prev:
                        Cvals = [0.0001] # Highly regularize to keep this constant
                    else:
                        Cvals = Cvals
                    prev_results = get_prevalence_splines(site_dat, C=Cvals, dt=dt, 
                                                          include_species=include_species,
                                                          response_variable='bd_pos',
                                                          plot_it=plot_it, n_knots=n_knots)

                    ## Fit the probability of a species being in the water, \theta_s(\tau)
        
                    if constant_water:
                        Cvals = [0.0001] # Highly regularize to keep this constant
                    else:
                        Cvals = Cvals
                    water_results = get_prevalence_splines(site_dat, C=Cvals, dt=dt, 
                                                          include_species=include_species,
                                                          response_variable='water',
                                                          plot_it=plot_it, n_knots=n_knots)

                    ## Fit the load splines, $\lambda_s(\tau)

                    # For site 10, Gray tree frogs have no load detected. So use 
                    # load from other sites
                    if site == 10:
                        fixed_mean = 1.5
                    else:
                        fixed_mean = None

                    if constant_intensity:
                        alpha_vals = [10000]
                    else:
                        alpha_vals = [0.5, 1, 2] # Non-inverse regularization
                    load_results = get_load_splines(site_dat, alpha=alpha_vals, dt=dt, 
                                                    include_species=include_species,
                                                    plot_it=plot_it, n_knots=n_knots,
                                                    fixed_mean=fixed_mean)

                    ## Fit the count splines $N_s(\tau)$

                    site_count = survey_dat.query("site_id == {0}".format(site))
            
                    # Abundance can be highly variable, let's allow for that
                    if constant_density:
                        alpha_vals = [10000]
                    else:
                        alpha_vals = [0.05, 0.1, 0.25, 0.3, 1]
                    n_knots = [4, 5, 6]
                    count_results = get_abundance_splines(site_count, alpha=alpha_vals, dt=dt, 
                                                          include_species=include_species,
                                                          plot_it=plot_it, n_knots=n_knots)


                    ## Calculate time-varying R0

                    surface_area = mass_spp.loc[include_species, :].surface_area.values
                    all_prev = np.vstack([prev_results[spp]['jd_effect'] for spp in include_species])
                    all_water = np.vstack([water_results[spp]['jd_effect'] for spp in include_species])
                    all_intensity_log10 = np.vstack([load_results[spp]['jd_effect'] for spp in include_species])
                    if scale_by_surface_area:
                        all_intensity = (10**all_intensity_log10) * surface_area[:, np.newaxis] # Scale by surface area
                    else:
                        all_intensity = (10**all_intensity_log10) # Note we are not accounting for variance here.
                    all_density = np.vstack([count_results[spp]['jd_effect'] for spp in include_species])

                    # Assume that hosts are always at there max density
                    max_vals = all_density.max(axis=1)
                    all_density_max = np.vstack([np.repeat(x, all_density.shape[1]) for x in max_vals])


                    # Alternative way to calculate bvals: Fix loss of infection rate for three species 
                    # bvals_load_dependent = loss_of_infection_rate(all_intensity_log10)
                    # bvals = np.vstack([bvals_load_dependent[i, :] if spp not in ['psfe', 'pscr', 'hych'] else np.repeat(1 / 6, all_prev.shape[1]) for i, spp in enumerate(include_species)])
                    bvals = loss_of_infection_rate(all_intensity_log10)

                    phi = 1 / dt # 1 / Average time step (per day)
                    time_steps = np.arange(1, 366, step=dt)

                    # Save the site-specific values for plotting
                    pd.to_pickle({'site': site, 'species': include_species, 
                                  'prev': all_prev,
                                  'intensity': all_intensity_log10, # Don't pass in the scaled intensity
                                  'density': all_density,
                                  'water': all_water,
                                  'surface_area': surface_area,
                                  'loss_rates' : bvals, 
                                  'time_steps': time_steps}, "../results/all_values_{0}_results_microhabitat={1}_surface_area={2}_constant_density={3}_constant_water={4}_constant_intensity={5}_constant_prev={6}.pkl".format(site, include_microhabitat, scale_by_surface_area, constant_density, constant_water, constant_intensity, constant_prev))

                    # Constant pathogen decay
                    path_decay = np.repeat(zoospore_decay_rate, all_prev.shape[1]) 

                    # If False, all species are always in the water with probability 1
                    if include_microhabitat:
                        updated_water = all_water
                    else:
                        updated_water = np.repeat(1, len(all_water))

                    # Compute dz/dt from the observed data
                    all_dzdt = get_all_dzdt(all_intensity,
                                            all_prev,
                                            all_density,
                                            updated_water,
                                            path_decay, dt, 
                                            period=365, num_prior_periods=2)


                    # Compute overlaps. These are temporally fixed and the same for all pairs
                    overlap = np.eye(len(include_species)).astype(float) # Diagonals are 1
                    overlap[np.tril_indices_from(overlap, k=-1)] = overlap_weight
                    overlap[np.triu_indices_from(overlap, k=1)] = overlap_weight

                    # Get the species-level, time-varying R0 values
                    R0t_vals = get_species_level_seasonal_R0_full(
                                         all_intensity, all_prev, 
                                         all_density, 
                                         updated_water,
                                         overlap,
                                         bvals, 
                                         phi, path_decay, all_dzdt,
                                         include_alternative_site=False) # If this is True, you can test whether constant abundance and fluctuating availability changes results


                    if np.any(R0t_vals < 0):
                        check_site.append((site))

                    # Note: There are a few a situations where the rapidity of 
                    # change in R0 for upland chorus frogs and peepers
                    # leads to dips in R0 below 0.  This is only the case for 
                    # certain loss of infection rates for a few sites. Our results
                    # are not sensitive to this so we set these R0 values to 0.
                    R0t_vals[R0t_vals < 0] = 0

                    # Get variability in R0t values through time
                    means = R0t_vals.mean(axis=1)
                    sds = R0t_vals.std(axis=1)
                    cvs = sds / means
                    all_cvs.append(pd.DataFrame({'cv': cvs, 
                                                 'species': include_species,
                                                 'site': site}))

                    # Save the time-varying R0 values per species and site
                    include_species_rep = np.repeat(include_species, R0t_vals.shape[1])
                    jd = np.tile(time_steps[1:], R0t_vals.shape[0])
                    flat_R0t = R0t_vals.ravel()
                    all_R0s.append(pd.DataFrame({'R0': flat_R0t,
                                                 'julian_date': jd,
                                                 'species': include_species_rep,
                                                 'site' : site}))

                    # Save the time-varying densities values per species and site
                    include_species_rep = np.repeat(include_species, all_density.shape[1])
                    jd = np.tile(time_steps[:], all_density.shape[0])
                    flat_density = (all_density * all_water**2).ravel()
                    all_densities.append(pd.DataFrame({'density': flat_density,
                                                       'julian_date': jd,
                                                       'species': include_species_rep,
                                                       'site' : site}))

                    ## Plot time-varying R0 vals for visual checks

                    colsums = R0t_vals.sum(axis=0)
                    norm_R0t = (R0t_vals / colsums)
                    
                    # Make a stack plot
                    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
                    axes = axes.ravel()
                    tcolors = [spp_colors[spp] for spp in include_species] #sns.color_palette()[:len(include_species)]

                    axes[0].stackplot(time_steps[1:], norm_R0t, alpha=0.5, labels=include_species, colors=tcolors)
                    axes[0].set_xlabel("Julian date")
                    axes[1].set_xlabel("Julian date")
                    axes[0].set_ylabel("Proportional contribution to endemicity")
                    
                    norm_R0_cs = (norm_R0t*dt).cumsum(axis=1)
                    for j in range(norm_R0_cs.shape[0]):
                        axes[1].plot(time_steps[1:], norm_R0_cs[j, :], color=tcolors[j], label=include_species[j])
                        axes[2].plot(time_steps[1:], R0t_vals[j, :], color=tcolors[j])
                    axes[1].set_ylabel("Cumulative seasonal contribution\nto Bd persistence")
                    axes[1].set_title(site)
                    axes[1].legend(loc="upper left")

                    axes[2].set_xlabel("Julian Date")
                    axes[2].set_ylabel("Relative R0")
                    
                    plt.tight_layout()
                    plt.savefig("../results/{0}_results_microhabitat={1}_surface_area={2}_constant_density={3}_constant_water={4}_constant_intensity={5}_constant_prev={6}_omega={7}.png".format(site, include_microhabitat, scale_by_surface_area, constant_density, constant_water, constant_intensity, constant_prev, overlap_weight), bbox_inches="tight")
                    plt.close("all")

                    # Compute baseline R0 community only for the default situation
                    if not (constant_density or constant_water or constant_prev or constant_intensity):

                        R0s = R0t_vals.T.ravel()
                        S = len(include_species)
                        T = R0t_vals.shape[1]
                        ϕ = 1 / dt
                        γs = path_decay[1:]
                        λs = all_intensity[:, 1:].T.ravel()
                        bs = bvals[:, 1:].T.ravel()
                        θs = updated_water[:, 1:].T.ravel()

                        # Baseline seasonal R0
                        print("Computing baseline R0...")
                        R = get_seasonal_R_matrix(S, T, R0s, ϕ, γs, λs, bs, θs, overlap.ravel())[0]

                        # Save the R matrix
                        pd.to_pickle((R, jd, S, T, include_species), "../results/R_matrix_site={0}_microhabitat={1}_surface_area={2}_constant_density={3}_constant_water={4}_constant_intensity={5}_constant_prev={6}_omega={7}.pkl".format(site, include_microhabitat, scale_by_surface_area, constant_density, constant_water, constant_intensity, constant_prev, overlap_weight))

            all_R0s_df = pd.concat(all_R0s)
            all_R0s_df.to_csv("../results/all_R0s_microhabitat={0}_surface_area={1}_constant_density={2}_constant_water={3}_constant_intensity={4}_constant_prev={5}_omega={6}.csv".format(include_microhabitat, scale_by_surface_area, constant_density, constant_water, constant_intensity, constant_prev, overlap_weight), index=False)







