
# Load packages ----

required_Packages_Install <- c("rethinking", "bayesplot", "data.table", "cmdstanr","GGally","ggplot2", "dplyr")

for(Package in required_Packages_Install){
  if(!require(Package,character.only = TRUE)) { 
    install.packages(Package, dependencies=TRUE)
  }
  library(Package,character.only = TRUE)
}

# Load datasets ----

ves = read.csv("../data/ves_dataframe.csv")

dat = as.data.frame(ves)


# Prepare data ----

## Scale data ----

dat$site = factor(dat$site_id)

# Scale continuous covariates
dat$survey_effort_z = as.vector(scale(dat$survey_effort)) #seconds per m^2
dat$doy_z =as.vector(scale(dat$doy_ave))
dat$temp_z =as.vector(scale(dat$avg_temp))
dat$humid_z =as.vector(scale(dat$humidity_percent))
dat$mins_sunset_z = as.vector(scale(dat$ves_min_since_sunset))

# Prep quadratic term for doy
dat$doy2 = (dat$doy_ave)^2
dat$doy2_z = as.vector(scale(dat$doy2))

# Not all site-batch combinations were surveyed the same number of times
max_surveys = max(table(dat$sample_set))
M = length(unique(dat$sample_set))


## Assign arrays ----

### Species count arrays ----
# Assign a -1 to start. All surveys with no counts will be minus one. Helpful for Stan that doesn't like ragged arrays
obs_mat_racl = array(-1, dim=c(max_surveys, M))
obs_mat_raca = array(-1, dim=c(max_surveys, M))
obs_mat_pscr = array(-1, dim=c(max_surveys, M))
obs_mat_psfe = array(-1, dim=c(max_surveys, M))
obs_mat_novi = array(-1, dim=c(max_surveys, M))
obs_mat_hych = array(-1, dim=c(max_surveys, M))

### Detection covariate arrays ----
survey_effort = array(-1, dim=c(max_surveys, M))
temperature = array(-1, dim=c(max_surveys, M))
humidity = array(-1, dim=c(max_surveys, M))
mins_after_sunset = array(-1, dim=c(max_surveys, M))

### Abundance covariate arrays ----
doy = array(NA, dim=c(M))
doy2 = array(NA, dim=c(M))
location = array(NA, dim=c(M))


## Fill in Data ----

# Loop through and set everything up
unq_sys = unique(dat$sample_set)

for(m in 1:length(unq_sys)){
  tdat = dat[dat$sample_set == unq_sys[m],]
  tn = nrow(tdat)
  
  # Set-up observed counts
  obs_mat_racl[1:tn, m] = tdat[["racl"]]
  obs_mat_raca[1:tn, m] = tdat[["raca"]]
  obs_mat_pscr[1:tn, m] = tdat[["pscr"]]
  obs_mat_psfe[1:tn, m] = tdat[["psfe"]]
  obs_mat_novi[1:tn, m] = tdat[["novi"]]
  obs_mat_hych[1:tn, m] = tdat[["hych"]]
  
  # Set-up detection probability covariates
  survey_effort[1:tn, m] = tdat$survey_effort_z
  temperature[1:tn, m] = tdat$temp_z
  humidity[1:tn, m] = tdat$humid_z
  mins_after_sunset[1:tn, m] = tdat$mins_sunset_z
  
  # Set-up abundance covariates
  doy[m] = unique(tdat$doy_z)
  doy2[m] = unique(tdat$doy2_z)
  location[m] = unique(tdat$site)
}

# Set-up max abundance
max_obs_racl = apply(obs_mat_racl, 2, max)
max_obs_raca = apply(obs_mat_raca, 2, max)
max_obs_pscr = apply(obs_mat_pscr, 2, max)
max_obs_psfe = apply(obs_mat_psfe, 2, max)
max_obs_novi = apply(obs_mat_novi, 2, max)
max_obs_hych = apply(obs_mat_hych, 2, max)

# Site-up location/site IDs 
site_id = coerce_index(as.factor(location))

# Create dummy variables for each site
site_wh <- mapply(function(x) if (x==4) (1) else 0, site_id)
site_cw <- mapply(function(x) if (x==1) (1) else 0, site_id)
site_ij1 <- mapply(function(x) if (x==2) (1) else 0, site_id)
site_ij2 <- mapply(function(x) if (x==3) (1) else 0, site_id)


# Stan Lists ----

#stan data for racl models
stan_data_racl = list(M=M,
                      J=max_surveys,
                      obs=obs_mat_racl,
                      max_obs=max_obs_racl,
                      site_cw=site_cw,
                      site_wh=site_wh,
                      site_ij1=site_ij1,
                      site_ij2=site_ij2,
                      survey_effort=survey_effort,
                      temperature=temperature,
                      doy=doy,
                      doy2=doy2,
                      Nmax=750)

#stan data for raca models
stan_data_raca = list(M=M,
                      J=max_surveys,
                      obs=obs_mat_raca,
                      max_obs=max_obs_raca,
                      site_cw=site_cw,
                      site_wh=site_wh,
                      site_ij1=site_ij1,
                      site_ij2=site_ij2,
                      survey_effort=survey_effort,
                      temperature=temperature,
                      doy=doy,
                      doy2=doy2,
                      Nmax=100)

#stan data for pscr models
stan_data_pscr = list(M=M,
                      J=max_surveys,
                      obs=obs_mat_pscr,
                      max_obs=max_obs_pscr,
                      site_cw=site_cw,
                      site_wh=site_wh,
                      site_ij1=site_ij1,
                      site_ij2=site_ij2,
                      survey_effort=survey_effort,
                      temperature=temperature,
                      doy=doy,
                      doy2=doy2,
                      Nmax=1000)

#stan data for psfe models
stan_data_psfe = list(M=M,
                      J=max_surveys,
                      obs=obs_mat_psfe,
                      max_obs=max_obs_psfe,
                      site_cw=site_cw,
                      site_wh=site_wh,
                      site_ij1=site_ij1,
                      site_ij2=site_ij2,
                      survey_effort=survey_effort,
                      temperature=temperature,
                      doy=doy,
                      doy2=doy2,
                      Nmax=1000)

#stan data for novi models
stan_data_novi = list(M=M,
                      J=max_surveys,
                      obs=obs_mat_novi,
                      max_obs=max_obs_novi,
                      site_cw=site_cw,
                      site_wh=site_wh,
                      site_ij1=site_ij1,
                      site_ij2=site_ij2,
                      survey_effort=survey_effort,
                      temperature=temperature,
                      doy=doy,
                      doy2=doy2,
                      Nmax=1000)

#stan data for hych models
stan_data_hych = list(M=M,
                      J=max_surveys,
                      obs=obs_mat_hych,
                      max_obs=max_obs_hych,
                      site_cw=site_cw,
                      site_wh=site_wh,
                      site_ij1=site_ij1,
                      site_ij2=site_ij2,
                      survey_effort=survey_effort,
                      temperature=temperature,
                      doy=doy,
                      doy2=doy2,
                      Nmax=500)


# Stan Models ----

## Model 1 ----

### Model ----

mod_nb_1 = "
data { 
  int<lower=1> J; // Number of surveys per site
  int<lower=1> M; // Number of sites
  array[J, M] int obs; // Observed abundance surveys
  array[M] int max_obs; // Maximum abundance per site
  
  //Abundance
  array[M] real site_wh; // Binary variable (0 or 1) to indicate site William Hastie
  array[M] real site_cw; // Binary variable (0 or 1) to indicate site Cherokee Woodlot 
  array[M] real site_ij1; // Binary variable (0 or 1) to indicate site Ijams Lotus 
  array[M] real site_ij2; // Binary variable (0 or 1) to indicate site Ijams Cmplx3
  array[M] real doy; // Day of the year
  
  //Detection
  array[J,M] real survey_effort; // survey effort
  
  int Nmax; // Upper bound of the marginalization

} 
parameters {

  real beta0; // Detection prob
  real gamma1; // mean abundance at CW
  real gamma2; // mean abundance at IJ Lotus
  real gamma3; // mean abundance at IJ cmplx 3
  real gamma4; // mean abundance at WH
  
  real beta_se; // Effect of effort

  real gamma_doy; // Effect of day of the year on mean abundance among sites
  real gamma_doy2; // Quadratic effect of day of the year on mean abundance among sites
  real<lower=0> phi; // Dispersion parameter

  
} model {

  real log_negbinom; // Hold the poisson log-likelihood
  real ll_within_survey; // Hold likelihood within survey
  real lik_marg; // Hold the marginalized likelihood
  
  // Priors
  beta0 ~ normal(0, 2);
  beta_se ~ normal(0,1.5);
  gamma1 ~ normal(1, 2);
  gamma2 ~ normal(1, 2);
  gamma3 ~ normal(1, 2);
  gamma4 ~ normal(1, 2);
  gamma_doy ~ normal(0, 2);
  gamma_doy2 ~ normal(0, 2);
  phi ~ normal(1,2);
  

  // Loop over sites
  for(m in 1:M){

      lik_marg = 0;

      // Marginalization
      for(n in max_obs[m]:Nmax){

        log_negbinom = neg_binomial_2_log_lpmf(n | gamma1*site_cw[m] + gamma2*site_ij1[m] + gamma3*site_ij2[m] + gamma4*site_wh[m] + gamma_doy*doy[m] + gamma_doy2*(doy[m])^2, phi);

        ll_within_survey = 0;

        // Loop through replication within a site
        for(j in 1:J){
        
          // Accounts for different number of observations per site
          if(obs[j][m] != -1){
          
            // Likelihood within a survey
            ll_within_survey += binomial_logit_lpmf(obs[j][m] | n, beta0 + beta_se*survey_effort[j][m]);
          }

        } // End survey loop

        lik_marg += exp(ll_within_survey + log_negbinom);

    } // End marginalization loop

    target += log(lik_marg);
  }
} generated quantities {

  // Holds the unnormalized [Ni = k] for each site M
  matrix[Nmax + 1, M] Kmat;
  vector[M] log_lik;
  
  // Temporary variables
  real<lower=0> negbinom_pmf; 
  real<lower=0> prob_within_survey;
  real log_negbinom; // Hold the negbinom log-likelihood
  real ll_within_survey; // Hold likelihood within survey
  real lik_marg; // Hold the marginalized likelihood
  
  
  // Loop over sites
  for(m in 1:M){

      lik_marg = 0;

      // Marginalization
      for(n in max_obs[m]:Nmax){

        log_negbinom = neg_binomial_2_log_lpmf(n |gamma1*site_cw[m] + gamma2*site_ij1[m] + gamma3*site_ij2[m] + gamma4*site_wh[m] + gamma_doy*doy[m] + gamma_doy2*(doy[m])^2,phi );

        ll_within_survey = 0;

        // Loop through replication within a site
        for(j in 1:J){
        
            // Accounts for different number of observations per site
          if(obs[j][m] != -1){
            
            // Likelihood within a survey
            ll_within_survey += binomial_logit_lpmf(obs[j][m] | n, beta0 + beta_se*survey_effort[j][m]);
          }

        } // End survey loop

        lik_marg += exp(ll_within_survey + log_negbinom);

    } // End marginalization loop

    log_lik[m] = log(lik_marg);
  }

  // Loop over different sampling times
  for(m in 1:M){

    for(k in 0:Nmax) {
      
      // Probability of seeing abundance n given gamma0
      negbinom_pmf = exp(neg_binomial_2_log_lpmf(k | gamma1*site_cw[m] + gamma2*site_ij1[m] + gamma3*site_ij2[m] + gamma4*site_wh[m] + gamma_doy*doy[m] + gamma_doy2*(doy[m])^2, phi));

      // Loop over surveys
      prob_within_survey = 1;

      for(j in 1:J){

        // If the predicted abundance is less than maximum obs observed set probability to 0
        if(k < max_obs[m]){
          prob_within_survey *= 0;
        } else{
        if(obs[j][m] != -1){
          prob_within_survey *= exp(binomial_logit_lpmf(obs[j][m] | k, beta0 + beta_se*survey_effort[j][m]));
        }
      }

      } // End survey replicate loop
      
      // The unnormalized probability of site having abundance
      // k (k + 1 because Stan uses 1 indexing)
      Kmat[k + 1, m] = negbinom_pmf * prob_within_survey;  

    } // End k loop
  } // End site loop
} 
"
write_stan_file(mod_nb_1, dir=".", basename="nmix_nb_1")


### Compile ----
nmixture_nb_1 = cmdstan_model("nmix_nb_1.stan")


### Fit ----

fit_nb_1_racl = nmixture_nb_1$sample(data=stan_data_racl,
                                           chains=4,
                                           iter_warmup=500,
                                           iter_sampling=1000,
                                           parallel_chains=4)

fit_nb_1_raca = nmixture_nb_1$sample(data=stan_data_raca,
                                           chains=4,
                                           iter_warmup=500,
                                           iter_sampling=1000,
                                           parallel_chains=4)

fit_nb_1_pscr = nmixture_nb_1$sample(data=stan_data_pscr,
                                           chains=4,
                                           iter_warmup=500,
                                           iter_sampling=1000,
                                           parallel_chains=4)

fit_nb_1_psfe = nmixture_nb_1$sample(data=stan_data_psfe,
                                           chains=4,
                                           iter_warmup=500,
                                           iter_sampling=1000,
                                           parallel_chains=4)

fit_nb_1_novi = nmixture_nb_1$sample(data=stan_data_novi,
                                           chains=4,
                                           iter_warmup=500,
                                           iter_sampling=1000,
                                           parallel_chains=4)

fit_nb_1_hych = nmixture_nb_1$sample(data=stan_data_hych,
                                           chains=4,
                                           iter_warmup=500,
                                           iter_sampling=1000,
                                           parallel_chains=4)


### Diagnostics ----

#### RACL ----

# Extract posterior and check convergence
post_nb_1_racl = fit_nb_1_racl$draws(variables=c("beta0",  "beta_se","gamma_doy","gamma_doy2", "gamma1", "gamma2", "gamma3", "gamma4","phi"))
nb_1_trace_racl = mcmc_trace(post_nb_1_racl, pars=c("beta0", "beta_se","gamma_doy","gamma_doy2", "gamma1", "gamma2", "gamma3", "gamma4","phi"))
nb_1_rank_racl = mcmc_rank_overlay(post_nb_1_racl, pars=c("beta0", "beta_se","gamma_doy", "gamma_doy2","gamma1", "gamma2", "gamma3", "gamma4","phi"))
racl_params_1 = fit_nb_1_racl$summary(variables=c("beta0",  "beta_se","gamma_doy","gamma_doy2", "gamma1", "gamma2", "gamma3", "gamma4","phi"))

write.csv(racl_params_1,"../results/racl_params_1.csv")


#### RACA ----

# Extract posterior and check convergence
post_nb_1_raca = fit_nb_1_raca$draws(variables=c("beta0",  "beta_se","gamma_doy","gamma_doy2", "gamma1", "gamma2", "gamma3", "gamma4","phi"))
nb_1_trace_raca = mcmc_trace(post_nb_1_raca, pars=c("beta0",  "beta_se","gamma_doy","gamma_doy2", "gamma1", "gamma2", "gamma3", "gamma4","phi"))
nb_1_rank_raca = mcmc_rank_overlay(post_nb_1_raca, pars=c("beta0",  "beta_se","gamma_doy", "gamma1", "gamma2", "gamma3", "gamma4","phi"))
raca_params_1 = fit_nb_1_raca$summary(variables=c("beta0",  "beta_se","gamma_doy","gamma_doy2", "gamma1", "gamma2", "gamma3", "gamma4","phi"))

write.csv(raca_params_1,"../results/raca_params_1.csv")


#### PSCR ----

# Extract posterior and check convergence
post_nb_1_pscr = fit_nb_1_pscr$draws(variables=c("beta0",  "beta_se","gamma_doy","gamma_doy2", "gamma1", "gamma2", "gamma3", "gamma4","phi"))
nb_1_trace_pscr = mcmc_trace(post_nb_1_pscr, pars=c("beta0",  "beta_se","gamma_doy","gamma_doy2", "gamma1", "gamma2", "gamma3", "gamma4","phi"))
nb_1_rank_pscr = mcmc_rank_overlay(post_nb_1_pscr, pars=c("beta0",  "beta_se","gamma_doy", "gamma1", "gamma2", "gamma3", "gamma4","phi"))
pscr_params_1 = fit_nb_1_pscr$summary(variables=c("beta0",  "beta_se","gamma_doy","gamma_doy2", "gamma1", "gamma2", "gamma3", "gamma4","phi"))

write.csv(pscr_params_1,"../results/pscr_params_1.csv")


#### PSFE ----

# Extract posterior and check convergence
post_nb_1_psfe = fit_nb_1_psfe$draws(variables=c("beta0",  "beta_se","gamma_doy","gamma_doy2", "gamma1", "gamma2", "gamma3", "gamma4","phi"))
nb_1_trace_psfe = mcmc_trace(post_nb_1_psfe, pars=c("beta0",  "beta_se","gamma_doy","gamma_doy2", "gamma1", "gamma2", "gamma3", "gamma4","phi"))
nb_1_rank_psfe = mcmc_rank_overlay(post_nb_1_psfe, pars=c("beta0",  "beta_se","gamma_doy", "gamma1", "gamma2", "gamma3", "gamma4","phi"))
psfe_params_1 = fit_nb_1_psfe$summary(variables=c("beta0",  "beta_se","gamma_doy","gamma_doy2", "gamma1", "gamma2", "gamma3", "gamma4","phi"))

write.csv(psfe_params_1,"../results/psfe_params_1.csv")


#### NOVI ----

# Extract posterior and check convergence
post_nb_1_novi = fit_nb_1_novi$draws(variables=c("beta0",  "beta_se","gamma_doy","gamma_doy2", "gamma1", "gamma2", "gamma3", "gamma4","phi"))
nb_1_trace_novi = mcmc_trace(post_nb_1_novi, pars=c("beta0",  "beta_se","gamma_doy","gamma_doy2", "gamma1", "gamma2", "gamma3", "gamma4","phi"))
nb_1_rank_novi = mcmc_rank_overlay(post_nb_1_novi, pars=c("beta0",  "beta_se","gamma_doy", "gamma1", "gamma2", "gamma3", "gamma4","phi"))
novi_params_1 = fit_nb_1_novi$summary(variables=c("beta0",  "beta_se","gamma_doy","gamma_doy2", "gamma1", "gamma2", "gamma3", "gamma4","phi"))

write.csv(novi_params_1,"../results/novi_params_1.csv")


#### HYCH ----

# Extract posterior and check convergence
post_nb_1_hych = fit_nb_1_hych$draws(variables=c("beta0",  "beta_se","gamma_doy","gamma_doy2", "gamma1", "gamma2", "gamma3", "gamma4","phi"))
nb_1_trace_hych = mcmc_trace(post_nb_1_hych, pars=c("beta0",  "beta_se","gamma_doy","gamma_doy2", "gamma1", "gamma2", "gamma3", "gamma4","phi"))
nb_1_rank_hych = mcmc_rank_overlay(post_nb_1_hych, pars=c("beta0",  "beta_se","gamma_doy", "gamma1", "gamma2", "gamma3", "gamma4","phi"))
hych_params_1 = fit_nb_1_hych$summary(variables=c("beta0",  "beta_se","gamma_doy","gamma_doy2", "gamma1", "gamma2", "gamma3", "gamma4","phi"))

write.csv(hych_params_1,"../results/hych_params_1.csv")


### Posterior Analysis ----

#### Format ----

#unscale variables

doy_nat = doy * sd(dat$doy_ave) + mean(dat$doy_ave)
temp_nat = temperature * sd(dat$avg_temp) + mean(dat$avg_temp)
se_nat = survey_effort * sd(dat$survey_effort) + mean(dat$survey_effort)


#### Parameter plots ----

nb_1_param_plt_racl = mcmc_areas(post_nb_1_racl, prob=0.95, pars=c("beta0",  "beta_se","gamma_doy","gamma_doy2", "gamma1", "gamma2", "gamma3", "gamma4","phi"))
png(file="../results/nb_1_param_plt_racl.png",
    width=600, height=600)
print(nb_1_param_plt_racl)
dev.off()

nb_1_param_plt_raca = mcmc_areas(post_nb_1_raca, prob=0.95, pars=c("beta0",  "beta_se","gamma_doy","gamma_doy2", "gamma1", "gamma2", "gamma3", "gamma4"))
png(file="../results/nb_1_param_plt_raca.png",
    width=600, height=600)
print(nb_1_param_plt_raca)
dev.off()

nb_1_param_plt_pscr = mcmc_areas(post_nb_1_pscr, prob=0.95, pars=c("beta0",  "beta_se","gamma_doy","gamma_doy2", "gamma1", "gamma2", "gamma3", "gamma4","phi"))
png(file="../results/nb_1_param_plt_pscr.png",
    width=600, height=600)
print(nb_1_param_plt_pscr)
dev.off()

nb_1_param_plt_psfe = mcmc_areas(post_nb_1_psfe, prob=0.95, pars=c("beta0",  "beta_se","gamma_doy","gamma_doy2", "gamma1", "gamma2", "gamma3", "gamma4"))
png(file="../results/nb_1_param_plt_psfe.png",
    width=600, height=600)
print(nb_1_param_plt_psfe)
dev.off()

nb_1_param_plt_novi = mcmc_areas(post_nb_1_novi, prob=0.95, pars=c("beta0",  "beta_se","gamma_doy","gamma_doy2", "gamma1", "gamma2", "gamma3", "gamma4","phi"))
png(file="../results/nb_1_param_plt_novi.png",
    width=600, height=600)
print(nb_1_param_plt_novi)
dev.off()

nb_1_param_plt_hych = mcmc_areas(post_nb_1_hych, prob=0.95, pars=c("beta0",  "beta_se","gamma_doy","gamma_doy2", "gamma1", "gamma2", "gamma3", "gamma4"))
png(file="../results/nb_1_param_plt_hych.png",
    width=600, height=600)
print(nb_1_param_plt_hych)
dev.off()


dat_lim = dat[,c("site_id","doy_ave","sample_set")]
dat_lim=dplyr::mutate(dat_lim, ID = row_number())

dat_join = dat_lim %>%
  group_by(sample_set) %>%
  summarize(site_id=mean(site_id),doy_ave=mean(doy_ave),order=median(ID))

dat_join = dat_join[order(dat_join$order),]


### Predicted abundance ----

draw_posterior_abundances = function(mod_fit,stan_data){
  # Draw random values from the posterior distribution
  # true abundance at each site given a model. 
  #
  # Use an inverse probability transform to sample from the distributions
  #
  # Returns: d x M matrix, where d is the number of samples in the posterior and M is the number
  # of sites
  
  # Extract and format K
  post = mod_fit$draws(variables="Kmat", inc_warmup = FALSE)
  post_mat = as_draws_matrix(post)
  Kmat = array(NA, dim=c(nrow(post_mat), stan_data$Nmax + 1, M))
  for(i in 1:nrow(post_mat)){
    Kmat[i, ,] = matrix(post_mat[i, ], nrow=stan_data$Nmax + 1, ncol=M)
  }
  
  # Normalize over all matrices and compute CDF
  Knorms = array(NA, dim=dim(Kmat))
  for(i in 1:dim(Kmat)[1]){
    Knorms[i, ,] = apply(Kmat[i, ,], 2, function(x) cumsum(x / sum(x)))
  }
  
  # Generate random samples for abundances using an inverse probability transform
  draws = dim(Knorms)[1]
  N_abunds = array(NA, dim=c(draws, dim(Knorms)[3]))
  for(d in 1:draws){
    
    tK = Knorms[d, ,]
    rand = runif(ncol(tK))
    N_abunds[d, ] = sapply(1:ncol(tK), function(x) which.min(tK[, x] < rand[x]) - 1)
    
  }
  
  return(N_abunds)
  
}


#### RACL ----

abunds_racl = draw_posterior_abundances(fit_nb_1_racl,stan_data_racl)
med_abund_racl = apply(abunds_racl, 2, median)
lower_abund_racl = apply(abunds_racl, 2, quantile, 0.025)
upper_abund_racl = apply(abunds_racl, 2, quantile, 0.975)

abund_df_racl = as.data.frame(cbind(med_abund_racl,lower_abund_racl,upper_abund_racl,dat_join))
abund_df_racl$species = "racl"
colnames(abund_df_racl) = c("median","lower","upper","sample_set","site","doy","species")

write.csv(abund_df_racl,"../data/abund_df_racl.csv")


#### RACA ----

abunds_raca = draw_posterior_abundances(fit_nb_1_raca,stan_data_raca)
med_abund_raca = apply(abunds_raca, 2, median)
lower_abund_raca = apply(abunds_raca, 2, quantile, 0.025)
upper_abund_raca = apply(abunds_raca, 2, quantile, 0.975)

abund_df_raca = as.data.frame(cbind(med_abund_raca,lower_abund_raca,upper_abund_raca,dat_join))
abund_df_raca$species = "raca"
colnames(abund_df_raca) = c("median","lower","upper","sample_set","site","doy","species")

write.csv(abund_df_raca,"../data/abund_df_raca.csv")


#### PSCR ----

abunds_pscr = draw_posterior_abundances(fit_nb_1_pscr,stan_data_pscr)
med_abund_pscr = apply(abunds_pscr, 2, median)
lower_abund_pscr = apply(abunds_pscr, 2, quantile, 0.025)
upper_abund_pscr = apply(abunds_pscr, 2, quantile, 0.975)

abund_df_pscr = as.data.frame(cbind(med_abund_pscr,lower_abund_pscr,upper_abund_pscr,dat_join))
abund_df_pscr$species = "pscr"
colnames(abund_df_pscr) = c("median","lower","upper","sample_set","site","doy","species")

write.csv(abund_df_pscr,"../data/abund_df_pscr.csv")


#### PSFE ----

abunds_psfe = draw_posterior_abundances(fit_nb_1_psfe,stan_data_psfe)
med_abund_psfe = apply(abunds_psfe, 2, median)
lower_abund_psfe = apply(abunds_psfe, 2, quantile, 0.025)
upper_abund_psfe = apply(abunds_psfe, 2, quantile, 0.975)

abund_df_psfe = as.data.frame(cbind(med_abund_psfe,lower_abund_psfe,upper_abund_psfe,dat_join))
abund_df_psfe$species = "psfe"
colnames(abund_df_psfe) = c("median","lower","upper","sample_set","site","doy","species")

write.csv(abund_df_psfe,"../data/abund_df_psfe.csv")

#### NOVI ----

abunds_novi = draw_posterior_abundances(fit_nb_1_novi,stan_data_novi)
med_abund_novi = apply(abunds_novi, 2, median)
lower_abund_novi = apply(abunds_novi, 2, quantile, 0.025)
upper_abund_novi = apply(abunds_novi, 2, quantile, 0.975)

abund_df_novi = as.data.frame(cbind(med_abund_novi,lower_abund_novi,upper_abund_novi,dat_join))
abund_df_novi$species = "novi"
colnames(abund_df_novi) = c("median","lower","upper","sample_set","site","doy","order","species")

write.csv(abund_df_novi,"../data/abund_df_novi.csv")


#### HYCH ----

abunds_hych = draw_posterior_abundances(fit_nb_1_hych,stan_data_hych)
med_abund_hych = apply(abunds_hych, 2, median)
lower_abund_hych = apply(abunds_hych, 2, quantile, 0.025)
upper_abund_hych = apply(abunds_hych, 2, quantile, 0.975)

abund_df_hych = as.data.frame(cbind(med_abund_hych,lower_abund_hych,upper_abund_hych,dat_join))
abund_df_hych$species = "hych"
colnames(abund_df_hych) = c("median","lower","upper","sample_set","site","doy","species")

write.csv(abund_df_hych,"../data/abund_df_hych.csv")

#### Combine abundance estimates ----

# Read in again so to avoid re-running
abund_df_racl = read.csv("../data/abund_df_racl.csv")
abund_df_raca = read.csv("../data/abund_df_raca.csv")
abund_df_pscr = read.csv("../data/abund_df_pscr.csv")
abund_df_psfe = read.csv("../data/abund_df_psfe.csv")
abund_df_novi = read.csv("../data/abund_df_novi.csv")
abund_df_hych = read.csv("../data/abund_df_hych.csv")

abund_df = rbind(abund_df_racl,abund_df_raca,abund_df_pscr,abund_df_psfe,abund_df_novi,abund_df_hych)
abund_df$site_spec = as.factor(paste0(abund_df$species,"_",abund_df$site))

ggplot() + geom_point(aes(x=doy,y=median,color=species),data=abund_df) + theme_bw() + facet_wrap(abund_df$site)

abund_dataframe = as.data.frame(abund_df)

write.csv(abund_dataframe,"../data/abund_dataframe.csv")

#### Add in pipe sampling for newts ----

### Pipe ----
pipe_summary_df = read.csv("../data/pipe_summary_df.csv")
pipe_summary_df = pipe_summary_df[,c("site_id","doy","newt_abund2")] %>% rename(
  site = site_id,
  abund = newt_abund2
)
pipe_summary_df$species = "novi"
pipe_summary_df$site_spec = paste0(pipe_summary_df$species,"_",pipe_summary_df$site)

### N-mix Abundance ----
abund_dataframe = read.csv("../data/abund_dataframe.csv")
abund_dataframe = abund_dataframe[,c("median","site","species","doy","site_spec")] %>% rename(
  abund = median
)
abund_dataframe = abund_dataframe[abund_dataframe$site_spec != "novi_15" & abund_dataframe$site_spec != "novi_7",]
abund_gam = rbind(abund_dataframe,pipe_summary_df)
abund_gam = abund_gam[,names(abund_gam)!="site_spec"]

write.csv(abund_gam,"../data/nmixture_abundance_estimates.csv",row.names=FALSE)

