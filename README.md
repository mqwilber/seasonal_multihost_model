# Contents of seasonal_multihost repository

The repository contains the scripts and data needed to reproduce the results described in the manuscript "Spillover dynamics and asynchronous temporal fluctuations in host density and competence drive parasite persistence in multi-host seasonal communities".  Below we describe the directory structure and subsequent scripts and data.

The repository contains both R and Python code.  To run the Python code, build the conda environment specified by the `environment.yml` file.  See [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#) for details on building a conda environment. The R script requires cmdstanr.  This requires a Stan installation.  See [here](https://mc-stan.org/install/) to get Stan and cmdstanr installed on your machine.

When running the code, run `seasonal_R0_analysis_by_site.py` before running `manuscript_plots.ipynb`.

- `data/`:  The `data` folder contains raw and model-derived data used for the analyses
	- `ves_dataframe.csv`: The visual encounter data used for the n-mixture model analyses.  Three visual encounter surveys were performed for each sampling event over three days.  The columns are
		- `site_id`: The site number.  For reference to the main text: {2: 'Woodlot', 7: 'Lotus', 10: 'Complex', 15: "Hastie"}
		- `sampling`: Unique visit identifier structured as "site.visit.survey"
		- `hych`, `novi`, `pscr`, `psfe`, `raca`, `racl`: The observed abundances of each species over the visit. 
		- `survey_time`: The duration of the VES in seconds
		- `avg_temp`: The average temperature during the visit
		- `doy_ave`: The average day of year over the 2-4 visits for a survey
		- `ves_min_since_sunset`: Minutes since sunset that the VES began
		- `humidity_percet`: The humidity during the VES
		- `precipitation_accumulation_cm`: The precipitation accumulation during the day of the survey
		- `visit`: The visit number to the site. Ranges from 1-15 or 16 per site. 
		- `surv_num`: The survey number within a visit
		- `area`: The area that was surveyed
		- `survey_effort`: The seconds per m^2 that a site was surveyed. survey_time / area
		- `sample_set`: Has the format `site_id`_`visit`
	- `ves_observations.csv`: An alternative, raw representation of the VES data that is used to explore empirical values for spatial overlap. Columns are
		- `species`: The species code
		- `number_of_individuals`: Number of individuals seen at the particular location when the abundance point was taken (individuals within a meter of observer).
		- `longitude_ves`, `latitude_ves`: The longitude and latitude of the where the individual(s) were observed.
		- `site_id`: The site number where individuals were observed
		- `date`: The date where individuals were observed
	- `survey_data`: The dates of surveys/visits across all sites in east TN.  Used to show the temporal span of the study
		- `survey_id`: Unique ID for survey 
		- `site_id`: Unique numeric identifier for site
		- `date`: Date of the survey/visit
	- `pipe_summary.csv`: Summary of newt abundance as calculated from pipe sampling at site_id 7 and 15. Key columns are
		- `newt_abund2`: The area corrected newt abundance at a site
		- `doy`: THe day of year that corresponds to the newt abundance.
	- `pipe_dataframe.csv`: The locations of pipe samples through time and corresponding newt counts for the pipes.  `pipe_summary.csv` is summarize from this file and this file is used to help estimate omega. Columns are
		- `site_id`: The site id where the pipe samples were taken
		- `date`: Date of the pipe sample
		- `newt_count`: Number of newts counted in the pipe
		- `latitude_pipe`: Latitude of the pipe
		- `longitude_pipe`: The longitude of the pipe
	- `all_field_samples.csv`: The Bd samples for all amphibians across all sites.  Relevant columns are as given as follows
		- `site_id`: The site identifier where {2: 'Woodlot', 7: 'Lotus', 10: 'Complex', 15: "Hastie"}.
		- `microhabitat`: The microhabitat where an animal was found.
		- `species`: The four letter species id. first two letters are genus name and second two letters are species name.
		- `weight_of_animal_bag_g`: Weight of animal in the bag
		- `weight_of_bag`: Weight of just the bag. The difference between this column and `weight_of_animal_bag_g` gives the mass in grams of the animal.
		- `date`: The date when the animal was swabbed
		- `bd_load`: The number of zoospore equivalents (not genomic equivalents) detected on the swab after applying standard curve that links CT value and zoospore counts based on a local TN Bd strain.
	- `nmixture_abundance_estimates.csv`: The median, time-varying abundance estimates from N-mixture models or pipe sampling for all species across all sites.  Columns are
		- `abund`: The estimated abundance of the species
		- `site` : The site identifier where {2: 'Woodlot', 7: 'Lotus', 10: 'Complex', 15: "Hastie"}.
		- 'species': The four letter code for host species. The first two letters are genus name and second two letters are species name.
		- 'doy': The day of the year corresponding to the abundance estimate.
- `code/`: The code folder contains the scripts necessary to reproduce all of the results in the main text.  The scripts are described below
	- `model_derivation_and_examples.ipynb`: Jupyter notebook shows the derivation of the model and provides some toy examples to test that the model works as expected.
	- `seasonal_R0_analysis_by_site.py`: This script computes the seasonal R_0 and R_{0, T} for all species and sites for different levels of omega.  Internal functions contained in this script translate the model into code.
	- `nmixture_model_analysis.R`: This script contains code to run the Bayesian N-mixture models used to estimated host abundance.
	- `manuscript_plots.ipynb`: This Jupyter notebook makes all of the plots for the study and performs the control analyses for Q2.
	- `calculate_empirical_omega.ipynb`: This Jupyter notebook provides coarse calculations for the possible empirical range of omega values (i.e., spatial overlap values) of the six species used in the paper.
- `results/`: Output from scripts are stored in this folder
