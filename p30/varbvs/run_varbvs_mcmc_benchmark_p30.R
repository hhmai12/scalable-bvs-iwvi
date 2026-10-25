########################################################################
# Author: Matias Quiroz
# Institution: University of Technology Sydney
#
# Purpose:
# Benchmark the mean-field variational PIPs from varbvs
# (Carbonetto and Stephens, 2012) against a componentwise collapsed Gibbs
# sampler targeting the SAME posterior conditional on the hyperparameter
# values used by varbvs.
#
# Moderate-dimensional PIP-accuracy experiment:
#   R = 50 replicates
#   n = 1000 observations
#   p = 30 candidate predictors
#   5 truly active predictors
#
# Key isolation:
#   - varbvs is fitted first on each dataset.
#   - For every varbvs hyperparameter setting, sigma, sa and logodds are
#     held fixed at the values used by varbvs.
#   - The varbvs hyperparameter weights are kept fixed.
#   - beta is integrated out exactly in the Gibbs sampler.
#   - The only approximation being replaced is the mean-field
#     approximation to the inclusion indicators.
#
# Therefore, the main comparison is
#
#   PIP_varbvs versus PIP_Gibbs
#
# where both PIP vectors use exactly the same varbvs hyperparameter
# weighting scheme.
#
# The varbvs parameterisation for family = "gaussian" is
#
#   y = beta0 * 1 + X beta + epsilon,
#   epsilon ~ N(0, sigma I),
#
#   beta_j | gamma_j = 1, sigma, sa ~ N(0, sigma * sa),
#
# with an intercept included automatically. Here sigma is the residual
# VARIANCE, not the residual standard deviation.
#
# The intercept has a flat prior. In the matched Gibbs sampler it is
# integrated out exactly by centring y and each column of X.
#
# Settings used in the thesis (Table 4.3 varbvs row, Appendix D):
#   31 prior log-odds settings -3, -2.9, ..., 0; varbvs tol = 1e-6;
#   4 Gibbs chains per setting, 2,000 burn-in and 20,000 retained sweeps.
# The run is resumable: varbvs fits and finished replicates are cached and
# reused, but only when they were produced with exactly these settings.
#
# Usage (from any working directory):
#   Rscript p30/varbvs/run_varbvs_mcmc_benchmark_p30.R
# or source() the file in RStudio. The data are read from p30/data_new/.
# Outputs go to p30/varbvs/output_p30/, and the PIP table used by the
# post-processing notebook is written to
#   p30/varbvs/varbvs_vs_gibbs_pips_by_variable.csv
########################################################################

rm(list = ls())

library(varbvs)

########################################################################
# 1. User controls
########################################################################

EXPECTED_VARBVS_VERSION <- "2.6.10"

# The script automatically looks for the 50 seed files and the truth file
# below this script's folder, its parent folder (p30/, which holds
# data_new/) and the working directory.
#
# If the CSV files are not found but data_new.zip is found somewhere under
# the project directory, the archive is extracted beside the zip file and
# the search is repeated.

# Determine the directory containing this script. This means the script can
# be sourced with an absolute path from any RStudio working directory.
get_script_dir <- function() {
  ofiles <- vapply(
    sys.frames(),
    function(fr) {
      if (!is.null(fr$ofile) && length(fr$ofile) == 1L) {
        as.character(fr$ofile)
      } else {
        NA_character_
      }
    },
    character(1)
  )

  ofiles <- ofiles[!is.na(ofiles) & nzchar(ofiles)]

  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(ofiles) == 0L && length(file_arg) > 0L) {
    ofiles <- sub("^--file=", "", file_arg[1L])
  }

  if (length(ofiles) > 0L) {
    dirname(normalizePath(tail(ofiles, 1L), mustWork = TRUE))
  } else {
    normalizePath(getwd(), mustWork = TRUE)
  }
}

project_dir <- get_script_dir()
output_dir <- file.path(project_dir, "output_p30")
mcmc_output_dir <- file.path(output_dir, "varbvs_mcmc_benchmark")
varbvs_cache_dir <- file.path(output_dir, "varbvs_fit_cache")

# Resume controls.
# - Cached varbvs fits are reused whenever available.
# - Completed replicate-level benchmark files are reused as well, so an
#   interrupted long run can simply be sourced again and will continue.
REUSE_CACHED_VARBVS_FITS <- TRUE
REUSE_COMPLETED_REPLICATES <- TRUE

expected_seed_basenames <- sprintf("seed_%02d.csv", 0:49)
truth_basename <- "fixed_beta_and_active_set.csv"

# Return the first directory under `root` that contains all 50 seed files
# and the truth file. Prefer shallower directories so that a natural project
# data folder is chosen before deeply nested duplicates.
find_complete_data_dir <- function(root) {
  if (!dir.exists(root)) return(NULL)

  dirs <- unique(c(root, list.dirs(root, recursive = TRUE, full.names = TRUE)))
  depths <- lengths(strsplit(normalizePath(dirs, mustWork = FALSE), .Platform$file.sep, fixed = TRUE))
  dirs <- dirs[order(depths, nchar(dirs))]

  for (d in dirs) {
    seeds_ok <- all(file.exists(file.path(d, expected_seed_basenames)))
    truth_ok <- file.exists(file.path(d, truth_basename))
    if (seeds_ok && truth_ok) return(normalizePath(d, mustWork = TRUE))
  }

  NULL
}

# Search below the script directory, then its parent (p30/), then the
# working directory.
search_roots <- unique(c(
  project_dir,
  dirname(project_dir),
  normalizePath(getwd(), mustWork = TRUE)
))
data_dir <- NULL

for (root in search_roots) {
  data_dir <- find_complete_data_dir(root)
  if (!is.null(data_dir)) break
}

# If the CSVs have not yet been found, look for data_new.zip below the same
# roots, extract it where it currently lives, and search again.
if (is.null(data_dir)) {
  zip_candidates <- character(0)
  for (root in search_roots) {
    zip_candidates <- c(
      zip_candidates,
      list.files(
        root,
        pattern = "^data_new\\.zip$",
        recursive = TRUE,
        full.names = TRUE,
        ignore.case = TRUE
      )
    )
  }
  zip_candidates <- unique(zip_candidates)

  if (length(zip_candidates) > 0L) {
    data_zip <- zip_candidates[1L]
    unzip_dir <- dirname(data_zip)
    message("Data CSVs not found; extracting: ", data_zip)
    unzip(data_zip, exdir = unzip_dir)

    for (root in search_roots) {
      data_dir <- find_complete_data_dir(root)
      if (!is.null(data_dir)) break
    }
  }
}

if (is.null(data_dir)) {
  stop(
    "Could not locate the p=30 data. I searched recursively below: ",
    paste(search_roots, collapse = "; "),
    ". The required files are seed_00.csv, ..., seed_49.csv and ",
    truth_basename,
    " in the same folder."
  )
}

seed_files <- file.path(data_dir, expected_seed_basenames)
truth_file <- file.path(data_dir, truth_basename)

# Full run over all 50 datasets.
replicates_to_run <- 0:49

# varbvs prior log-odds grid (base 10). The true inclusion rate 5/30 = 1/6
# corresponds to base-10 log-odds of about -0.70, inside the grid.
logodds_grid <- seq(-3, 0, by = 0.1)

# Use a tighter inner-loop tolerance than the varbvs default so that the
# comparison is deliberately favourable to the Carbonetto-Stephens fit.
varbvs_tol <- 1e-6
varbvs_maxiter <- 10000

# Use ALL varbvs hyperparameter settings in the matched Gibbs average.
# There is no 99.9% weight truncation in this p = 30 experiment.
weight_coverage <- 1

# Gibbs controls: four dispersed chains per hyperparameter setting.
burn_sweeps <- 2000
keep_sweeps <- 20000
thin <- 1
n_chains <- 4

# Systematic componentwise Gibbs: gamma_1, ..., gamma_p in each sweep.
random_scan <- FALSE

base_seed <- 20260818

# Optional exact-enumeration validation of the Gibbs transition on a
# reduced subset. This validates the sampler mechanics, not the p = 30
# posterior itself. The sampler has already passed this check, so it
# is FALSE by default.
RUN_VALIDATION <- FALSE
validation_replicate <- 0
validation_p <- 10
validation_burn_sweeps <- 1000
validation_keep_sweeps <- 5000
validation_chains <- 2

########################################################################
# 2. Basic checks, data setup and readers
########################################################################

installed_version <- as.character(packageVersion("varbvs"))
if (installed_version != EXPECTED_VARBVS_VERSION) {
  stop(
    "This benchmark was written for varbvs ",
    EXPECTED_VARBVS_VERSION,
    ", but the installed version is ",
    installed_version,
    "."
  )
}

# The data directory was located automatically above. At this point all
# required inputs have already been verified to exist in the same folder.

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

if (!dir.exists(mcmc_output_dir)) {
  dir.create(mcmc_output_dir, recursive = TRUE)
}

if (!dir.exists(varbvs_cache_dir)) {
  dir.create(varbvs_cache_dir, recursive = TRUE)
}

cat("Project directory: ", project_dir, "\n", sep = "")
cat("Data directory:    ", data_dir, "\n", sep = "")
cat("Output directory:  ", output_dir, "\n", sep = "")

read_p30_dataset <- function(file) {
  dat <- read.csv(file)
  x_names <- paste0("X", 1:30)

  stopifnot("y" %in% names(dat))
  stopifnot(all(x_names %in% names(dat)))

  list(
    X = as.matrix(dat[, x_names]),
    y = as.numeric(dat$y),
    file = file
  )
}

truth <- read.csv(truth_file)
truth_predictors <- subset(truth, variable_name != "Intercept")
truth_predictors <- truth_predictors[
  match(paste0("X", 1:30), truth_predictors$variable_name),
]

if (nrow(truth_predictors) != 30 || anyNA(truth_predictors$variable_name)) {
  stop("Truth file does not contain X1, ..., X30 in the expected form.")
}

true_active <- truth_predictors$is_true_active == 1
true_beta <- truth_predictors$beta

if (sum(true_active) != 5) {
  warning(
    "Expected 5 active predictors in the new design, found ",
    sum(true_active), "."
  )
}

# Verify dimensions and standardisation before fitting anything.
standardisation_summary <- data.frame(
  replicate = 0:49,
  n = NA_integer_,
  p = NA_integer_,
  max_abs_col_mean = NA_real_,
  min_col_sd = NA_real_,
  max_col_sd = NA_real_,
  mean_y = NA_real_,
  sd_y = NA_real_
)

for (r in 0:49) {
  d <- read_p30_dataset(seed_files[r + 1])

  standardisation_summary$n[r + 1] <- nrow(d$X)
  standardisation_summary$p[r + 1] <- ncol(d$X)
  standardisation_summary$max_abs_col_mean[r + 1] <- max(abs(colMeans(d$X)))

  x_sd <- apply(d$X, 2, sd)
  standardisation_summary$min_col_sd[r + 1] <- min(x_sd)
  standardisation_summary$max_col_sd[r + 1] <- max(x_sd)
  standardisation_summary$mean_y[r + 1] <- mean(d$y)
  standardisation_summary$sd_y[r + 1] <- sd(d$y)
}

if (any(standardisation_summary$n != 1000)) {
  stop("At least one dataset does not have n = 1000 observations.")
}

if (any(standardisation_summary$p != 30)) {
  stop("At least one dataset does not have p = 30 predictors.")
}

write.csv(
  standardisation_summary,
  file.path(output_dir, "data_standardisation_summary.csv"),
  row.names = FALSE
)

########################################################################
# 3. Helper functions for the matched collapsed posterior
########################################################################

# Convert the base-10 prior log-odds used by varbvs to an inclusion
# probability.
logodds_to_q <- function(logodds) {
  1 / (1 + 10^(-logodds))
}

# Centre y and X. This is exactly the residualisation required after
# integrating out the always-included intercept under a flat prior.
centre_for_flat_intercept <- function(X, y) {
  list(
    X = sweep(X, 2, colMeans(X), FUN = "-"),
    y = y - mean(y)
  )
}

# Precompute sufficient statistics. Once these are available, the Gibbs
# sampler does not revisit the n observations at every indicator update.
make_sufficient_statistics <- function(X, y) {
  centred <- centre_for_flat_intercept(X, y)

  Xc <- centred$X
  yc <- centred$y

  list(
    G = crossprod(Xc),
    xy = as.numeric(crossprod(Xc, yc)),
    yy = sum(yc^2),
    n = nrow(Xc),
    p = ncol(Xc)
  )
}

# Inverse of
#
#   M_A = X_A' X_A + (1/sa) I
#
# for the current active set A.
initial_active_inverse <- function(active, G, lambda) {
  k <- length(active)

  if (k == 0) {
    return(matrix(numeric(0), nrow = 0, ncol = 0))
  }

  M <- G[active, active, drop = FALSE] + diag(lambda, k)
  chol2inv(chol(M))
}

# Remove one currently active variable from M_A^{-1} without recomputing
# the inverse from scratch.
remove_from_inverse <- function(Minv, pos) {
  k <- nrow(Minv)

  if (k == 1) {
    return(matrix(numeric(0), nrow = 0, ncol = 0))
  }

  keep <- setdiff(seq_len(k), pos)

  E <- Minv[keep, keep, drop = FALSE]
  f <- Minv[keep, pos, drop = FALSE]
  g <- Minv[pos, pos]

  E - tcrossprod(f) / g
}

# Compute the exact collapsed log Bayes factor for adding variable j to
# the current active set A.
#
# If
#
#   M_A = X_A'X_A + lambda I, lambda = 1/sa,
#
# define
#
#   s = x_j'x_j + lambda - c' M_A^{-1} c,
#   t = x_j'y - c' M_A^{-1} X_A'y,
#
# where c = X_A'x_j. Then
#
#   log BF(add j)
#     = -0.5 log(sa * s) + t^2/(2 * sigma * s).
#
# sigma is the residual VARIANCE used by varbvs.
candidate_addition <- function(j, active, Minv, G, xy, sigma, sa) {
  lambda <- 1 / sa
  d <- G[j, j] + lambda

  if (length(active) == 0) {
    s <- d
    t <- xy[j]
    cvec <- numeric(0)
    u <- numeric(0)
  } else {
    cvec <- G[active, j, drop = TRUE]
    u <- as.numeric(Minv %*% cvec)

    s <- d - sum(cvec * u)

    bA <- xy[active]
    t <- xy[j] - sum(u * bA)
  }

  if (!is.finite(s) || s <= 0) {
    stop(
      "Non-positive Schur complement encountered for variable ",
      j,
      ": s = ",
      s,
      "."
    )
  }

  log_bf <- -0.5 * log(sa * s) + (t^2) / (2 * sigma * s)

  list(
    log_bf = log_bf,
    s = s,
    cvec = cvec,
    u = u
  )
}

# Add a variable to the current active-set inverse using the block inverse
# formula. The candidate object must have been computed against the same
# active set and Minv.
add_to_inverse <- function(Minv, candidate) {
  s <- candidate$s

  if (nrow(Minv) == 0) {
    return(matrix(1 / s, nrow = 1, ncol = 1))
  }

  u <- matrix(candidate$u, ncol = 1)

  top_left <- Minv + tcrossprod(u) / s
  top_right <- -u / s

  rbind(
    cbind(top_left, top_right),
    cbind(t(top_right), 1 / s)
  )
}

# Exact collapsed log target, up to a constant not depending on gamma.
# Used only for exact-enumeration validation.
collapsed_log_target <- function(gamma, G, xy, sigma, sa, logodds) {
  active <- which(gamma == 1)
  k <- length(active)

  log_prior_odds <- log(10) * logodds

  if (k == 0) {
    return(0)
  }

  lambda <- 1 / sa
  M <- G[active, active, drop = FALSE] + diag(lambda, k)
  R <- chol(M)

  logdet_M <- 2 * sum(log(diag(R)))
  Minv_b <- backsolve(R, forwardsolve(t(R), xy[active]))
  quad <- sum(xy[active] * Minv_b)

  -0.5 * k * log(sa) -
    0.5 * logdet_M +
    quad / (2 * sigma) +
    k * log_prior_odds
}

########################################################################
# 4. Componentwise collapsed Gibbs sampler
########################################################################

run_componentwise_gibbs <- function(
    G,
    xy,
    sigma,
    sa,
    logodds,
    gamma_init = NULL,
    burn_sweeps = 2000,
    keep_sweeps = 20000,
    thin = 1,
    seed = 1,
    random_scan = FALSE) {

  set.seed(seed)

  p <- length(xy)

  if (!is.finite(sigma) || sigma <= 0) {
    stop("sigma must be positive.")
  }

  if (!is.finite(sa) || sa <= 0) {
    stop("sa must be positive.")
  }

  if (is.null(gamma_init)) {
    gamma <- rep(FALSE, p)
  } else {
    if (length(gamma_init) != p) {
      stop("gamma_init must have length p.")
    }
    gamma <- as.logical(gamma_init)
  }

  active <- which(gamma)
  Minv <- initial_active_inverse(active, G, lambda = 1 / sa)

  prior_log_odds <- log(10) * logodds
  n_total_sweeps <- burn_sweeps + keep_sweeps * thin

  gamma_draws <- matrix(FALSE, nrow = keep_sweeps, ncol = p)
  model_size_draws <- numeric(keep_sweeps)

  save_index <- 0L

  for (sweep in seq_len(n_total_sweeps)) {

    update_order <- if (random_scan) sample.int(p) else seq_len(p)

    for (j in update_order) {

      # Condition on gamma_{-j}. If j is currently active, first remove it
      # from the active set and update M^{-1}.
      if (gamma[j]) {
        pos <- match(j, active)

        Minv <- remove_from_inverse(Minv, pos)
        active <- active[-pos]
        gamma[j] <- FALSE
      }

      candidate <- candidate_addition(
        j = j,
        active = active,
        Minv = Minv,
        G = G,
        xy = xy,
        sigma = sigma,
        sa = sa
      )

      log_posterior_odds <- prior_log_odds + candidate$log_bf
      inclusion_prob <- plogis(log_posterior_odds)

      if (runif(1) < inclusion_prob) {
        Minv <- add_to_inverse(Minv, candidate)
        active <- c(active, j)
        gamma[j] <- TRUE
      }
    }

    if (
      sweep > burn_sweeps &&
      ((sweep - burn_sweeps) %% thin == 0)
    ) {
      save_index <- save_index + 1L
      gamma_draws[save_index, ] <- gamma
      model_size_draws[save_index] <- length(active)
    }
  }

  half <- floor(keep_sweeps / 2)
  pip <- colMeans(gamma_draws)
  pip_split1 <- colMeans(gamma_draws[seq_len(half), , drop = FALSE])
  pip_split2 <- colMeans(
    gamma_draws[(half + 1):keep_sweeps, , drop = FALSE]
  )

  list(
    pip = pip,
    pip_split1 = pip_split1,
    pip_split2 = pip_split2,
    mean_model_size = mean(model_size_draws),
    sd_model_size = sd(model_size_draws),
    model_size_draws = model_size_draws,
    gamma_draws = gamma_draws,
    final_gamma = gamma
  )
}

# Four deliberately dispersed initialisations for the Gibbs reference:
# empty model, full model, varbvs median-probability model, and a random
# draw from the varbvs mean-field approximation. Additional chains, if
# requested, use independent Bernoulli(0.5) starts.
make_chain_initialisation <- function(chain, alpha, seed) {
  p <- length(alpha)

  if (chain == 1) {
    return(rep(FALSE, p))
  }

  if (chain == 2) {
    return(rep(TRUE, p))
  }

  if (chain == 3) {
    return(alpha >= 0.5)
  }

  set.seed(seed)

  if (chain == 4) {
    return(as.logical(rbinom(p, size = 1, prob = alpha)))
  }

  as.logical(rbinom(p, size = 1, prob = 0.5))
}

# Standard potential scale reduction factor (R-hat) for a scalar sampled
# in m chains of equal length. Constant, identical chains return 1.
scalar_rhat <- function(draw_matrix) {
  draw_matrix <- as.matrix(draw_matrix)

  n <- nrow(draw_matrix)
  m <- ncol(draw_matrix)

  if (m < 2 || n < 2) {
    return(NA_real_)
  }

  chain_means <- colMeans(draw_matrix)
  chain_vars <- apply(draw_matrix, 2, var)

  W <- mean(chain_vars)
  B <- n * var(chain_means)

  if (!is.finite(W) || W < 0 || !is.finite(B) || B < 0) {
    return(NA_real_)
  }

  if (W == 0) {
    if (max(chain_means) - min(chain_means) < 1e-12) {
      return(1)
    }
    return(Inf)
  }

  var_hat <- ((n - 1) / n) * W + B / n
  sqrt(var_hat / W)
}

compute_chain_diagnostics <- function(chain_outputs) {
  m <- length(chain_outputs)
  p <- length(chain_outputs[[1]]$pip)
  n_keep <- nrow(chain_outputs[[1]]$gamma_draws)

  chain_pips <- sapply(chain_outputs, function(x) x$pip)
  if (m == 1) {
    chain_pips <- matrix(chain_pips, nrow = p, ncol = 1)
  }

  pip_range <- apply(chain_pips, 1, function(x) max(x) - min(x))

  rhat <- numeric(p)
  for (j in seq_len(p)) {
    draws_j <- matrix(
      NA_real_,
      nrow = n_keep,
      ncol = m
    )

    for (chain in seq_len(m)) {
      draws_j[, chain] <- as.numeric(chain_outputs[[chain]]$gamma_draws[, j])
    }

    rhat[j] <- scalar_rhat(draws_j)
  }

  model_size_matrix <- sapply(
    chain_outputs,
    function(x) x$model_size_draws
  )

  split_half_mae <- sapply(
    chain_outputs,
    function(x) mean(abs(x$pip_split1 - x$pip_split2))
  )

  finite_rhat <- rhat[is.finite(rhat)]
  max_rhat <- if (length(finite_rhat) > 0) max(finite_rhat) else Inf

  list(
    chain_pips = chain_pips,
    pip_range = pip_range,
    rhat = rhat,
    model_size_rhat = scalar_rhat(model_size_matrix),
    mean_split_half_mae = mean(split_half_mae),
    max_split_half_mae = max(split_half_mae),
    mean_between_chain_pip_range = mean(pip_range),
    max_between_chain_pip_range = max(pip_range),
    max_rhat = max_rhat,
    n_rhat_gt_1_01 = sum(rhat > 1.01, na.rm = TRUE),
    n_nonfinite_rhat = sum(!is.finite(rhat))
  )
}

########################################################################
# 5. Exact enumeration for optional validation
########################################################################

exact_pips_by_enumeration <- function(G, xy, sigma, sa, logodds) {
  p <- length(xy)

  if (p > 20) {
    stop("Exact enumeration is restricted to p <= 20.")
  }

  n_models <- 2^p
  logpost <- numeric(n_models)
  gamma_matrix <- matrix(FALSE, nrow = n_models, ncol = p)

  for (m in 0:(n_models - 1)) {
    gamma <- as.logical(intToBits(m)[seq_len(p)])
    gamma_matrix[m + 1, ] <- gamma

    logpost[m + 1] <- collapsed_log_target(
      gamma = gamma,
      G = G,
      xy = xy,
      sigma = sigma,
      sa = sa,
      logodds = logodds
    )
  }

  w <- exp(logpost - max(logpost))
  w <- w / sum(w)

  as.numeric(crossprod(w, gamma_matrix))
}

########################################################################
# 6. Hyperparameter-setting selection
########################################################################

select_hyperparameter_settings <- function(w, coverage = 1) {
  w <- as.numeric(w)
  w <- w / sum(w)

  if (coverage <= 0 || coverage > 1) {
    stop("weight_coverage must lie in (0, 1].")
  }

  if (coverage == 1) {
    return(seq_along(w))
  }

  ord <- order(w, decreasing = TRUE)
  cumulative <- cumsum(w[ord])
  k <- which(cumulative >= coverage)[1]

  sort(ord[seq_len(k)])
}

########################################################################
# 7. Cache/resume helpers
########################################################################

varbvs_cache_file <- function(r) {
  file.path(
    varbvs_cache_dir,
    sprintf("replicate_%02d_varbvs_fit.rds", r)
  )
}

replicate_result_file <- function(r) {
  file.path(
    mcmc_output_dir,
    sprintf("replicate_%02d_varbvs_gibbs.rds", r)
  )
}

# Controls that a reused replicate result must have been produced with.
current_controls <- function() {
  list(
    logodds_grid = logodds_grid,
    varbvs_tol = varbvs_tol,
    varbvs_maxiter = varbvs_maxiter,
    weight_coverage = weight_coverage,
    burn_sweeps = burn_sweeps,
    keep_sweeps = keep_sweeps,
    thin = thin,
    n_chains = n_chains,
    random_scan = random_scan,
    base_seed = base_seed
  )
}

same_controls <- function(saved) {
  if (!is.list(saved)) return(FALSE)
  cur <- current_controls()
  for (nm in names(cur)) {
    a <- saved[[nm]]
    b <- cur[[nm]]
    if (is.null(a) || length(a) != length(b)) return(FALSE)
    if (is.numeric(b)) {
      if (max(abs(as.numeric(a) - b)) > 1e-12) return(FALSE)
    } else if (!identical(as.vector(a), as.vector(b))) {
      return(FALSE)
    }
  }
  TRUE
}

# A cached varbvs fit is reused only if it used the same grid, tol and
# maxiter as the current run.
same_varbvs_controls <- function(obj) {
  is.list(obj) &&
    !is.null(obj$varbvs_tol) && !is.null(obj$varbvs_maxiter) &&
    !is.null(obj$logodds_grid) &&
    length(obj$logodds_grid) == length(logodds_grid) &&
    max(abs(as.numeric(obj$logodds_grid) - logodds_grid)) < 1e-12 &&
    abs(obj$varbvs_tol - varbvs_tol) < 1e-15 &&
    obj$varbvs_maxiter == varbvs_maxiter
}

valid_varbvs_fit <- function(fit) {
  is.list(fit) &&
    !is.null(fit$alpha) &&
    !is.null(fit$w) &&
    !is.null(fit$sigma) &&
    !is.null(fit$sa) &&
    !is.null(fit$logodds) &&
    nrow(fit$alpha) == 30L &&
    length(fit$w) == length(logodds_grid) &&
    length(fit$logodds) == length(logodds_grid) &&
    max(abs(as.numeric(fit$logodds) - logodds_grid)) < 1e-12
}

load_cached_varbvs <- function(r) {
  cache_file <- varbvs_cache_file(r)

  if (REUSE_CACHED_VARBVS_FITS && file.exists(cache_file)) {
    obj <- readRDS(cache_file)

    # New cache format stores both fit and elapsed runtime.
    if (is.list(obj) && !is.null(obj$fit) && valid_varbvs_fit(obj$fit) &&
        same_varbvs_controls(obj)) {
      return(list(
        fit = obj$fit,
        elapsed_seconds = if (!is.null(obj$elapsed_seconds)) obj$elapsed_seconds else NA_real_,
        source = "dedicated cache"
      ))
    }
  }

  # A completed replicate RDS also stores its varbvs fit. Recover it (and
  # recreate the dedicated cache) if it was fitted with the same settings.
  old_file <- replicate_result_file(r)
  if (REUSE_CACHED_VARBVS_FITS && file.exists(old_file)) {
    old_obj <- readRDS(old_file)
    if (is.list(old_obj) && !is.null(old_obj$fit) && valid_varbvs_fit(old_obj$fit) &&
        same_varbvs_controls(old_obj$controls)) {
      elapsed <- NA_real_
      if (!is.null(old_obj$replicate_summary) &&
          "varbvs_runtime_seconds" %in% names(old_obj$replicate_summary)) {
        elapsed <- as.numeric(old_obj$replicate_summary$varbvs_runtime_seconds[1])
      }

      saveRDS(
        list(
          replicate = r,
          fit = old_obj$fit,
          elapsed_seconds = elapsed,
          logodds_grid = logodds_grid,
          varbvs_tol = varbvs_tol,
          varbvs_maxiter = varbvs_maxiter
        ),
        cache_file
      )

      return(list(
        fit = old_obj$fit,
        elapsed_seconds = elapsed,
        source = "previous benchmark RDS"
      ))
    }
  }

  NULL
}

########################################################################
# 8. Fit varbvs and run the matched Gibbs benchmark
########################################################################

replicate_summaries <- vector("list", length(replicates_to_run))
variable_results <- vector("list", length(replicates_to_run))
hyperparameter_summaries <- vector("list", length(replicates_to_run))
varbvs_runtime_rows <- vector("list", length(replicates_to_run))

for (rr in seq_along(replicates_to_run)) {
  r <- replicates_to_run[rr]

  cat("\n============================================================\n")
  cat("Replicate ", r, "\n", sep = "")
  cat("============================================================\n")

  # If this replicate has already completed with exactly the current
  # settings, reuse it and move on, so an interrupted run can be resumed.
  # A result produced with different settings is recomputed.
  completed_file <- replicate_result_file(r)
  if (REUSE_COMPLETED_REPLICATES && file.exists(completed_file)) {
    completed <- readRDS(completed_file)

    if (is.list(completed) && !same_controls(completed$controls)) {
      cat("Saved result used different settings; recomputing: ", completed_file, "\n", sep = "")
    } else if (is.list(completed) &&
        !is.null(completed$variable_results) &&
        !is.null(completed$hyperparameter_summary) &&
        !is.null(completed$replicate_summary)) {

      cat("Completed benchmark found. Reusing: ", completed_file, "\n", sep = "")

      variable_results[[rr]] <- completed$variable_results
      hyperparameter_summaries[[rr]] <- completed$hyperparameter_summary
      replicate_summaries[[rr]] <- completed$replicate_summary

      vb_time <- NA_real_
      if ("varbvs_runtime_seconds" %in% names(completed$replicate_summary)) {
        vb_time <- as.numeric(completed$replicate_summary$varbvs_runtime_seconds[1])
      }
      varbvs_runtime_rows[[rr]] <- data.frame(
        replicate = r,
        elapsed_seconds = vb_time,
        elapsed_minutes = vb_time / 60
      )

      # Also create the dedicated varbvs cache for future use if possible.
      if (!is.null(completed$fit) && valid_varbvs_fit(completed$fit) &&
          !file.exists(varbvs_cache_file(r))) {
        saveRDS(
          list(
            replicate = r,
            fit = completed$fit,
            elapsed_seconds = vb_time,
            logodds_grid = logodds_grid,
            varbvs_tol = varbvs_tol,
            varbvs_maxiter = varbvs_maxiter
          ),
          varbvs_cache_file(r)
        )
      }

      next
    }
  }

  d <- read_p30_dataset(seed_files[r + 1])
  stats <- make_sufficient_statistics(d$X, d$y)

  ######################################################################
  # 8.1 Load or fit varbvs
  ######################################################################

  cached <- load_cached_varbvs(r)

  if (!is.null(cached)) {
    fit <- cached$fit
    varbvs_elapsed <- cached$elapsed_seconds
    cat("Loading existing varbvs fit from ", cached$source, ".\n", sep = "")
  } else {
    cat("No cached varbvs fit found. Fitting varbvs...\n")
    varbvs_start <- Sys.time()

    fit <- varbvs(
      X = d$X,
      Z = NULL,
      y = d$y,
      family = "gaussian",
      logodds = logodds_grid,
      tol = varbvs_tol,
      maxiter = varbvs_maxiter,
      verbose = FALSE
    )

    varbvs_end <- Sys.time()
    varbvs_elapsed <- as.numeric(
      difftime(varbvs_end, varbvs_start, units = "secs")
    )

    saveRDS(
      list(
        replicate = r,
        fit = fit,
        elapsed_seconds = varbvs_elapsed,
        logodds_grid = logodds_grid,
        varbvs_tol = varbvs_tol,
        varbvs_maxiter = varbvs_maxiter
      ),
      varbvs_cache_file(r)
    )

    cat("Saved varbvs fit cache: ", varbvs_cache_file(r), "\n", sep = "")
  }

  varbvs_runtime_rows[[rr]] <- data.frame(
    replicate = r,
    elapsed_seconds = varbvs_elapsed,
    elapsed_minutes = varbvs_elapsed / 60
  )

  if (nrow(fit$alpha) != 30) {
    stop("Unexpected number of varbvs PIPs in replicate ", r, ".")
  }

  if (length(fit$w) != length(logodds_grid)) {
    stop("Unexpected number of varbvs hyperparameter weights.")
  }

  ######################################################################
  # 8.2 Optional reduced exact-enumeration validation
  ######################################################################

  if (RUN_VALIDATION && r == validation_replicate) {
    cat("Running reduced exact-enumeration validation...\n")

    h_val <- which.max(fit$w)
    vars_val <- order(abs(fit$alpha[, h_val] - 0.5))[1:validation_p]
    X_small <- d$X[, vars_val, drop = FALSE]
    stats_small <- make_sufficient_statistics(X_small, d$y)

    sigma_h <- as.numeric(fit$sigma[h_val])
    sa_h <- as.numeric(fit$sa[h_val])
    logodds_h <- as.numeric(fit$logodds[h_val])
    alpha_h <- as.numeric(fit$alpha[vars_val, h_val])

    exact_pip <- exact_pips_by_enumeration(
      G = stats_small$G,
      xy = stats_small$xy,
      sigma = sigma_h,
      sa = sa_h,
      logodds = logodds_h
    )

    chain_pips_val <- matrix(
      NA_real_,
      nrow = validation_p,
      ncol = validation_chains
    )

    for (chain in seq_len(validation_chains)) {
      gamma_init <- make_chain_initialisation(
        chain = chain,
        alpha = alpha_h,
        seed = base_seed + 900000 + chain
      )

      g_val <- run_componentwise_gibbs(
        G = stats_small$G,
        xy = stats_small$xy,
        sigma = sigma_h,
        sa = sa_h,
        logodds = logodds_h,
        gamma_init = gamma_init,
        burn_sweeps = validation_burn_sweeps,
        keep_sweeps = validation_keep_sweeps,
        thin = 1,
        seed = base_seed + 910000 + chain,
        random_scan = random_scan
      )

      chain_pips_val[, chain] <- g_val$pip
    }

    gibbs_pip_val <- rowMeans(chain_pips_val)

    validation_table <- data.frame(
      variable = paste0("X", vars_val),
      exact_pip = exact_pip,
      gibbs_pip = gibbs_pip_val,
      abs_difference = abs(gibbs_pip_val - exact_pip)
    )

    print(validation_table)

    write.csv(
      validation_table,
      file.path(mcmc_output_dir, "gibbs_exact_enumeration_validation.csv"),
      row.names = FALSE
    )
  }

  ######################################################################
  # 8.3 Matched Gibbs at every varbvs hyperparameter setting
  ######################################################################

  w_all <- as.numeric(fit$w)
  w_all <- w_all / sum(w_all)

  settings <- select_hyperparameter_settings(
    w = w_all,
    coverage = weight_coverage
  )

  retained_weight <- sum(w_all[settings])
  w_selected <- w_all[settings] / retained_weight

  cat(
    "Hyperparameter settings retained: ",
    length(settings),
    " / ",
    length(w_all),
    "\n",
    sep = ""
  )

  cat(
    "Original varbvs weight retained: ",
    signif(retained_weight, 8),
    "\n",
    sep = ""
  )

  conditional_gibbs_pip <- matrix(
    NA_real_,
    nrow = stats$p,
    ncol = length(settings)
  )

  conditional_vb_pip <- fit$alpha[, settings, drop = FALSE]
  hyper_rows <- vector("list", length(settings))

  gibbs_start <- Sys.time()

  for (hh in seq_along(settings)) {
    h <- settings[hh]

    sigma_h <- as.numeric(fit$sigma[h])
    sa_h <- as.numeric(fit$sa[h])
    logodds_h <- as.numeric(fit$logodds[h])
    alpha_h <- as.numeric(fit$alpha[, h])

    cat(
      "  setting ", h,
      ": logodds=", signif(logodds_h, 5),
      ", sigma=", signif(sigma_h, 5),
      ", sa=", signif(sa_h, 5),
      ", weight=", signif(w_all[h], 5),
      "\n",
      sep = ""
    )

    chain_outputs <- vector("list", n_chains)

    for (chain in seq_len(n_chains)) {
      init_seed <- base_seed + 1000000 * r + 10000 * h + 100 * chain
      gamma_init <- make_chain_initialisation(
        chain = chain,
        alpha = alpha_h,
        seed = init_seed
      )

      chain_outputs[[chain]] <- run_componentwise_gibbs(
        G = stats$G,
        xy = stats$xy,
        sigma = sigma_h,
        sa = sa_h,
        logodds = logodds_h,
        gamma_init = gamma_init,
        burn_sweeps = burn_sweeps,
        keep_sweeps = keep_sweeps,
        thin = thin,
        seed = init_seed + 1,
        random_scan = random_scan
      )
    }

    diag_h <- compute_chain_diagnostics(chain_outputs)
    gibbs_pip_h <- rowMeans(diag_h$chain_pips)
    conditional_gibbs_pip[, hh] <- gibbs_pip_h

    vb_h <- conditional_vb_pip[, hh]

    hyper_rows[[hh]] <- data.frame(
      replicate = r,
      setting_index = h,
      logodds = logodds_h,
      q = logodds_to_q(logodds_h),
      sigma = sigma_h,
      sa = sa_h,
      varbvs_weight = w_all[h],
      retained_normalized_weight = w_selected[hh],
      vb_gibbs_mae = mean(abs(vb_h - gibbs_pip_h)),
      vb_gibbs_rmse = sqrt(mean((vb_h - gibbs_pip_h)^2)),
      vb_gibbs_max_abs = max(abs(vb_h - gibbs_pip_h)),
      mean_split_half_mae = diag_h$mean_split_half_mae,
      max_split_half_mae = diag_h$max_split_half_mae,
      mean_between_chain_pip_range = diag_h$mean_between_chain_pip_range,
      max_between_chain_pip_range = diag_h$max_between_chain_pip_range,
      max_rhat = diag_h$max_rhat,
      n_rhat_gt_1_01 = diag_h$n_rhat_gt_1_01,
      n_nonfinite_rhat = diag_h$n_nonfinite_rhat,
      model_size_rhat = diag_h$model_size_rhat,
      mean_gibbs_model_size = mean(
        sapply(chain_outputs, function(x) x$mean_model_size)
      )
    )
  }

  gibbs_end <- Sys.time()
  gibbs_elapsed <- as.numeric(
    difftime(gibbs_end, gibbs_start, units = "secs")
  )

  ######################################################################
  # 8.4 Combine conditional PIPs using EXACTLY the varbvs weights
  ######################################################################

  vb_pip_matched <- as.numeric(
    conditional_vb_pip %*% w_selected
  )

  gibbs_pip_matched <- as.numeric(
    conditional_gibbs_pip %*% w_selected
  )

  # With weight_coverage = 1, this should agree with fit$pip to numerical
  # precision. Record the discrepancy as a check.
  max_vb_reconstruction_error <- max(
    abs(vb_pip_matched - as.numeric(fit$pip))
  )

  abs_error <- abs(vb_pip_matched - gibbs_pip_matched)
  squared_error <- (vb_pip_matched - gibbs_pip_matched)^2

  variable_results[[rr]] <- data.frame(
    replicate = r,
    variable = paste0("X", seq_len(stats$p)),
    index = seq_len(stats$p),
    true_active = true_active,
    true_beta = true_beta,
    varbvs_pip = vb_pip_matched,
    gibbs_pip = gibbs_pip_matched,
    signed_pip_error = vb_pip_matched - gibbs_pip_matched,
    abs_pip_error = abs_error,
    squared_pip_error = squared_error
  )

  hyperparameter_summaries[[rr]] <- do.call(rbind, hyper_rows)

  replicate_summaries[[rr]] <- data.frame(
    replicate = r,
    n = stats$n,
    p = stats$p,
    settings_retained = length(settings),
    settings_total = length(w_all),
    retained_varbvs_weight = retained_weight,
    pip_mae = mean(abs_error),
    pip_rmse = sqrt(mean(squared_error)),
    pip_max_abs = max(abs_error),
    pip_correlation = cor(vb_pip_matched, gibbs_pip_matched),
    mean_abs_error_true_active = mean(abs_error[true_active]),
    mean_abs_error_true_inactive = mean(abs_error[!true_active]),
    max_vb_reconstruction_error = max_vb_reconstruction_error,
    varbvs_runtime_seconds = varbvs_elapsed,
    gibbs_runtime_seconds = gibbs_elapsed,
    total_runtime_seconds = varbvs_elapsed + gibbs_elapsed,
    max_setting_rhat = max(hyperparameter_summaries[[rr]]$max_rhat, na.rm = TRUE),
    max_setting_between_chain_pip_range = max(
      hyperparameter_summaries[[rr]]$max_between_chain_pip_range,
      na.rm = TRUE
    ),
    max_setting_split_half_mae = max(
      hyperparameter_summaries[[rr]]$max_split_half_mae,
      na.rm = TRUE
    )
  )

  ######################################################################
  # 8.5 Save replicate-level results immediately
  ######################################################################

  saveRDS(
    list(
      replicate = r,
      fit = fit,
      settings = settings,
      retained_weight = retained_weight,
      weights = w_selected,
      conditional_vb_pip = conditional_vb_pip,
      conditional_gibbs_pip = conditional_gibbs_pip,
      varbvs_pip = vb_pip_matched,
      gibbs_pip = gibbs_pip_matched,
      variable_results = variable_results[[rr]],
      hyperparameter_summary = hyperparameter_summaries[[rr]],
      replicate_summary = replicate_summaries[[rr]],
      controls = list(
        varbvs_version = installed_version,
        logodds_grid = logodds_grid,
        varbvs_tol = varbvs_tol,
        varbvs_maxiter = varbvs_maxiter,
        weight_coverage = weight_coverage,
        burn_sweeps = burn_sweeps,
        keep_sweeps = keep_sweeps,
        thin = thin,
        n_chains = n_chains,
        random_scan = random_scan,
        base_seed = base_seed
      )
    ),
    file.path(
      mcmc_output_dir,
      sprintf("replicate_%02d_varbvs_gibbs.rds", r)
    )
  )

  write.csv(
    variable_results[[rr]],
    file.path(
      mcmc_output_dir,
      sprintf("replicate_%02d_varbvs_vs_gibbs_pips.csv", r)
    ),
    row.names = FALSE
  )

  # Direct PIP-versus-PIP diagnostic plot for this replicate.
  png(
    file.path(
      mcmc_output_dir,
      sprintf("replicate_%02d_varbvs_vs_gibbs_pips.png", r)
    ),
    width = 900,
    height = 900
  )

  plot(
    gibbs_pip_matched,
    vb_pip_matched,
    xlim = c(0, 1),
    ylim = c(0, 1),
    xlab = "Gibbs PIP",
    ylab = "varbvs PIP",
    main = paste("Replicate", r)
  )
  abline(a = 0, b = 1, lty = 2)
  dev.off()

  cat("Replicate summary:\n")
  print(replicate_summaries[[rr]])
}

########################################################################
# 9. Aggregate outputs over the requested replicates
########################################################################

variable_results <- do.call(rbind, variable_results)
replicate_summaries <- do.call(rbind, replicate_summaries)
hyperparameter_summaries <- do.call(rbind, hyperparameter_summaries)
varbvs_runtime_rows <- do.call(rbind, varbvs_runtime_rows)

overall_summary <- data.frame(
  metric = c(
    "Mean replicate PIP MAE",
    "Median replicate PIP MAE",
    "Mean replicate PIP RMSE",
    "Median replicate PIP RMSE",
    "Mean replicate maximum absolute PIP error",
    "Median replicate maximum absolute PIP error",
    "Mean PIP correlation",
    "Mean absolute PIP error for true active predictors",
    "Mean absolute PIP error for true inactive predictors",
    "Mean varbvs runtime seconds",
    "Mean Gibbs runtime seconds",
    "Maximum R-hat over all settings run",
    "Maximum between-chain PIP range over all settings run",
    "Maximum split-half PIP MAE over all settings run",
    "Mean retained varbvs hyperparameter weight"
  ),
  value = c(
    mean(replicate_summaries$pip_mae),
    median(replicate_summaries$pip_mae),
    mean(replicate_summaries$pip_rmse),
    median(replicate_summaries$pip_rmse),
    mean(replicate_summaries$pip_max_abs),
    median(replicate_summaries$pip_max_abs),
    mean(replicate_summaries$pip_correlation),
    mean(replicate_summaries$mean_abs_error_true_active),
    mean(replicate_summaries$mean_abs_error_true_inactive),
    mean(replicate_summaries$varbvs_runtime_seconds),
    mean(replicate_summaries$gibbs_runtime_seconds),
    max(hyperparameter_summaries$max_rhat, na.rm = TRUE),
    max(hyperparameter_summaries$max_between_chain_pip_range, na.rm = TRUE),
    max(hyperparameter_summaries$max_split_half_mae, na.rm = TRUE),
    mean(replicate_summaries$retained_varbvs_weight)
  )
)

write.csv(
  variable_results,
  file.path(mcmc_output_dir, "varbvs_vs_gibbs_pips_by_variable.csv"),
  row.names = FALSE
)

# Copy used by p30_postprocess_tables_figures.ipynb (Table 4.3 varbvs row).
write.csv(
  variable_results,
  file.path(project_dir, "varbvs_vs_gibbs_pips_by_variable.csv"),
  row.names = FALSE
)

write.csv(
  replicate_summaries,
  file.path(mcmc_output_dir, "varbvs_vs_gibbs_summary_by_replicate.csv"),
  row.names = FALSE
)

write.csv(
  hyperparameter_summaries,
  file.path(mcmc_output_dir, "varbvs_vs_gibbs_by_hyperparameter_setting.csv"),
  row.names = FALSE
)

write.csv(
  varbvs_runtime_rows,
  file.path(mcmc_output_dir, "varbvs_runtime_by_replicate.csv"),
  row.names = FALSE
)

write.csv(
  overall_summary,
  file.path(mcmc_output_dir, "varbvs_vs_gibbs_overall_summary.csv"),
  row.names = FALSE
)

# Aggregate PIP-versus-PIP plot over all requested replicates.
png(
  file.path(mcmc_output_dir, "varbvs_vs_gibbs_pips_all_replicates.png"),
  width = 1000,
  height = 1000
)
plot(
  variable_results$gibbs_pip,
  variable_results$varbvs_pip,
  xlim = c(0, 1),
  ylim = c(0, 1),
  xlab = "Gibbs PIP",
  ylab = "varbvs PIP",
  main = "varbvs versus matched Gibbs PIPs"
)
abline(a = 0, b = 1, lty = 2)
dev.off()

saveRDS(
  list(
    variable_results = variable_results,
    replicate_summaries = replicate_summaries,
    hyperparameter_summaries = hyperparameter_summaries,
    overall_summary = overall_summary,
    controls = list(
      varbvs_version = installed_version,
      logodds_grid = logodds_grid,
      varbvs_tol = varbvs_tol,
      varbvs_maxiter = varbvs_maxiter,
      weight_coverage = weight_coverage,
      burn_sweeps = burn_sweeps,
      keep_sweeps = keep_sweeps,
      thin = thin,
      n_chains = n_chains,
      random_scan = random_scan,
      base_seed = base_seed,
      replicates_to_run = replicates_to_run
    )
  ),
  file.path(mcmc_output_dir, "varbvs_mcmc_benchmark_all_results.rds")
)

cat("\n============================================================\n")
cat("Benchmark finished.\n")
cat("============================================================\n")
print(overall_summary)
cat(
  "\nOutputs are in: ",
  mcmc_output_dir,
  "\n",
  sep = ""
)
cat(
  "\nReplicates: ",
  paste(range(replicates_to_run), collapse = "-"),
  "\n",
  sep = ""
)
