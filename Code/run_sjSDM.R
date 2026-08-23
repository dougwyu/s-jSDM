# Walkthrough: fitting sjSDM models using the pixi Python environment
#
# This script uses the PyTorch 2.5.1 environment installed at
# work/mojo-backend/.pixi/envs/default (NOT the old r-sjsdm conda env).
# Run top-to-bottom, e.g. with source("Code/run_sjSDM.R") or interactively.

## ---- 0. Point reticulate at the pixi Python -------------------------------
# Must happen BEFORE library(sjSDM). The package would otherwise pick up
# the r-miniconda "r-sjsdm" env, which we are avoiding.
# here() resolves the repo root via sjSDM.Rproj, so it works regardless
# of the current working directory.
Sys.setenv(RETICULATE_PYTHON = here::here(
  "work/mojo-backend/reticulate-venv/bin/python"
))
stopifnot(file.exists(Sys.getenv("RETICULATE_PYTHON")))

# load_all() runs the REPO code (including our patched Python backend in
# sjSDM/inst/python), not the copy in your R library. Re-running this
# section picks up edits without reinstalling.
devtools::load_all(here::here("sjSDM"), quiet = TRUE)

## ---- 1. Simulate example data ---------------------------------------------
# 200 sites, 3 environmental predictors, 10 species.
set.seed(42)
n_sites   <- 200
n_species <- 10

X <- matrix(rnorm(n_sites * 3), n_sites, 3)
colnames(X) <- c("temp", "precip", "elev")

# True species coefficients and associations
beta <- matrix(rnorm(3 * n_species), 3, n_species)
Sigma <- matrix(0.5, n_species, n_species)
diag(Sigma) <- 1

# Latent multivariate-normal draws -> binomial responses
library(mvtnorm)
logit_p <- cbind(1, X) %*% rbind(rnorm(n_species), beta)
Z <- rmvnorm(n_sites, sigma = Sigma)
Y <- matrix(rbinom(n_sites * n_species, 1, plogis(logit_p + Z)),
            n_sites, n_species)

## ---- 2. Fit a basic linear sjSDM ------------------------------------------
# iter / sampling / batch_size are the key speed knobs.
fit <- sjSDM(
  Y = Y,
  env = linear(data = X, formula = ~ temp + precip + elev),
  biotic = bioticStruct(df = 5),   # low-rank covariance factor
  iter = 100,
  sampling = 100,
  # batch size is set internally (~10% of sites); see ?sjSDMControl
  # for optimizer, LR scheduling and early-stopping options
  verbose = FALSE
)

summary(fit)

## ---- 3. Inspect the fitted species-association structure ------------------
cov_mat <- getCov(fit)          # estimated species covariance matrix
round(cov_mat, 2)

## ---- 4. Predictions --------------------------------------------------------
pred <- predict(fit)            # marginal occurrence probabilities
dim(pred)                             # sites x species
head(pred[, 1:4])

## ---- 5. Model fit metrics ---------------------------------------------------
Rsquared(fit)                         # Nagelkerke-style R2

## ---- 6. Variation partitioning (slower: refits submodels) ------------------
anova(fit)

## ---- 7. Cross-validation over regularization strength ---------------------
# NOTE: sjSDM_cv() takes raw data (Y + env), NOT a fitted model.
cv <- sjSDM_cv(
  Y = Y,
  env = linear(data = X),
  biotic = bioticStruct(df = 5),
  tune = "random", tune_steps = 5,
  CV = 3, iter = 100, sampling = 100,
  n_cores = 2
)
plot(cv)

## ---- 8. Spatial model ------------------------------------------------------
coords <- matrix(runif(n_sites * 2), n_sites, 2)
SP <- generateSpatialEV(coords)     # spatial eigenvectors
fit_sp <- sjSDM(Y = Y,
                env = linear(data = X),
                spatial = linear(data = SP),
                iter = 100, verbose = FALSE)
anova(fit_sp)                       # partition env vs space vs associations

## ---- 9. Plot outputs -------------------------------------------------------
# plot.sjSDM draws a ggplot2 coefficient plot with error bars.
# NOTE: it computes standard errors first (per-species Hessians),
# which is by far the most expensive post-fitting step.
dir.create(here::here("Code", "plots"), showWarnings = FALSE)

p_coef <- plot(fit)
ggsave(here::here("Code/plots/coefficients.png"), p_coef,
       width = 10, height = 8)

# Assembly effects: species responses along environmental gradients.
# NOTE: internalStructure() takes an anova object from a SPATIAL model,
# not the model itself.
intstr <- internalStructure(anova(fit_sp))
png(here::here("Code/plots/assembly.png"), width = 1200, height = 900)
plotAssemblyEffects(intstr)
dev.off()

# Species-species correlation heatmap
png(here::here("Code/plots/cor_heatmap.png"), width = 800, height = 800)
fields::image.plot(cov2cor(getCov(fit)),
                   main = "Estimated species correlations",
                   xlab = "species", ylab = "species")
dev.off()

cat("plots written to Code/plots/\n")

# NOTE: the weak-vs-strong association recovery experiment that used to live
# here has moved to Code/sjSDM_mojo_tutorial.Rmd ("Do fits recover known
# association strength?").
