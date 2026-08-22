Sys.setenv(RETICULATE_PYTHON = "/Users/douglasyu/src/s-jSDM/work/port-feasibility/.pixi/envs/default/bin/python")
devtools::load_all("/Users/douglasyu/src/s-jSDM/sjSDM", quiet = TRUE)

cat("python dir:", system.file("python", package = "sjSDM"), "\n")
inspect <- reticulate::import("inspect", convert = TRUE)
fa <- sjSDM:::pkg.env$fa
cat("Model_sjSDM loaded from:", inspect$getfile(fa$Model_sjSDM), "\n")

set.seed(42)
X <- matrix(rnorm(200 * 3), 200, 3)
Y <- matrix(rbinom(200 * 10, 1, 0.5), 200, 10)

fit <- sjSDM(Y = Y, env = linear(data = X), biotic = bioticStruct(df = 5),
             iter = 20, sampling = 50, verbose = FALSE)
cat("fit ok\n")

cv <- try(
  sjSDM_cv(Y = Y, env = linear(data = X), biotic = bioticStruct(df = 5),
           tune = "random", tune_steps = 2, CV = 2,
           iter = 20, sampling = 50, n_cores = NULL),
  silent = TRUE)
if (inherits(cv, "try-error")) {
  cat("CV FAILED:\n", conditionMessage(attr(cv, "condition")), "\n")
  pe <- reticulate::py_last_error()
  if (!is.null(pe)) cat(pe$message, "\n", pe$trace, "\n")
} else {
  cat("CV OK\n")
}
