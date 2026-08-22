Sys.setenv(RETICULATE_PYTHON = "/Users/douglasyu/src/s-jSDM/work/port-feasibility/.pixi/envs/default/bin/python")
devtools::load_all("/Users/douglasyu/src/s-jSDM/sjSDM", quiet = TRUE)

set.seed(42)
X <- matrix(rnorm(100 * 3), 100, 3)
Y <- matrix(rbinom(100 * 5, 1, 0.5), 100, 5)
fit <- sjSDM(Y = Y, env = linear(data = X), biotic = bioticStruct(df = 3),
             iter = 20, sampling = 50, verbose = FALSE)

steps <- list(
  getSe       = function() invisible(getSe(fit)),
  coef_plot   = function() { p <- plot(fit); print(class(p)) },
  internal    = function() { i <- internalStructure(fit); print(class(i)) },
  assembly    = function() {
    png("/tmp/test_assembly.png"); plotAssemblyEffects(internalStructure(fit)); dev.off()
  },
  cor_heatmap = function() {
    png("/tmp/test_cor.png"); fields::image.plot(cov2cor(getCov(fit))); dev.off()
  }
)

for (nm in names(steps)) {
  res <- tryCatch({ steps[[nm]](); "ok" },
                  error = function(e) paste("FAILED:", conditionMessage(e)))
  cat(sprintf("%-12s %s\n", nm, res))
}
