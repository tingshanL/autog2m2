#!/usr/bin/env Rscript

# Single-condition mclust worker for the matched Figure 2 extension.
#
# The Python orchestrator writes one little-endian row-major float64 matrix,
# invokes this script, validates the returned metadata/labels, and removes the
# temporary files. G is fixed at 3. No reference labels are passed to R.

EXPECTED_MCLUST_VERSION <- "6.1.3"

clean_text <- function(x) {
  x <- gsub("[\t\r\n]+", " ", as.character(x))
  trimws(x)
}

write_metadata <- function(path, values) {
  keys <- names(values)
  text <- vapply(values, clean_text, character(1))
  table <- data.frame(key = keys, value = text, stringsAsFactors = FALSE)
  write.table(
    table,
    file = path,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE,
    col.names = FALSE
  )
}

require_mclust <- function() {
  if (!requireNamespace("mclust", quietly = TRUE)) {
    stop(
      "R package 'mclust' is not installed. Run install.packages('mclust').",
      call. = FALSE
    )
  }
  observed <- as.character(utils::packageVersion("mclust"))
  if (!identical(observed, EXPECTED_MCLUST_VERSION)) {
    stop(
      sprintf(
        "Expected mclust %s, found %s. Update/freeze the R environment first.",
        EXPECTED_MCLUST_VERSION,
        observed
      ),
      call. = FALSE
    )
  }

  # Mclust() constructs and evaluates a call to mclustBIC() by name. Attach
  # the package so that lookup succeeds when this helper is launched with
  # Rscript; loading only the namespace is insufficient for that internal call.
  suppressPackageStartupMessages(
    library("mclust", character.only = TRUE)
  )
  observed
}

args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 1L && identical(args[[1]], "--check-only")) {
  version <- require_mclust()
  cat("status\tok\n", sep = "")
  cat("mclust_version\t", version, "\n", sep = "")
  cat("r_version\t", R.version.string, "\n", sep = "")
  quit(save = "no", status = 0L)
}

if (length(args) != 7L) {
  stop(
    paste(
      "Usage: fit_mclust_fixed_k.R",
      "INPUT_BINARY N_ROWS N_COLUMNS SEED LABELS_BINARY METADATA_TSV G"
    ),
    call. = FALSE
  )
}

input_path <- normalizePath(args[[1]], mustWork = TRUE)
n_rows <- as.integer(args[[2]])
n_columns <- as.integer(args[[3]])
seed <- as.integer(args[[4]])
labels_path <- args[[5]]
metadata_path <- args[[6]]
fixed_g <- as.integer(args[[7]])

if (
  is.na(n_rows) || is.na(n_columns) || is.na(seed) || is.na(fixed_g) ||
    n_rows < 2L || n_columns < 1L || fixed_g != 3L
) {
  stop("Invalid dimensions, seed, or fixed G; G must equal 3.", call. = FALSE)
}

mclust_version <- require_mclust()

connection <- file(input_path, open = "rb")
values <- tryCatch(
  readBin(
    connection,
    what = "double",
    n = n_rows * n_columns,
    size = 8L,
    endian = "little"
  ),
  finally = close(connection)
)

if (length(values) != n_rows * n_columns || any(!is.finite(values))) {
  stop("Input matrix is truncated or nonfinite.", call. = FALSE)
}

# NumPy wrote C-order rows; reconstruct those rows explicitly in R.
X <- matrix(values, nrow = n_rows, ncol = n_columns, byrow = TRUE)

# These are mclust's documented defaults. For n <= d, mclust restricts the
# search to spherical and diagonal models because unrestricted ellipsoidal
# covariance models are not estimable in the ambient space.
if (n_rows > n_columns) {
  model_names <- mclust::mclust.options("emModelNames")
  model_policy <- "documented multivariate default for n>d"
} else {
  model_names <- c("EII", "VII", "EEI", "EVI", "VEI", "VVI")
  model_policy <- "documented spherical/diagonal default for n<=d"
}

set.seed(seed)
captured_warnings <- character(0)
started <- proc.time()[["elapsed"]]
fit <- withCallingHandlers(
  mclust::Mclust(
    data = X,
    G = fixed_g,
    modelNames = model_names,
    verbose = FALSE
  ),
  warning = function(warning_condition) {
    captured_warnings <<- c(
      captured_warnings,
      conditionMessage(warning_condition)
    )
    invokeRestart("muffleWarning")
  }
)
elapsed <- proc.time()[["elapsed"]] - started

if (
  is.null(fit) || is.null(fit$classification) ||
    length(fit$classification) != n_rows || fit$G != fixed_g
) {
  stop("mclust returned no valid fixed-G=3 classification.", call. = FALSE)
}

labels <- as.integer(fit$classification)
if (any(is.na(labels)) || length(unique(labels)) != fixed_g) {
  stop("mclust returned missing labels or fewer than three occupied groups.", call. = FALSE)
}

label_connection <- file(labels_path, open = "wb")
tryCatch(
  writeBin(labels, label_connection, size = 4L, endian = "little"),
  finally = close(label_connection)
)

format_number <- function(value) {
  if (length(value) != 1L || !is.finite(value)) {
    return("NA")
  }
  format(value, digits = 17L, scientific = TRUE)
}

metadata <- list(
  status = "ok",
  selected_components = as.character(fit$G),
  occupied_components = as.character(length(unique(labels))),
  selected_model = fit$modelName,
  bic = format_number(fit$bic),
  bic_orientation = "higher_is_better",
  log_likelihood = format_number(fit$loglik),
  degrees_of_freedom = as.character(fit$df),
  icl = format_number(fit$icl),
  elapsed_seconds = format(elapsed, digits = 17L, scientific = TRUE),
  model_policy = model_policy,
  candidate_models = paste(model_names, collapse = ","),
  mclust_version = mclust_version,
  r_version = R.version.string,
  warnings = paste(unique(captured_warnings), collapse = " || ")
)
write_metadata(metadata_path, metadata)
