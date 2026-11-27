args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6) stop("Expected input.csv output.csv metadata.json G model seed")
if (!requireNamespace("mclust", quietly = TRUE)) stop("Install mclust first; see README.md")
suppressPackageStartupMessages(library(mclust))
data <- read.csv(args[1], row.names = 1, check.names = FALSE)
x <- as.matrix(data)
storage.mode(x) <- "double"
if (any(!is.finite(x))) stop("Embedding contains non-finite values")
set.seed(as.integer(args[6]))
fit <- mclust::Mclust(x, G = as.integer(args[4]), modelNames = args[5], verbose = FALSE)
if (is.null(fit) || is.null(fit$classification)) stop("mclust failed to fit requested model")
if (length(unique(fit$classification)) != as.integer(args[4])) stop("mclust did not return requested number of clusters")
write.csv(data.frame(barcode = rownames(x), prediction = fit$classification), args[2], row.names = FALSE)
# Base R only: avoid requiring jsonlite just to record environment metadata.
metadata <- sprintf('{"mclust_version":"%s","R_version":"%s","model":"%s","G":%d,"seed":%d,"loglik":%.16g}',
                    as.character(utils::packageVersion("mclust")), as.character(getRversion()),
                    fit$modelName, fit$G, as.integer(args[6]), fit$loglik)
writeLines(metadata, args[3])
