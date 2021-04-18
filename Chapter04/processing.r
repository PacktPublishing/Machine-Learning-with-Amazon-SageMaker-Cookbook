library(readr)
library("argparse")

parser <- ArgumentParser()
parser$add_argument("--sample-argument", default=1L)
args <- parser$parse_args()
print(args)

filename <- "/opt/ml/processing/input/dataset.processing.csv"
df <- read_csv(filename)

print(df)

cat("output",file="/opt/ml/processing/output/output.csv",sep="\n")