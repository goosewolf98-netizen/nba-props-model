options(timeout = 600)
options(repos = c(CRAN = "https://cloud.r-project.org"))

dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("hoopR", quietly = TRUE)) {
  install.packages("hoopR")
}

if (!requireNamespace("hoopR", quietly = TRUE)) {
  install.packages("remotes")
  remotes::install_github("sportsdataverse/hoopR")
}

library(hoopR)

df <- hoopR::load_nba_player_box(seasons = 2024)

out <- "data/raw/nba_player_box_2024.csv"
write.csv(df, out, row.names = FALSE)

cat("Saved:", out, "\n")
cat("Rows:", nrow(df), "Cols:", ncol(df), "\n")
