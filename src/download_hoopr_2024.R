options(timeout = 600)
options(repos = c(CRAN = "https://cloud.r-project.org"))

dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)

# 1) Try CRAN install first (most reliable)
if (!requireNamespace("hoopR", quietly = TRUE)) {
  install.packages("hoopR")
}

# 2) If CRAN didn't work, fallback to GitHub
if (!requireNamespace("hoopR", quietly = TRUE)) {
  install.packages("remotes")
  remotes::install_github("sportsdataverse/hoopR")
}

library(hoopR)

# 2024 = 2024–25 season (season start year)
df <- hoopR::load_nba_player_box(seasons = 2024)

out <- "data/raw/nba_player_box_2024.csv"
write.csv(df, out, row.names = FALSE)

cat("Saved:", out, "\n")
cat("Rows:", nrow(df), "Cols:", ncol(df), "\n")
