options(timeout = 600)
options(repos = c(CRAN = "https://cloud.r-project.org"))

dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)

# Install hoopR if missing (try CRAN, then GitHub fallback)
if (!requireNamespace("hoopR", quietly = TRUE)) {
  install.packages("hoopR")
}
if (!requireNamespace("hoopR", quietly = TRUE)) {
  install.packages("remotes")
  remotes::install_github("sportsdataverse/hoopR")
}

library(hoopR)

# IMPORTANT:
# hoopR expects the SEASON END YEAR.
# 2025 = 2024–25 season
# 2026 = 2025–26 season
seasons_to_load <- c(2025, 2026)

df <- hoopR::load_nba_player_box(seasons = seasons_to_load)

out <- "data/raw/nba_player_box_2024_25_and_2025_26.csv"
write.csv(df, out, row.names = FALSE)

cat("Saved:", out, "\n")
cat("Rows:", nrow(df), "Cols:", ncol(df), "\n")
