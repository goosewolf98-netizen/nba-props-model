# src/download_hoopr_2024.R
# Downloads NBA player box scores for the 2024 season start year (2024–25)
# and saves a CSV we can train on.

options(timeout = 600)

# Install hoopR from SportsDataverse R-universe if not present
if (!requireNamespace("hoopR", quietly = TRUE)) {
  install.packages(
    "hoopR",
    repos = c("https://sportsdataverse.r-universe.dev", "https://cloud.r-project.org")
  )
}

if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages("dplyr", repos = "https://cloud.r-project.org")
}

library(hoopR)
library(dplyr)

dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)

# This loads NBA player box scores from the hoopR data repo
# seasons = 2024 corresponds to the 2024–25 NBA season in most hoopR examples
nba_player_box <- hoopR::load_nba_player_box(seasons = 2024)

# Save CSV for Python training
out_path <- "data/raw/nba_player_box_2024.csv"
write.csv(nba_player_box, out_path, row.names = FALSE)

cat("Saved:", out_path, "\n")
cat("Rows:", nrow(nba_player_box), "Cols:", ncol(nba_player_box), "\n")
