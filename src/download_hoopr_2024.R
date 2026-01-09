options(timeout = 600)
dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)

# Install hoopR if missing
if (!requireNamespace("hoopR", quietly = TRUE)) {
  install.packages(
    "hoopR",
    repos = c("https://sportsdataverse.r-universe.dev", "https://cloud.r-project.org")
  )
}

library(hoopR)

# 2024 = 2024–25 season (season start year)
df <- hoopR::load_nba_player_box(seasons = 2024)

out <- "data/raw/nba_player_box_2024.csv"
write.csv(df, out, row.names = FALSE)

cat("Saved:", out, "\n")
cat("Rows:", nrow(df), "Cols:", ncol(df), "\n")
