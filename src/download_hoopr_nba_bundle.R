options(timeout = 1200)
options(repos = c(CRAN = "https://cloud.r-project.org"))

dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("hoopR", quietly = TRUE)) install.packages("hoopR")
if (!requireNamespace("hoopR", quietly = TRUE)) {
  install.packages("remotes")
  remotes::install_github("sportsdataverse/hoopR")
}
library(hoopR)

# hoopR uses SEASON END YEAR:
# 2025 = 2024-25, 2026 = 2025-26
seasons_to_load <- c(2025, 2026)

player_box <- hoopR::load_nba_player_box(seasons = seasons_to_load)
team_box   <- hoopR::load_nba_team_box(seasons = seasons_to_load)
sched      <- hoopR::load_nba_schedule(seasons = seasons_to_load)
pbp        <- hoopR::load_nba_pbp(seasons = seasons_to_load)

write.csv(player_box, "data/raw/nba_player_box.csv", row.names = FALSE)
write.csv(team_box,   "data/raw/nba_team_box.csv",   row.names = FALSE)
write.csv(sched,      "data/raw/nba_schedule.csv",   row.names = FALSE)
write.csv(pbp,        "data/raw/nba_pbp.csv",        row.names = FALSE)

cat("Saved bundle files:\n")
cat("player_box:", nrow(player_box), "rows\n")
cat("team_box:",   nrow(team_box),   "rows\n")
cat("schedule:",   nrow(sched),      "rows\n")
cat("pbp:",        nrow(pbp),        "rows\n")
