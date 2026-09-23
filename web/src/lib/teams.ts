/**
 * Team codes, names and primary colours.
 *
 * Codes match `gridiron.data.team_names.CANONICAL_TEAMS` exactly -- all 32,
 * current-era only, since the pipeline already maps historical codes (OAK to
 * LV, SD to LAC, LA to LAR, JAC to JAX) before anything reaches the API.
 *
 * Colours are each club's own primary, used only as a thin accent: a left
 * border on a row, a bar under a card. They are never a fill behind text,
 * which would put contrast at the mercy of whichever team happens to be
 * playing. No logos or wordmarks are used anywhere in this app.
 */

export interface Team {
  code: string;
  city: string;
  name: string;
  color: string;
  conference: "AFC" | "NFC";
  division: string;
}

export const TEAMS: Record<string, Team> = {
  ARI: { code: "ARI", city: "Arizona", name: "Cardinals", color: "#97233f", conference: "NFC", division: "NFC West" },
  ATL: { code: "ATL", city: "Atlanta", name: "Falcons", color: "#a71930", conference: "NFC", division: "NFC South" },
  BAL: { code: "BAL", city: "Baltimore", name: "Ravens", color: "#241773", conference: "AFC", division: "AFC North" },
  BUF: { code: "BUF", city: "Buffalo", name: "Bills", color: "#00338d", conference: "AFC", division: "AFC East" },
  CAR: { code: "CAR", city: "Carolina", name: "Panthers", color: "#0085ca", conference: "NFC", division: "NFC South" },
  CHI: { code: "CHI", city: "Chicago", name: "Bears", color: "#0b162a", conference: "NFC", division: "NFC North" },
  CIN: { code: "CIN", city: "Cincinnati", name: "Bengals", color: "#fb4f14", conference: "AFC", division: "AFC North" },
  CLE: { code: "CLE", city: "Cleveland", name: "Browns", color: "#ff3c00", conference: "AFC", division: "AFC North" },
  DAL: { code: "DAL", city: "Dallas", name: "Cowboys", color: "#041e42", conference: "NFC", division: "NFC East" },
  DEN: { code: "DEN", city: "Denver", name: "Broncos", color: "#fb4f14", conference: "AFC", division: "AFC West" },
  DET: { code: "DET", city: "Detroit", name: "Lions", color: "#0076b6", conference: "NFC", division: "NFC North" },
  GB: { code: "GB", city: "Green Bay", name: "Packers", color: "#203731", conference: "NFC", division: "NFC North" },
  HOU: { code: "HOU", city: "Houston", name: "Texans", color: "#03202f", conference: "AFC", division: "AFC South" },
  IND: { code: "IND", city: "Indianapolis", name: "Colts", color: "#002c5f", conference: "AFC", division: "AFC South" },
  JAX: { code: "JAX", city: "Jacksonville", name: "Jaguars", color: "#006778", conference: "AFC", division: "AFC South" },
  KC: { code: "KC", city: "Kansas City", name: "Chiefs", color: "#e31837", conference: "AFC", division: "AFC West" },
  LAC: { code: "LAC", city: "Los Angeles", name: "Chargers", color: "#0080c6", conference: "AFC", division: "AFC West" },
  LAR: { code: "LAR", city: "Los Angeles", name: "Rams", color: "#003594", conference: "NFC", division: "NFC West" },
  LV: { code: "LV", city: "Las Vegas", name: "Raiders", color: "#a5acaf", conference: "AFC", division: "AFC West" },
  MIA: { code: "MIA", city: "Miami", name: "Dolphins", color: "#008e97", conference: "AFC", division: "AFC East" },
  MIN: { code: "MIN", city: "Minnesota", name: "Vikings", color: "#4f2683", conference: "NFC", division: "NFC North" },
  NE: { code: "NE", city: "New England", name: "Patriots", color: "#002244", conference: "AFC", division: "AFC East" },
  NO: { code: "NO", city: "New Orleans", name: "Saints", color: "#d3bc8d", conference: "NFC", division: "NFC South" },
  NYG: { code: "NYG", city: "New York", name: "Giants", color: "#0b2265", conference: "NFC", division: "NFC East" },
  NYJ: { code: "NYJ", city: "New York", name: "Jets", color: "#125740", conference: "AFC", division: "AFC East" },
  PHI: { code: "PHI", city: "Philadelphia", name: "Eagles", color: "#004c54", conference: "NFC", division: "NFC East" },
  PIT: { code: "PIT", city: "Pittsburgh", name: "Steelers", color: "#ffb612", conference: "AFC", division: "AFC North" },
  SEA: { code: "SEA", city: "Seattle", name: "Seahawks", color: "#002244", conference: "NFC", division: "NFC West" },
  SF: { code: "SF", city: "San Francisco", name: "49ers", color: "#aa0000", conference: "NFC", division: "NFC West" },
  TB: { code: "TB", city: "Tampa Bay", name: "Buccaneers", color: "#d50a0a", conference: "NFC", division: "NFC South" },
  TEN: { code: "TEN", city: "Tennessee", name: "Titans", color: "#4b92db", conference: "AFC", division: "AFC South" },
  WAS: { code: "WAS", city: "Washington", name: "Commanders", color: "#5a1414", conference: "NFC", division: "NFC East" },
};

/** Never throws: an unknown code renders in neutral grey rather than crashing. */
export function team(code: string | null | undefined): Team {
  if (code && TEAMS[code]) return TEAMS[code];
  return {
    code: code ?? "—",
    city: "",
    name: code ?? "Unknown",
    color: "#3a4356",
    conference: "AFC",
    division: "",
  };
}

export function teamColor(code: string | null | undefined): string {
  return team(code).color;
}

export function teamName(code: string | null | undefined): string {
  const found = team(code);
  return found.city ? `${found.city} ${found.name}` : found.name;
}
