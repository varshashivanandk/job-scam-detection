// A pasted job link: the whole text is one http(s) URL, with optional whitespace around it. Same rule as the
// server (backend/services/url_fetch.py, single_url), so the page shows "Fetching page..." exactly when the
// server will fetch.
const URL_ONLY = /^\s*(https?:\/\/\S+)\s*$/i;

export const FETCHING_LABEL = "Fetching page";
export const ANALYZING_LABEL = "Analyzing";

export function singleUrl(text) {
  if (typeof text !== "string") return null;
  const match = URL_ONLY.exec(text);
  return match ? match[1] : null;
}

// The waiting message for a pasted text: fetching a link takes longer than analyzing text.
export function busyLabel(text) {
  return singleUrl(text) ? FETCHING_LABEL : ANALYZING_LABEL;
}
