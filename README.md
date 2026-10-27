# ICP Checker (prototype)

Scores sales leads 0–100 against an Ideal Customer Profile (ICP) and finds the right stakeholder when the
lead isn't the buyer. Built as a standalone prototype for **AI Associate**. It's meant to show whether the
approach works, so it is not production-hardened.

- **Model calls:** Azure OpenAI only, through one wrapper (`llm.py`), with strict JSON-schema output and Pydantic validation.
- **Web data:** Serper only (search, news and scrape). **LinkedIn pages are never fetched.** LinkedIn data comes
  only from Google result titles and snippets. `serper.py` enforces this: `scrape()` and every HTTP client reject
  `*.linkedin.com`.
- **Scoring:** a deterministic rules engine (`scoring.py`). The LLM only extracts and normalizes facts.
- **Evidence:** every fact carries its source URL. Facts whose URL wasn't in the provided text are dropped.
- **Storage:** local JSON files under `data/`, with disk caches for Serper (7 days), LLM responses and company
  research (30 days). There is no database, Docker or auth.

## Setup

```bash
uv sync
cp .env.example .env   # fill in Azure OpenAI + Serper keys
```

Azure needs a deployment that supports structured outputs (e.g. gpt-4o 2024-08-06+, gpt-4.1, gpt-4o-mini) and
API version `2024-08-01-preview` or later (default `2024-10-21`).

## Run

**Web UI** (draft/edit/approve an ICP; then either upload a CSV or use the **Single lead** tab to enter one
LinkedIn profile + company for quick testing; watch rows score live, override, re-score, export):

```bash
uv run uvicorn icp_checker.api:app --reload
# open http://127.0.0.1:8000  (API docs at /docs)
```

**CLI** (quickest way to try a list):

```bash
uv run icp-checker draft-icp examples/offering_hr_talent.txt --team hr_talent -o my_icp.json   # optional
uv run icp-checker run --icp examples/icp_hr_talent.json --csv examples/leads_sample.csv -o scored.csv
```

The 5-row sample uses about 25–40 Serper credits on a cold cache. Re-running it is free because both caches hit.

**Tests** (all mocked, no keys needed): `uv run pytest`

## How a list is processed

1. **Map columns.** Headers are matched heuristically (name/title/company/website/linkedin/email) and the user
   confirms the mapping in the UI. Rows are deduplicated.
   **LinkedIn column (optional):** put a profile URL (`https://in.linkedin.com/in/jane-doe`) or just the ID
   (`jane-doe`) in a column such as `LinkedIn`. A row can contain *only* that. Each profile costs one Serper search
   (`site:linkedin.com/in/<id>`), and the name, title and company are read from the Google result. The profile page is
   never opened. Values already in the list take priority. If the result shows no company, add it to the row.
   Try it: `uv run icp-checker run --icp examples/icp_hr_talent.json --csv examples/leads_linkedin_sample.csv`
2. **Group by company.** Rows are grouped by domain (from website or work email), else by normalized name, and each
   company is researched once.
3. **Research the company** (Serper):
   - Resolve the domain if it's missing.
   - Scrape up to 3 pages: the homepage, plus careers and about pages if found.
   - Get 6 months of news snippets and an employee-count snippet.
4. **Extract facts** (LLM `company_extract`): industry plus whether it matches the ICP, size, locations and need
   signals, each with a source URL.
5. **Company gate.** An excluded industry or location, or a company fit below `company_fit_min_pct`, makes the lead
   `not_a_fit`. Person and stakeholder searches are skipped for those leads.
6. **Check the person.** Takes the list title plus one `site:linkedin.com/in "name" "company"` search, which only
   verifies or fills the title from snippets. The LLM `title_normalize` step then sets function, seniority and role
   type.
7. **Find stakeholders** (only if the contact is weak, isn't a buyer, or is missing):
   - A LinkedIn-site search for the ICP's top role titles, using snippets only.
   - Fallback: the company's leadership page.
   - Candidates are scored with the same engine and the top 3 are kept.
8. **Score and categorize.** Categories are ideal / good / wrong_contact / contact_not_found / not_a_fit. Then one
   LLM call writes the explanation and recommendation from the breakdown.

**Re-score** (`POST /lists/{id}/rescore` or the button) runs the rules engine against another approved ICP version
using stored facts. Serper runs in offline mode, so no new web calls are made.

## API

| Method | Path | |
|---|---|---|
| POST | `/icps/draft` | offering text (+ answers, won companies) → draft ICP |
| POST | `/icps` | save draft (include `id` to update) |
| POST | `/icps/{id}/approve` | snapshot the draft as a new immutable version |
| GET | `/icps`, `/icps/{id}`, `/icps/{id}/versions` | |
| POST | `/lists` | upload CSV → proposed column mapping |
| POST | `/lists/{id}/start` | `{icp_id, version?, mapping?}` → background job |
| POST | `/leads/single` | `{icp_id, version?, linkedin_url?, company?, name?, title?, website?}`: score one lead entered by hand (runs as a one-row list) |
| GET | `/lists/{id}` | progress, summary counts, Serper/LLM usage |
| GET | `/lists/{id}/leads?category=&sort=score` | ranked leads with evidence and stakeholders |
| POST | `/lists/{id}/leads/{lead_id}/override` | `{category, reason}` |
| POST | `/lists/{id}/rescore` | `{icp_id, version}`, no new research |
| GET | `/lists/{id}/export.csv` | original columns + Category, Score, Confidence, Explanation, Recommendation, Evidence, Stakeholders 1–3 |
| DELETE | `/lists/{id}` | removes the list, its leads and its stakeholders |

## Layout

```
src/icp_checker/
  config.py        all limits, TTLs, deployments (env / .env)
  models.py        ICP, Company, Lead, Score, Stakeholder, Evidence
  serper.py        Serper client + LinkedIn guard + retry/backoff + per-company call cap
  llm.py           Azure OpenAI wrapper (strict JSON schema, validate, retry once, else unknown)
  llm_schemas.py   output model per prompt
  prompts/         one Jinja template per prompt
  research.py      company research (Serper) + fact extraction (LLM)
  people.py        person check + stakeholder finder
  scoring.py       deterministic rules engine
  pipeline.py      background job, re-score, CSV export, summary
  icp_service.py   ICP drafting + versioning
  api.py, cli.py, static/index.html
tests/             rules engine, LinkedIn guard, normalization, clients, end-to-end pipeline (mocked)
```

## Not in the prototype yet

- Google Sheets input and write-back (CSV only for now)
- Rep-assisted LinkedIn capture extension (milestone 5)
- Admin override report and calibration script
- Production storage, auth and multi-process job queue
