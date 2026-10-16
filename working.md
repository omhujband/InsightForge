# How to run and use the project

## 1. Requirements
* Python **3.10 – 3.13** (developed and tested on 3.12), and `pip`.
* An internet connection (for LLM calls and `pip install`).
* *Optional but recommended:* a free LLM key (Groq or Gemini) — see step 3. Without one, the app still runs,
  but the 🤖 features are disabled.

## 2. Install
```bash
cd ai-data-analyst
python -m venv venv

# Windows (PowerShell)         macOS / Linux
venv\Scripts\Activate.ps1      source venv/bin/activate
# (if PowerShell blocks it:  Set-ExecutionPolicy -Scope Process Bypass)

pip install -r requirements.txt
```

## 3. Get a free LLM key (5 minutes)
Pick **one** (both is better — they back each other up):
* **Groq:** https://console.groq.com → API Keys → create key.
* **Google Gemini:** https://aistudio.google.com → Get API key.

Then copy the example settings file and paste your key:
```bash
copy .env.example .env        # Windows        (macOS/Linux: cp .env.example .env)
```
Edit `.env`:
```
GROQ_API_KEY=your_key_here
GOOGLE_API_KEY=your_other_key_here     # optional
```
Several keys for one provider? Comma-separate them: `GROQ_API_KEY=key1,key2`. If one is rate-limited the app
automatically uses the next. A Google key also enables embedding-based document search (otherwise an offline
TF-IDF search is used — both work).

## 4. Run
```bash
streamlit run app.py
```
Open http://localhost:8501. The sidebar shows which model is active (or a warning if no key was found).

## 5. Modes
| Mode | When | What you get |
|---|---|---|
| **Demo mode** | `SUPABASE_URL`/`SUPABASE_KEY` empty (default) | No login. A **Demo role** dropdown in the sidebar lets you show what Admin / Manager / Analyst / Viewer each see. |
| **Supabase mode** | both variables set | Real email/password login; role comes from the user's account. |

### Setting up Supabase login (optional, ~10 min)
1. https://supabase.com → new project → *Project Settings → API*: copy **Project URL** and the **anon public** key
   into `.env` as `SUPABASE_URL` and `SUPABASE_KEY` (never use the `service_role` key).
2. *Authentication → Providers → Email*: for a quick demo you can turn **off** "Confirm email".
3. Restart the app, sign up in the UI. New users get `DEFAULT_ROLE` (default `analyst`).
4. To make someone Admin/Manager/Viewer, run this in Supabase *SQL Editor*:
   ```sql
   update auth.users
   set raw_app_meta_data = coalesce(raw_app_meta_data,'{}'::jsonb) || '{"role":"admin"}'::jsonb
   where email = 'you@example.com';
   ```
   (valid roles: `admin`, `manager`, `analyst`, `viewer`). They must log out and in again.

## 6. Using the app
1. **Load data:** sidebar → upload files, or switch on **Use sample data** (sales + customers tables and a short
   company report).
2. **Pick a dataset** at the top of the page.
3. Work through the tabs (you only see the ones your role allows):

| Tab | Try this |
|---|---|
| 📊 Dashboard | Look at KPIs, change the distribution/category/date selectors, open *Column profile*. |
| 🤖 AI Analyst | Click a suggested question, or ask: *"Which region has the highest total revenue?"*. Open **Generated code**. Then ask a follow-up: *"show only the top 2"*. |
| 🤖 Documents | Upload a PDF/TXT (or use the sample) and ask: *"What is the discount policy?"*. Open **Retrieved passages**. |
| ⚡ Evaluation | After a few questions see success rate/attempts. Click **Run benchmark** for a real accuracy score. |
| 🗄 SQL | SQLite is built into Python (nothing to install). Your data is the table **`data`** (or the file name, e.g. `sample_sales`). Run `SELECT * FROM data LIMIT 10`, or ask *"total revenue by category"* (needs an API key), edit the SQL, run it. Try `delete from data` to see it blocked. |
| 🧹 Cleaning | Tick the steps, **Apply**, read the audit log, then **Use cleaned data for analysis**. |
| 🔮 Forecast | Date = `order_date`, target = `revenue`, Monthly → **Generate**. Changing *periods to forecast* only adds more future rows (the first values stay the same) and changes the backtest window; changing target / frequency / aggregation changes the numbers. Read the backtest vs. naive. |
| 🤖 Root Cause | *"Revenue dropped recently"* → see computed evidence vs. hypotheses. |
| 🔗 Relationships | Sample data: detects `customer_id`; preview the join and add it as a dataset. |
| 📈 Statistics | **Correlation:** how strongly columns move together (-1 to 1). **Outliers:** values beyond Q1 − k·IQR / Q3 + k·IQR; a bigger multiplier moves the dashed bounds outward and flags fewer points. **Summary:** count/mean/min/max per column. |
| 🤖 Copilot | *"What are the biggest opportunities?"* |
| 📄 Reports | The PDF contains everything you ran on the selected dataset this session: overview + column profile, every AI analysis (result, code, interpretation), forecast, cleaning log, SQL, root cause, Copilot and Document Q&A. Run those tabs first, tick the sections you want, then build. Charts are not embedded. |

Tabs marked 🤖 need an LLM key; all others work without one.

## 7. Suggested 5-minute demo flow (for the presentation)
1. Start in **Demo mode**, toggle **Use sample data**, show Dashboard.
2. **AI Analyst**: ask a question → show answer, chart, **generated code**, the *number check* line.
3. Ask a follow-up. Mention: *"the AI writes code; the numbers are computed in a sandbox"*.
4. **Cleaning** → apply → switch to cleaned data → ask the same question again.
5. **Forecast** → show the backtest metrics (explain why it's honest).
6. **Documents** question with citations.
7. **Evaluation → Run benchmark** → accuracy score.
8. **Reports** → download PDF. Switch the demo role to *Viewer* to show access control.

## 8. Configuration (all optional, in `.env`)
| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY` | – | LLM keys (comma-separate several) |
| `GROQ_MODEL`, `GEMINI_MODEL`, … | see `config.py` | Override model names if a provider retires one |
| `LLM_PROVIDER_ORDER` | `groq,gemini,openrouter,openai,xai` | Which provider to try first |
| `SUPABASE_URL`, `SUPABASE_KEY` | – | Enables real login |
| `DEFAULT_ROLE` | `analyst` | Role for accounts with no role set |
| `MAX_UPLOAD_MB` | 50 | Upload size limit |
| `SANDBOX_TIMEOUT_S` | 20 | Max seconds generated code may run |
| `SANDBOX_MAX_MEMORY_MB` | 2048 | Memory cap for generated code (Linux/macOS; ignored on Windows) |
| `MAX_REPAIR_ATTEMPTS` | 2 | Extra tries when generated code fails |

Optional Prophet forecasting model: `pip install prophet`, restart, and it appears in the Forecast model list.

## 9. Troubleshooting
| Symptom | Cause / fix |
|---|---|
| Sidebar says *No LLM key configured* | `.env` missing or key name wrong. It must be in the project folder, named exactly `.env`. Restart the app after editing. |
| *All configured AI models failed … 429* | Free-tier rate limit. Wait a minute, or add a second provider/key. |
| *Model not found* error | A provider renamed a model — set `GROQ_MODEL` / `GEMINI_MODEL` in `.env`. |
| AI answer says *Could not produce a reliable answer* | The model's code failed after retries. Rephrase more specifically (name the columns), or check the shown code and error. |
| Supabase: *Could not reach Supabase* | Check internet and that `SUPABASE_URL` is the full `https://xxxx.supabase.co`. |
| Supabase: *Please confirm your email* | Confirm via the email, or disable "Confirm email" in Supabase for demos. |
| `streamlit` not recognised | Activate the venv first (step 2), or run `python -m streamlit run app.py`. |
| Port already in use | `streamlit run app.py --server.port 8502`. |
| Slow on big files | Keep demo files under ~100k rows; the app recomputes dashboard widgets on each interaction. |

## 10. What has and hasn't been tested
* **Tested (automated, headless):** all four roles render; sandbox blocks file access/imports/introspection and
  enforces timeouts; read-only SQL; cleaning; forecasting & backtest; RAG with real PDF text extraction;
  PDF/text reports; Supabase login gate and role mapping (with a mocked Supabase); provider failover; the whole
  question→code→repair→answer pipeline using a *scripted stand-in* for the LLM.
* **Not tested:** calls to real LLM providers and a real Supabase project (needs your keys). Before presenting,
  spend 5 minutes: run the demo flow above once with your key, and use **Evaluation → Run benchmark** to see how
  your chosen model performs.
