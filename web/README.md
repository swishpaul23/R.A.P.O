# R.A.P.O web

React + TypeScript frontend for the R.A.P.O backtester (Vite, Tailwind v4, lightweight-charts, Lucide).

```bash
# terminal 1, repo root: start the API
pip install -r requirements.txt
uvicorn api.main:app --reload

# terminal 2: start the web app (proxies /api to :8000)
cd web
npm install
npm run dev        # http://localhost:5173
```

- `npm test` runs the unit and component tests (Vitest + Testing Library).
- `http://localhost:5173/?demo=1` runs the UI on bundled synthetic data with no API. It's useful for UI work and screenshots. The demo numbers are random walks, not market data.

Design rules live in `../.ux-profile.md` (page pattern: Analytics / Reporting, report runner).
