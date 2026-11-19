import { useEffect, useState } from "react";
import BacktestPage from "./BacktestPage";
import ResearchPage from "./components/research/ResearchPage";
import { localISODate } from "./lib/date";

type Route = "backtest" | "research";
const routeFromHash = (): Route => (window.location.hash.startsWith("#/research") ? "research" : "backtest");

export default function App() {
  const [route, setRoute] = useState<Route>(routeFromHash);
  useEffect(() => {
    const onHash = () => setRoute(routeFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const search = window.location.search;  // keep ?demo=1 across tabs
  const tabs: { route: Route; href: string; label: string }[] = [
    { route: "backtest", href: `${search}#/`, label: "Backtest" },
    { route: "research", href: `${search}#/research`, label: "Research" },
  ];

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-10 border-b border-line bg-canvas/75 backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-[1280px] items-center gap-3 px-4 md:px-6">
          <span className="num rounded-md border border-line-strong px-2 py-0.5 text-sm font-medium tracking-wide text-accent">R.A.P.O</span>
          <span className="hidden text-sm text-fg-2 md:inline">Robo-Advisor Portfolio Optimizer</span>
          <nav aria-label="Main" className="ml-auto flex gap-1">
            {tabs.map((t) => (
              <a key={t.route} href={t.href} aria-current={route === t.route ? "page" : undefined}
                className={`inline-flex min-h-11 items-center rounded-md px-3 text-sm ${
                  route === t.route ? "bg-raised text-fg" : "text-fg-2 hover:text-fg"}`}>
                {t.label}
              </a>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto flex max-w-[1280px] flex-col gap-5 px-4 py-6 md:px-6 md:py-8">
        {route === "research" ? <ResearchPage today={localISODate()} /> : <BacktestPage />}
      </main>
    </div>
  );
}
