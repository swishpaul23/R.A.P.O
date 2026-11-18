import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { FIXTURE } from "./lib/fixture";

vi.mock("./components/EquityChart", () => ({ default: () => <div data-testid="equity-chart" /> }));

afterEach(() => vi.restoreAllMocks());

describe("App", () => {
  it("shows every parameter with a visible label", () => {
    render(<App />);
    for (const label of ["Start", "End", "Tickers", "Benchmark", "Lookback (days)", "Rebalance (days)", "Cost (bps)"]) {
      expect(screen.getByLabelText(label)).toBeInTheDocument();
    }
  });

  it("does not call the API and shows an inline error when tickers are empty", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    render(<App />);
    const user = userEvent.setup();
    await user.clear(screen.getByLabelText("Tickers"));
    await user.click(screen.getByRole("button", { name: /run backtest/i }));
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(screen.getByText("Add at least one ticker.")).toBeInTheDocument();
  });

  it("renders KPIs and the comparison table after a run", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify(FIXTURE), { status: 200, headers: { "Content-Type": "application/json" } }),
    );
    render(<App />);
    await userEvent.setup().click(screen.getByRole("button", { name: /run backtest/i }));
    const table = await screen.findByRole("table", { name: /strategy comparison/i });
    expect(within(table).getByText("HRP")).toBeInTheDocument();
    expect(within(table).getByText("SPY (benchmark)")).toBeInTheDocument();
    expect(screen.getByRole("region", { name: /key metrics/i })).toHaveTextContent("CAGR");
  });

  it("tells the user when results come from the saved snapshot instead of live data", async () => {
    const snap = { ...FIXTURE, meta: { ...FIXTURE.meta, data_source: "snapshot" } };
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify(snap), { status: 200, headers: { "Content-Type": "application/json" } }),
    );
    render(<App />);
    await userEvent.setup().click(screen.getByRole("button", { name: /run backtest/i }));
    expect(await screen.findByRole("status")).toHaveTextContent(/saved snapshot/i);
  });

  it("shows the API's error message when a run fails", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "No price data for: ZZZZ. Check the ticker symbols." }), { status: 400 }),
    );
    render(<App />);
    await userEvent.setup().click(screen.getByRole("button", { name: /run backtest/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("ZZZZ");
  });
});
