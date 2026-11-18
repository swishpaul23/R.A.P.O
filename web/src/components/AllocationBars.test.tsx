import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { FIXTURE } from "../lib/fixture";
import type { BacktestResponse } from "../lib/api";
import AllocationBars from "./AllocationBars";

const data: BacktestResponse = {
  ...FIXTURE,
  weights: { "Max Sharpe": { TLT: 0, GLD: 0, CRWD: 0.4, AMD: 0.397, SPY: 0.096, TSM: 0.107, NBIS: 0 } },
  eligible: ["TLT", "GLD", "CRWD", "AMD", "SPY", "TSM"],
  clusters: null,
};

describe("AllocationBars", () => {
  it("lists only held assets as bars, largest first", () => {
    render(<AllocationBars data={data} focus="Max Sharpe" />);
    const rows = within(screen.getByRole("list", { name: /target weights/i })).getAllByRole("listitem");
    expect(rows.map((r) => r.getAttribute("data-ticker"))).toEqual(["CRWD", "AMD", "TSM", "SPY"]);
  });

  it("names assets the strategy chose not to hold", () => {
    render(<AllocationBars data={data} focus="Max Sharpe" />);
    expect(screen.getByText(/not held/i).parentElement).toHaveTextContent("TLT, GLD");
  });

  it("separates assets without enough history yet", () => {
    render(<AllocationBars data={data} focus="Max Sharpe" />);
    expect(screen.getByText(/not enough history/i).parentElement).toHaveTextContent("NBIS");
  });
});
