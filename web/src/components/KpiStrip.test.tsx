import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { MetricRow } from "../lib/api";
import KpiStrip from "./KpiStrip";

const row: MetricRow = { name: "HRP", total_return: 0.031, cagr: 0.0762, ann_vol: 0.1, sharpe: 0.65, sortino: 0.9,
  max_drawdown: -0.11, calmar: 0.7, var_95: -0.0092, annual_turnover: 0.5 };
const bench: MetricRow = { ...row, name: "SPY (benchmark)", total_return: -0.04, cagr: -0.08 };

describe("KpiStrip", () => {
  it("shows CAGR for the full period", () => {
    render(<KpiStrip row={row} bench={bench} loading={false} isWindow={false} />);
    expect(screen.getByText("CAGR")).toBeInTheDocument();
    expect(screen.getByText("7.62%")).toBeInTheDocument();
  });

  it("shows total return, not annualized CAGR, for a stress window", () => {
    render(<KpiStrip row={row} bench={bench} loading={false} isWindow />);
    expect(screen.getByText("Return in window")).toBeInTheDocument();
    expect(screen.getByText("3.10%")).toBeInTheDocument();
    expect(screen.queryByText("7.62%")).not.toBeInTheDocument();
  });

  it("explains each metric from an info button", () => {
    render(<KpiStrip row={row} bench={bench} loading={false} isWindow={false} />);
    for (const name of ["CAGR", "Sharpe", "Max drawdown", "VaR 95% (1 day)", "Turnover / yr"]) {
      const btn = screen.getByRole("button", { name: `What is ${name}?` });
      fireEvent.mouseEnter(btn);
      expect(screen.getByRole("tooltip").textContent!.length).toBeGreaterThan(20);
      fireEvent.mouseLeave(btn);
      expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
    }
  });

  it("opens the explanation on keyboard focus and closes on Escape", () => {
    render(<KpiStrip row={row} bench={bench} loading={false} isWindow={false} />);
    const btn = screen.getByRole("button", { name: "What is Sharpe?" });
    fireEvent.focus(btn);
    expect(screen.getByRole("tooltip")).toHaveTextContent(/risk-free/i);
    fireEvent.keyDown(btn, { key: "Escape" });
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });
});
