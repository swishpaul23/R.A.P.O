import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { FIXTURE } from "../lib/fixture";
import ResultsTable from "./ResultsTable";

const rows = FIXTURE.table.map((r) => ({ ...r, total_return: 0.05 }));

describe("ResultsTable", () => {
  it("uses CAGR for the full period and total return for a stress window", () => {
    const { rerender } = render(<ResultsTable rows={rows} benchmarkLabel={FIXTURE.benchmark_label} focus="HRP" onFocus={() => {}} isWindow={false} />);
    expect(screen.getByRole("button", { name: /^CAGR/ })).toBeInTheDocument();
    rerender(<ResultsTable rows={rows} benchmarkLabel={FIXTURE.benchmark_label} focus="HRP" onFocus={() => {}} isWindow />);
    expect(screen.queryByRole("button", { name: /^CAGR/ })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^Return/ })).toBeInTheDocument();
  });

  it("has an explanation button for each metric column", () => {
    render(<ResultsTable rows={rows} benchmarkLabel={FIXTURE.benchmark_label} focus="HRP" onFocus={() => {}} isWindow={false} />);
    expect(screen.getByRole("button", { name: "What is Max drawdown?" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "What is Sortino?" })).toBeInTheDocument();
  });
});
