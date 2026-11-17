import { describe, expect, it } from "vitest";
import { tableToCsv } from "./csv";

describe("tableToCsv", () => {
  it("writes a header and one line per row, quoting names with commas", () => {
    const csv = tableToCsv([
      { name: "HRP", cagr: 0.1, sharpe: 0.9 },
      { name: "SPY, benchmark", cagr: null, sharpe: 0.6 },
    ]);
    expect(csv.split("\n")).toEqual([
      "name,cagr,sharpe",
      "HRP,0.1,0.9",
      '"SPY, benchmark",,0.6',
    ]);
  });
});
