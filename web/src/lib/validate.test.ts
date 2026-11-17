import { describe, expect, it } from "vitest";
import { parseTickers, validateParams } from "./validate";

const ok = { tickers: "SPY, TLT", benchmark: "SPY", start: "2015-01-01", end: "2025-01-01",
  lookback: "252", rebalance: "63", cost: "10" };

describe("parseTickers", () => {
  it("splits on commas and spaces, uppercases and dedupes", () => {
    expect(parseTickers(" spy,tlt  gld,SPY ")).toEqual(["SPY", "TLT", "GLD"]);
  });
});

describe("validateParams", () => {
  it("accepts valid input", () => {
    expect(validateParams(ok)).toEqual({});
  });
  it("flags empty tickers", () => {
    expect(validateParams({ ...ok, tickers: " , " }).tickers).toBe("Add at least one ticker.");
  });
  it("flags a start date after the end date", () => {
    expect(validateParams({ ...ok, start: "2025-02-01" }).start).toBe("Start must be before end.");
  });
  it("flags out-of-range lookback", () => {
    expect(validateParams({ ...ok, lookback: "5" }).lookback).toBe("Use 20 to 1260 days.");
  });
  it("flags a negative cost", () => {
    expect(validateParams({ ...ok, cost: "-1" }).cost).toBe("Use 0 to 200 bps.");
  });
});
