import { describe, expect, it } from "vitest";
import { fmtPct, fmtNum, toneOf } from "./format";

describe("fmtPct", () => {
  it("formats fractions as signed percentages", () => {
    expect(fmtPct(0.0984)).toBe("9.84%");
    expect(fmtPct(-0.183, 1)).toBe("−18.3%");
  });
  it("renders missing values as an em dash", () => {
    expect(fmtPct(null)).toBe("—");
  });
});

describe("fmtNum", () => {
  it("rounds to the given precision", () => {
    expect(fmtNum(0.91234)).toBe("0.91");
    expect(fmtNum(null)).toBe("—");
  });
});

describe("toneOf", () => {
  it("is positive, negative or neutral", () => {
    expect(toneOf(0.01)).toBe("pos");
    expect(toneOf(-0.01)).toBe("neg");
    expect(toneOf(0)).toBe("neutral");
    expect(toneOf(null)).toBe("neutral");
  });
});

describe("negative zero", () => {
  it("drops the minus sign when the rounded value is zero", () => {
    expect(fmtNum(-0.001)).toBe("0.00");
    expect(fmtPct(-0.00001)).toBe("0.00%");
  });
});
