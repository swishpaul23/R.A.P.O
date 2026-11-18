import { describe, expect, it } from "vitest";
import { localISODate } from "./date";

describe("localISODate", () => {
  it("uses the local calendar date, not UTC", () => {
    // 11:30 pm local on Nov 17 is already Nov 18 in UTC for anyone west of Greenwich.
    expect(localISODate(new Date(2026, 10, 17, 23, 30))).toBe("2026-11-17");
  });
  it("zero-pads month and day", () => {
    expect(localISODate(new Date(2026, 0, 5, 12))).toBe("2026-01-05");
  });
});
