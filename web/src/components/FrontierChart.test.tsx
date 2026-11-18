import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { FIXTURE } from "../lib/fixture";
import { fmtNum, fmtPct } from "../lib/format";
import FrontierChart, { nearest } from "./FrontierChart";

const hrp = FIXTURE.frontier!.strategies.find((s) => s.name === "HRP")!;

describe("FrontierChart strategy values", () => {
  it("shows return, volatility and Sharpe when a strategy is hovered in the legend", () => {
    render(<FrontierChart data={FIXTURE} focus="Equal weight" />);
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
    fireEvent.mouseEnter(screen.getByRole("button", { name: /^HRP:/ }));
    const tip = screen.getByRole("tooltip");
    expect(tip).toHaveTextContent("HRP");
    expect(tip).toHaveTextContent(fmtPct(hrp.ret));
    expect(tip).toHaveTextContent(fmtPct(hrp.vol));
    expect(tip).toHaveTextContent(fmtNum(hrp.sharpe));
  });

  it("hides the tooltip when the pointer leaves", () => {
    render(<FrontierChart data={FIXTURE} focus="HRP" />);
    const item = screen.getByRole("button", { name: /^Max Sharpe:/ });
    fireEvent.mouseEnter(item);
    fireEvent.mouseLeave(item);
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("shows the tooltip on keyboard focus", () => {
    render(<FrontierChart data={FIXTURE} focus="HRP" />);
    fireEvent.focus(screen.getByRole("button", { name: /^Equal weight:/ }));
    expect(screen.getByRole("tooltip")).toHaveTextContent("Equal weight");
  });
});

describe("nearest", () => {
  const pts = [{ name: "A", x: 10, y: 10 }, { name: "B", x: 14, y: 10 }, { name: "C", x: 100, y: 100 }];
  it("picks the closest marker within the radius, even when markers overlap", () => {
    expect(nearest(pts, 13, 10, 20)).toBe("B");
    expect(nearest(pts, 11, 10, 20)).toBe("A");
  });
  it("returns null when nothing is within the radius", () => {
    expect(nearest(pts, 50, 50, 20)).toBeNull();
  });
});
