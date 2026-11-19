import { render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { REGIME_FIXTURE } from "../../lib/regimeFixture";
import ResearchPage from "./ResearchPage";

vi.mock("./CycleChart", () => ({ default: () => <div data-testid="cycle-chart" /> }));
afterEach(() => vi.restoreAllMocks());

function mockApi(body: unknown, status = 200) {
  vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify(body), { status }));
}

describe("ResearchPage", () => {
  it("shows all four crises with SPY and HRP drawdowns", async () => {
    mockApi(REGIME_FIXTURE);
    render(<ResearchPage today="2026-11-18" />);
    const table = await screen.findByRole("table", { name: /four crises/i });
    for (const c of ["dot-com", "2008", "2020", "2022"]) {
      expect(within(table).getByText(new RegExp(c, "i"))).toBeInTheDocument();
    }
    expect(within(table).getByText("−55.2%")).toBeInTheDocument(); // SPY in 2008
  });

  it("reports whether each overlay difference is significant", async () => {
    mockApi(REGIME_FIXTURE);
    render(<ResearchPage today="2026-11-18" />);
    const table = await screen.findByRole("table", { name: /does timing help/i });
    expect(within(table).getAllByRole("row")).toHaveLength(1 + 8);
    expect(within(table).queryAllByText("Yes")).toHaveLength(0);
  });

  it("shows today's defensive tilt and stress probability", async () => {
    mockApi(REGIME_FIXTURE);
    render(<ResearchPage today="2026-11-18" />);
    const strip = await screen.findByRole("region", { name: /current reading/i });
    expect(strip).toHaveTextContent("10.35%");
    expect(strip).toHaveTextContent("0.0%");
    expect(screen.queryByText(/study data is from/i)).not.toBeInTheDocument();
  });

  it("warns when the study is stale", async () => {
    mockApi(REGIME_FIXTURE);
    render(<ResearchPage today="2027-03-01" />);
    expect(await screen.findByText(/study data is from 2026-11-17/i)).toBeInTheDocument();
  });

  it("shows the API's message when the study is missing", async () => {
    mockApi({ detail: "No regime study yet. Run scripts/run_regime_study.py and commit data/regime/." }, 404);
    render(<ResearchPage today="2026-11-18" />);
    expect(await screen.findByRole("alert")).toHaveTextContent("run_regime_study.py");
  });
});
