import { Play, LoaderCircle } from "lucide-react";
import { useState, type FormEvent } from "react";
import { validateField, validateParams, type FieldErrors, type ParamsInput } from "../lib/validate";

interface Props {
  value: ParamsInput;
  onChange: (v: ParamsInput) => void;
  onSubmit: () => void;
  running: boolean;
}

const FIELDS: { key: keyof ParamsInput; label: string; type: string; hint?: string; className: string; placeholder?: string; inputMode?: "numeric" | "decimal" }[] = [
  { key: "start", label: "Start", type: "date", className: "md:col-span-2" },
  { key: "end", label: "End", type: "date", className: "md:col-span-2" },
  { key: "tickers", label: "Tickers", type: "text", placeholder: "SPY, TLT, GLD", hint: "Comma or space separated", className: "md:col-span-5" },
  { key: "benchmark", label: "Benchmark", type: "text", placeholder: "SPY", className: "md:col-span-3" },
  { key: "lookback", label: "Lookback (days)", type: "text", inputMode: "numeric", className: "md:col-span-2" },
  { key: "rebalance", label: "Rebalance (days)", type: "text", inputMode: "numeric", className: "md:col-span-2" },
  { key: "cost", label: "Cost (bps)", type: "text", inputMode: "decimal", className: "md:col-span-2" },
];

export default function ParamsForm({ value, onChange, onSubmit, running }: Props) {
  const [errors, setErrors] = useState<FieldErrors>({});
  const [touched, setTouched] = useState<Partial<Record<keyof ParamsInput, boolean>>>({});

  function update(key: keyof ParamsInput, v: string) {
    const next = { ...value, [key]: v };
    onChange(next);
    if (errors[key]) setErrors((e) => ({ ...e, [key]: undefined }));  // clear on first corrective keystroke
  }

  function blur(key: keyof ParamsInput) {
    setTouched((t) => ({ ...t, [key]: true }));
    setErrors((e) => ({ ...e, [key]: validateField(key, value) }));
  }

  function submit(e: FormEvent) {
    e.preventDefault();
    const all = validateParams(value);
    setErrors(all);
    const first = Object.keys(all)[0];
    if (first) {
      document.getElementById(`p-${first}`)?.focus();
      return;
    }
    onSubmit();
  }

  return (
    <form onSubmit={submit} noValidate aria-label="Backtest parameters"
      className="grid grid-cols-2 gap-x-3 gap-y-3 md:grid-cols-12 md:items-start">
      {FIELDS.map((f) => {
        const err = touched[f.key] || errors[f.key] ? errors[f.key] : undefined;
        return (
          <div key={f.key} className={`flex flex-col gap-1 ${f.key === "tickers" ? "col-span-2" : ""} ${f.className}`}>
            <label htmlFor={`p-${f.key}`} className="text-xs font-medium text-fg-2">{f.label}</label>
            <input
              id={`p-${f.key}`}
              name={f.key}
              type={f.type}
              inputMode={f.inputMode}
              placeholder={f.placeholder}
              value={value[f.key]}
              onChange={(e) => update(f.key, e.target.value)}
              onBlur={() => blur(f.key)}
              aria-invalid={err ? true : undefined}
              aria-describedby={err ? `e-${f.key}` : f.hint ? `h-${f.key}` : undefined}
              autoComplete="off"
              spellCheck={false}
              className={`num h-11 w-full rounded-lg border bg-canvas px-3 text-sm text-fg placeholder:text-fg-3
                hover:border-line-strong focus:border-accent focus:outline-none ${err ? "border-loss" : "border-line"}`}
            />
            {err ? (
              <span id={`e-${f.key}`} className="text-xs text-loss">{err}</span>
            ) : f.hint ? (
              <span id={`h-${f.key}`} className="text-xs text-fg-3">{f.hint}</span>
            ) : null}
          </div>
        );
      })}
      <div className="col-span-2 flex md:col-span-6 md:justify-end md:pt-[1.375rem]">
        <button
          type="submit"
          aria-busy={running}
          className="inline-flex h-11 w-full items-center justify-center gap-2 rounded-lg bg-accent-fill px-5 text-sm font-medium text-white
            hover:bg-accent-fill-hover md:w-auto"
        >
          {running ? <LoaderCircle className="size-4 animate-spin motion-reduce:animate-none" aria-hidden="true" /> : <Play className="size-4" aria-hidden="true" />}
          {running ? "Running backtest…" : "Run backtest"}
        </button>
      </div>
    </form>
  );
}
