import type { ReactNode } from "react";

interface Props {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}

export default function Panel({ title, subtitle, actions, children, className = "" }: Props) {
  return (
    <section
      aria-label={title}
      className={`min-w-0 rounded-xl border border-line bg-surface shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] ${className}`}
    >
      <header className="flex flex-wrap items-center gap-3 border-b border-line px-4 py-3">
        <div className="min-w-0">
          <h2 className="text-sm font-medium text-fg">{title}</h2>
          {subtitle && <p className="text-xs text-fg-3">{subtitle}</p>}
        </div>
        {actions && <div className="ml-auto flex items-center gap-2">{actions}</div>}
      </header>
      <div className="p-4">{children}</div>
    </section>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div aria-hidden="true" className={`skeleton ${className}`} />;
}

export function EmptyHint({ children }: { children: ReactNode }) {
  return <p className="text-sm text-fg-3">{children}</p>;
}

interface SegProps<T extends string> {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}

export function Segmented<T extends string>({ label, value, options, onChange }: SegProps<T>) {
  return (
    <div role="group" aria-label={label} className="flex flex-wrap rounded-lg border border-line bg-canvas p-1">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          aria-pressed={o.value === value}
          onClick={() => onChange(o.value)}
          className={`min-h-11 rounded-md px-3 text-sm ${
            o.value === value ? "bg-raised text-fg shadow-[inset_0_0_0_1px_var(--color-line-strong)]" : "text-fg-2 hover:text-fg"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
