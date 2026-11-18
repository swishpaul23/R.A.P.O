import { Info } from "lucide-react";
import { useId, useRef, useState } from "react";

/** Small (i) button that shows a short explanation on hover or keyboard focus. */
export default function InfoTip({ term, children }: { term: string; children: string }) {
  const [open, setOpen] = useState(false);
  const [alignLeft, setAlignLeft] = useState(false);
  const ref = useRef<HTMLButtonElement>(null);
  const id = useId();
  // Open toward whichever side has room, so the tooltip never runs off-screen.
  const show = () => {
    const r = ref.current?.getBoundingClientRect();
    setAlignLeft(!!r && r.right < 260);
    setOpen(true);
  };
  return (
    <span className="relative inline-flex">
      <button
        ref={ref}
        type="button"
        aria-label={`What is ${term}?`}
        aria-describedby={open ? id : undefined}
        onMouseEnter={show}
        onMouseLeave={() => setOpen(false)}
        onFocus={show}
        onBlur={() => setOpen(false)}
        onClick={() => (open ? setOpen(false) : show())}
        onKeyDown={(e) => { if (e.key === "Escape") setOpen(false); }}
        className="-my-3 -mr-3 inline-flex size-11 items-center justify-center rounded-full text-fg-3 hover:text-fg"
      >
        <Info className="size-3.5" aria-hidden="true" />
      </button>
      {open && (
        <span id={id} role="tooltip"
          className={`absolute top-full z-20 mt-1 w-60 rounded-lg ${alignLeft ? "left-0" : "right-0"} border border-line-strong bg-raised/95 px-3 py-2 text-xs font-normal leading-relaxed text-fg-2 shadow-lg backdrop-blur-md`}>
          {children}
        </span>
      )}
    </span>
  );
}
