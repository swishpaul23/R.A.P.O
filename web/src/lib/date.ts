/** YYYY-MM-DD for the user's local calendar day (toISOString would give the UTC day). */
export function localISODate(d: Date = new Date()): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}
