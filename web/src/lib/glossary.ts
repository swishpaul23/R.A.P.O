export const GLOSSARY: Record<string, string> = {
  CAGR: "Compound annual growth rate: the steady yearly return that would turn the starting value into the ending value over the whole period.",
  "Return in window": "Total gain or loss from the start to the end of this stress window. Not annualized, because annualizing a few months exaggerates it.",
  Sharpe: "Return above the risk-free rate (T-bills) per unit of volatility. Higher is better; above 1 is strong, below 0 means T-bills did better.",
  "Max drawdown": "The largest fall from a peak to a later low. It's the worst loss you would have sat through before recovering.",
  "VaR 95% (1 day)": "Value at Risk: on 95% of days the loss was smaller than this. Roughly one trading day in 20 was worse.",
  "Turnover / yr": "How much of the portfolio is traded per year. 1.0× means the whole portfolio was replaced once; more trading means more costs.",
  Volatility: "How much returns swing around their average, annualized. Higher means a bumpier ride.",
  Sortino: "Like Sharpe, but only downside swings count as risk, so upside volatility isn't penalized.",
  Calmar: "CAGR divided by the size of the max drawdown: return earned per unit of worst-case loss.",
};
