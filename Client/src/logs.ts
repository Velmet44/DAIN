const tag = "%c[dain]";
const okTag = "%c[dain:ok]";
const warnTag = "%c[dain:warning]";
const errTag = "%c[dain:error]";
const dbgTag = "%c[dain:debug]";

const blue = "color:#3fb6ff;font-weight:bold";
const green = "color:#41d18a;font-weight:bold";
const amber = "color:#ffb84d;font-weight:bold";
const red = "color:#ff5c5c;font-weight:bold";
const gray = "color:#8a8a8a;font-weight:bold";

export function logInfo(...args: unknown[]): void {
  console.info(tag, blue, ...args);
}

export function logOk(...args: unknown[]): void {
  console.info(okTag, green, ...args);
}

export function logWarn(...args: unknown[]): void {
  console.warn(warnTag, amber, ...args);
}

export function logError(...args: unknown[]): void {
  console.error(errTag, red, ...args);
}

/** Verbose frame-level noise: only printed in the Vite dev server. */
export function logDebug(...args: unknown[]): void {
  if (import.meta.env.DEV) console.debug(dbgTag, gray, ...args);
}