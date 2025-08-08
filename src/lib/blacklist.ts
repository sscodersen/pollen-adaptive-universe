export const BLACKLIST_PATTERNS: RegExp[] = [
  /(^|[^a-z0-9])(donald\s+)?trump([^a-z0-9]|$)/i,
  /#trump/i,
];

export function isBlacklistedText(text?: string): boolean {
  if (!text) return false;
  const t = String(text).toLowerCase();
  return BLACKLIST_PATTERNS.some((re) => re.test(t));
}

export function filterBlacklisted<T extends Record<string, any>>(items: T[], fields: (keyof T)[]): T[] {
  return items.filter((item) => {
    for (const f of fields) {
      const v = item[f];
      if (typeof v === 'string' && isBlacklistedText(v)) return false;
      if (Array.isArray(v) && v.some((x) => typeof x === 'string' && isBlacklistedText(x))) return false;
    }
    return true;
  });
}
