export function stripUrls(text: string): string {
  if (!text) return '';
  // Remove URLs and trailing links
  return text.replace(/https?:\/\/\S+/g, '').replace(/\s+/g, ' ').trim();
}

export function normalizeWhitespace(text: string): string {
  return text.replace(/\s+/g, ' ').replace(/\s*([.,!?;:])\s*/g, '$1 ').trim();
}

export function ensureSentenceEnding(text: string): string {
  if (!text) return '';
  const end = text.trim().slice(-1);
  return ['.', '!', '?'].includes(end) ? text.trim() : text.trim() + '.';
}

export function cleanText(text: string): string {
  let t = text || '';
  t = stripUrls(t);
  t = normalizeWhitespace(t);
  // Collapse repeated punctuation
  t = t.replace(/([.!?]){2,}/g, '$1');
  // Remove excessive hashtags at start
  t = t.replace(/^(#[\w-]+\s*){3,}/, '');
  return ensureSentenceEnding(t);
}

export function truncateText(text: string, maxLen: number): string {
  const t = (text || '').trim();
  if (t.length <= maxLen) return t;
  const cut = t.slice(0, maxLen);
  const lastSpace = cut.lastIndexOf(' ');
  const safe = lastSpace > 40 ? cut.slice(0, lastSpace) : cut;
  return safe.replace(/[.,;:!?]*$/, '') + '…';
}

export function normalizeTags(tags: string[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const tag of tags || []) {
    const key = tag.trim().toLowerCase();
    if (!key || seen.has(key)) continue;
    seen.add(key);
    // Return original casing for first occurrence
    result.push(tag.replace(/^#/, ''));
  }
  return result.slice(0, 8);
}
