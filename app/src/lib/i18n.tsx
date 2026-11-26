import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react';
import { en, type Messages } from './i18n/en';
import { vi } from './i18n/vi';

export type Locale = 'en' | 'vi';

const DICTS: Record<Locale, Messages> = { en, vi };

export function getMessages(locale: Locale): Messages {
  return DICTS[locale];
}

function readStoredLocale(): Locale {
  if (typeof window === 'undefined') return 'en';
  try {
    return window.localStorage.getItem('lang') === 'vi' ? 'vi' : 'en';
  } catch {
    return 'en';
  }
}

interface I18nValue {
  locale: Locale;
  t: Messages;
  setLocale: (locale: Locale) => void;
  toggleLocale: () => void;
}

const I18nContext = createContext<I18nValue>({
  locale: 'en',
  t: en,
  setLocale: () => undefined,
  toggleLocale: () => undefined,
});

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(readStoredLocale);

  useEffect(() => {
    document.documentElement.lang = locale;
    try {
      window.localStorage.setItem('lang', locale);
    } catch {
      /* private window: the choice just does not persist */
    }
  }, [locale]);

  const setLocale = useCallback((next: Locale) => setLocaleState(next), []);
  const toggleLocale = useCallback(() => setLocaleState((prev) => (prev === 'en' ? 'vi' : 'en')), []);

  return (
    <I18nContext.Provider value={{ locale, t: DICTS[locale], setLocale, toggleLocale }}>
      {children}
    </I18nContext.Provider>
  );
}

export function useI18n(): I18nValue {
  return useContext(I18nContext);
}
