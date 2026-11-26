import type { ComponentType } from 'react';
import type { Question } from '../components/Ask';

/** What a section file provides once loaded. */
export interface SectionModule {
  Component: ComponentType;
  /** Every question the section can ask, by id: the lesson uses them and review recycles them. */
  questions: Record<string, Question>;
}

export interface Section {
  /** Stable id used in URLs and in the progress file, e.g. "5.3". */
  id: string;
  /** Shown in the rail: the book's section number, or blank for our own pages. */
  num: string;
  title: string;
  /** Vietnamese section title, when translated. */
  titleVi?: string;
  /** One sentence: what you will be able to do after it. */
  lede: string;
  /** Vietnamese lede, when translated. */
  ledeVi?: string;
  /** Book pages this section draws on, or null when it is ours. */
  pages: string | null;
  /**
   * Loads the section's code on demand. Sections are separate modules so that a
   * page only pulls in (and only reloads for) the section being read.
   */
  load: () => Promise<SectionModule>;
  /** Loads the Vietnamese translation of the section's code on demand, when available. */
  loadVi?: () => Promise<SectionModule>;
}

export interface Chapter {
  /** "ch05", "appA", "x1": the directory name under src/chapters and the URL prefix. */
  id: string;
  num: string;
  title: string;
  pages: string;
  /** Empty while the chapter is still planned. */
  sections: Section[];
}

export type Tier = 'Deep' | 'Standard' | 'Condensed' | 'Skipped';
export interface PlanRow {
  /** Chapter id, or null for a chapter that has no pages in the app. */
  id: string | null;
  num: string;
  title: string;
  pages: string;
  tier: Tier;
  why: string;
  milestone?: string;
  ours?: boolean;
}
