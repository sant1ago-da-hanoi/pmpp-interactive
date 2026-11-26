import { useEffect, useRef, useState, type ReactNode } from 'react';
import { useI18n } from '../lib/i18n';

// A narrated animation: a sequence of beats, each a narration clip plus a drawing
// that is a pure function of how far through the beat we are. Picture and sound
// stay in sync because a beat lasts exactly as long as its clip.
//
// Scripts, audio and the manifest live in content/<chapter>/narration/ (see
// tools/narration.mjs). Captions are the scripts themselves and are an option the
// viewer switches on (off by default, remembered in this browser). With no audio
// for a beat the animation still runs, on a duration estimated from the script
// length, and that beat's caption is shown regardless so nothing is lost.
export interface Beat {
  /** Clip id: content/<chapter>/narration/<clip>.txt and .mp3 */
  clip: string;
  /** Short name shown on the beat's button. */
  label: string;
  /** Draws the beat; p runs from 0 to 1. */
  draw: (p: number) => ReactNode;
}

interface Manifest {
  model: string;
  voice: string;
  clips: Record<string, { file: string; sha256: string; duration: number | null }>;
}

const stripTags = (s: string) => s.replace(/\[[^\]]*\]\s*/g, '').trim();
const estimate = (script: string) => Math.max(4, stripTags(script).split(/\s+/).length / 2.6 + 1);

export function Scene({
  chapter,
  title,
  beats,
  narrow = false,
}: {
  chapter: string;
  title: string;
  beats: Beat[];
  /**
   * The drawing is laid out for the current (narrow) screen, so the player must not give it a
   * minimum width and a sideways scroll below 640 px. The same as data-layout="narrow" on the
   * beat's svg; use this when the drawing is not an svg or when it switches layout with useNarrow().
   */
  narrow?: boolean;
}) {
  const { t: i18n } = useI18n();
  const base = `/content/${chapter}/narration/`;
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [scripts, setScripts] = useState<Record<string, string>>({});
  const [i, setI] = useState(0);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [sound, setSound] = useState(true);
  const [captions, setCaptions] = useState<boolean>(() => {
    try {
      return localStorage.getItem('captions') === 'on';
    } catch {
      return false;
    }
  });
  const toggleCaptions = (on: boolean) => {
    setCaptions(on);
    try {
      localStorage.setItem('captions', on ? 'on' : 'off');
    } catch {
      /* private window: the choice just does not persist */
    }
  };
  const audio = useRef<HTMLAudioElement | null>(null);
  const last = useRef<number | null>(null);
  // Bumped when a clip finishes, so the end of the audio itself moves things on
  // even when the clock has already reached the end of the beat and stopped changing.
  const [ended, setEnded] = useState(0);
  const endedBeat = useRef(-1); // index of the beat whose clip has finished
  const live = useRef(false); // the audio element is playing the current beat's clip
  const [broken, setBroken] = useState<Record<string, boolean>>({}); // clips that failed to load

  useEffect(() => {
    fetch(base + 'manifest.json')
      .then((r) => (r.ok ? r.json() : null))
      .then(setManifest)
      .catch(() => setManifest(null));
    for (const b of beats)
      fetch(base + b.clip + '.txt')
        .then((r) => (r.ok ? r.text() : ''))
        .then((text) => setScripts((s) => ({ ...s, [b.clip]: text })))
        .catch(() => undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chapter]);

  const beat = beats[i];
  const entry = manifest?.clips[beat.clip];
  const script = scripts[beat.clip] ?? '';
  const dur = entry?.duration ?? estimate(script);
  const hasAudio = !!entry && !broken[beat.clip];
  // With sound on and a clip available, the audio element is the clock.
  const audioLed = hasAudio && sound;

  // Start or stop this beat's audio when playback state or the beat changes.
  useEffect(() => {
    const a = audio.current;
    if (!a) return;
    if (playing && audioLed) {
      if (!a.src.endsWith(base + entry!.file)) {
        live.current = false; // until the new clip reports that it is playing
        a.src = base + entry!.file;
      }
      a.currentTime = Math.min(t, dur);
      a.play().catch(() => undefined);
    } else {
      live.current = false;
      a.pause();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, i, audioLed]);

  // The clock. An audio element that has just been given a new clip can still
  // report the old clip's position for a moment, so its time is trusted only
  // once it has said "playing" for this clip.
  useEffect(() => {
    if (!playing) {
      last.current = null;
      return;
    }
    let raf = 0;
    const tick = (now: number) => {
      const a = audio.current;
      const dt = last.current == null ? 0 : (now - last.current) / 1000;
      last.current = now;
      setT((prev) => Math.min(audioLed ? (a && live.current ? a.currentTime : prev) : prev + dt, dur));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, audioLed, dur]);

  // Advance at the end of a beat: when this beat's clip ends, or, without audio,
  // when the clock runs out.
  useEffect(() => {
    if (!playing) return;
    if (audioLed ? endedBeat.current !== i : t < dur) return;
    if (i + 1 < beats.length) {
      setI(i + 1);
      setT(0);
    } else {
      setT(dur);
      setPlaying(false);
    }
  }, [t, dur, playing, i, beats.length, audioLed, ended]);

  const go = (k: number) => {
    endedBeat.current = -1;
    setI(Math.max(0, Math.min(beats.length - 1, k)));
    setT(0);
  };
  const p = dur > 0 ? Math.min(1, t / dur) : 0;
  const finished = !playing && i === beats.length - 1 && t >= dur;

  return (
    <section className="bench wide scene" aria-label={title}>
      <header>
        <h4>{title}</h4>
        <span className="task">
          {i18n.scene.partsSummary(
            beats.length,
            Math.round(beats.reduce((s, b) => s + (manifest?.clips[b.clip]?.duration ?? estimate(scripts[b.clip] ?? '')), 0)),
          )}
        </span>
      </header>
      <div className="body">
        <div className="stage" data-layout={narrow ? 'narrow' : undefined}>
          {beat.draw(p)}
        </div>
        {captions || !hasAudio ? (
          <p className="caption" aria-live="polite">{stripTags(script) || '…'}</p>
        ) : null}
        <div className="row">
          <button className="primary" onClick={() => { if (finished) go(0); setPlaying(!playing); }}>
            {playing ? i18n.scene.pause : finished ? i18n.scene.playAgain : t > 0 || i > 0 ? i18n.scene.resume : i18n.scene.play}
          </button>
          <span className="seg" role="group" aria-label={i18n.scene.partsAria}>
            {beats.map((b, k) => (
              <button key={b.clip} aria-pressed={k === i} onClick={() => go(k)} title={b.label}>
                {k + 1}. {b.label}
              </button>
            ))}
          </span>
          <label className="check">
            <input type="checkbox" checked={sound} onChange={(e) => setSound(e.target.checked)} /> {i18n.scene.sound}
          </label>
          <label className="check">
            <input type="checkbox" checked={captions} onChange={(e) => toggleCaptions(e.target.checked)} /> {i18n.scene.captions}
          </label>
        </div>
        <input
          type="range" min="0" max="1000" value={Math.round(p * 1000)} aria-label={i18n.scene.positionAria}
          style={{ width: '100%', marginTop: '0.6rem' }}
          onChange={(e) => {
            const nt = (Number(e.target.value) / 1000) * dur;
            setT(nt);
            if (audio.current && hasAudio) audio.current.currentTime = nt;
          }}
        />
        <audio
          ref={audio}
          preload="auto"
          onPlaying={() => { live.current = true; }}
          onEnded={(e) => {
            // Ignore an "ended" that belongs to a previous part.
            if (!entry || !e.currentTarget.src.endsWith(base + entry.file)) return;
            live.current = false;
            endedBeat.current = i;
            setEnded((n) => n + 1);
          }}
          onError={() => setBroken((m) => ({ ...m, [beat.clip]: true }))}
        />
      </div>
      <footer>
        {manifest === null
          ? i18n.scene.noManifest
          : hasAudio
            ? i18n.scene.generatedWith(manifest.model)
            : i18n.scene.noAudioYet}
      </footer>
    </section>
  );
}
