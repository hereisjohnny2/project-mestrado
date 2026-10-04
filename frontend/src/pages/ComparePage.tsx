import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  ModelSummary,
  SegmentationResult,
  SegmentationRun,
  classLabel,
  getRunResults,
  listModels,
  listRuns,
  resultBinUrl,
} from "../api/client";
import { useToast } from "../components/ToastProvider";

const COLORS = { a: "#3b82f6", b: "#f59e0b" };

const pct = (v: number) => `${(v * 100).toFixed(2)}%`;

interface MetricRow {
  label: string;
  a: number | null;
  b: number | null;
  format: (v: number) => string;
  higherIsBetter: boolean | null; // null = neutral (no highlight)
}

function metricRows(a: ModelSummary, b: ModelSummary): MetricRow[] {
  const rows: MetricRow[] = [
    { label: "Acurácia", a: a.metrics.accuracy, b: b.metrics.accuracy, format: pct, higherIsBetter: true },
  ];
  const classKeys = [...new Set([...Object.keys(a.metrics.per_class), ...Object.keys(b.metrics.per_class)])];
  for (const cls of classKeys) {
    for (const [key, label] of [["precision", "precisão"], ["recall", "recall"], ["f1", "F1"], ["iou", "IoU"]] as const) {
      rows.push({
        label: `${classLabel(cls)} — ${label}`,
        a: a.metrics.per_class[cls]?.[key] ?? null,
        b: b.metrics.per_class[cls]?.[key] ?? null,
        format: pct,
        higherIsBetter: true,
      });
    }
  }
  rows.push(
    {
      label: "Perda final",
      a: a.metrics.loss_curve[a.metrics.loss_curve.length - 1] ?? null,
      b: b.metrics.loss_curve[b.metrics.loss_curve.length - 1] ?? null,
      format: (v) => v.toFixed(4),
      higherIsBetter: false,
    },
    { label: "Épocas", a: a.metrics.loss_curve.length || null, b: b.metrics.loss_curve.length || null, format: String, higherIsBetter: null },
  );
  return rows;
}

function MultiLossChart({ a, b }: { a: number[]; b: number[] }) {
  const W = 480;
  const H = 160;
  const pad = { l: 44, r: 12, t: 10, b: 24 };
  const all = [...a, ...b];
  if (all.length === 0) return <p className="text-sm text-zinc-500">Nenhum dos modelos tem curva de perda (modelos importados não têm).</p>;
  const n = Math.max(a.length, b.length, 2);
  const max = Math.max(...all);
  const min = Math.min(...all, 0);
  const x = (i: number) => pad.l + (i / (n - 1)) * (W - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - (v - min) / (max - min || 1)) * (H - pad.t - pad.b);
  const line = (values: number[], color: string) =>
    values.length > 0 && (
      <>
        <polyline points={values.map((v, i) => `${x(i)},${y(v)}`).join(" ")} fill="none" stroke={color} strokeWidth="2" />
        {values.map((v, i) => (
          <circle key={i} cx={x(i)} cy={y(v)} r="3" fill={color}>
            <title>{`Época ${i + 1}: ${v.toFixed(4)}`}</title>
          </circle>
        ))}
      </>
    );
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Curvas de perda dos dois modelos">
      {[max, (max + min) / 2, min].map((t, i) => (
        <g key={i}>
          <line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} stroke="#3f3f46" strokeDasharray="3 3" />
          <text x={pad.l - 6} y={y(t) + 3} textAnchor="end" fontSize="10" fill="#a1a1aa">
            {t.toFixed(3)}
          </text>
        </g>
      ))}
      {line(a, COLORS.a)}
      {line(b, COLORS.b)}
      <text x={W - pad.r} y={H - 6} textAnchor="end" fontSize="10" fill="#a1a1aa">
        época {n}
      </text>
    </svg>
  );
}

function modelLabel(m: ModelSummary) {
  return `${m.name} v${m.version} · ${m.architecture}`;
}

export default function ComparePage() {
  const { projectId } = useParams<{ projectId: string }>();
  const { showError } = useToast();

  const [models, setModels] = useState<ModelSummary[]>([]);
  const [runs, setRuns] = useState<SegmentationRun[]>([]);
  const [aId, setAId] = useState("");
  const [bId, setBId] = useState("");
  const [runAId, setRunAId] = useState("");
  const [runBId, setRunBId] = useState("");
  const [resultsA, setResultsA] = useState<SegmentationResult[]>([]);
  const [resultsB, setResultsB] = useState<SegmentationResult[]>([]);
  const [focusImageId, setFocusImageId] = useState<string | null>(null);

  useEffect(() => {
    if (!projectId) return;
    listModels(projectId)
      .then((list) => {
        setModels(list);
        setAId(list[0]?.id ?? "");
        setBId(list[1]?.id ?? list[0]?.id ?? "");
      })
      .catch((e) => showError(e, "Não foi possível carregar os modelos."));
    listRuns(projectId)
      .then((list) => setRuns(list.filter((r) => r.status === "done")))
      .catch((e) => showError(e, "Não foi possível carregar o histórico de segmentações."));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  const a = models.find((m) => m.id === aId) ?? null;
  const b = models.find((m) => m.id === bId) ?? null;
  const runsA = useMemo(() => runs.filter((r) => r.model_id === aId), [runs, aId]);
  const runsB = useMemo(() => runs.filter((r) => r.model_id === bId), [runs, bId]);

  // Default to each model's most recent finished run (list is newest-first).
  useEffect(() => setRunAId(runsA[0]?.id ?? ""), [runsA]);
  useEffect(() => setRunBId(runsB[0]?.id ?? ""), [runsB]);

  useEffect(() => {
    if (!runAId) return setResultsA([]);
    getRunResults(runAId).then(setResultsA).catch((e) => showError(e, "Não foi possível carregar os resultados."));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runAId]);
  useEffect(() => {
    if (!runBId) return setResultsB([]);
    getRunResults(runBId).then(setResultsB).catch((e) => showError(e, "Não foi possível carregar os resultados."));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runBId]);

  const paired = useMemo(() => {
    const byImageB = new Map(resultsB.map((r) => [r.image_id, r]));
    return resultsA
      .map((ra) => ({ a: ra, b: byImageB.get(ra.image_id) }))
      .filter((p): p is { a: SegmentationResult; b: SegmentationResult } => p.b !== undefined)
      .sort((x, y) => x.a.filename.localeCompare(y.a.filename));
  }, [resultsA, resultsB]);

  const focused = paired.find((p) => p.a.image_id === focusImageId) ?? paired[0] ?? null;

  if (models.length < 2) {
    return (
      <main className="mx-auto max-w-5xl px-6 py-12">
        <Link to={`/projects/${projectId}/training`} className="text-sm text-zinc-400 hover:text-zinc-200">
          ← Dataset e treino
        </Link>
        <h1 className="mt-2 text-2xl font-semibold text-zinc-100">Comparar modelos</h1>
        <p className="mt-6 text-zinc-400">Publique ou importe ao menos dois modelos para compará-los.</p>
      </main>
    );
  }

  const modelSelect = (value: string, onChange: (v: string) => void, color: string, label: string) => (
    <label className="flex items-center gap-2 text-sm text-zinc-300">
      <span className="h-3 w-3 rounded-sm" style={{ background: color }} />
      {label}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-sm text-zinc-200"
      >
        {models.map((m) => (
          <option key={m.id} value={m.id}>
            {modelLabel(m)}
          </option>
        ))}
      </select>
    </label>
  );

  const runSelect = (value: string, onChange: (v: string) => void, list: SegmentationRun[]) => (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      disabled={list.length === 0}
      className="rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-sm text-zinc-200 disabled:opacity-50"
    >
      {list.length === 0 && <option value="">sem segmentações</option>}
      {list.map((r) => (
        <option key={r.id} value={r.id}>
          {new Date(r.created_at).toLocaleString()} — {r.total} imagens
        </option>
      ))}
    </select>
  );

  return (
    <main className="mx-auto max-w-5xl px-6 py-12">
      <Link to={`/projects/${projectId}/training`} className="text-sm text-zinc-400 hover:text-zinc-200">
        ← Dataset e treino
      </Link>
      <h1 className="mt-2 text-2xl font-semibold text-zinc-100">Comparar modelos</h1>

      <div className="mt-6 flex flex-wrap gap-6">
        {modelSelect(aId, setAId, COLORS.a, "Modelo A")}
        {modelSelect(bId, setBId, COLORS.b, "Modelo B")}
      </div>

      {a && b && (
        <>
          <section className="mt-8 rounded-lg border border-zinc-800 p-5">
            <h2 className="text-lg font-medium text-zinc-100">Métricas</h2>
            <p className="mt-1 text-xs text-zinc-500">
              Modelos treinados aqui são avaliados no conjunto de teste do treino; importados, no dataset escolhido na
              importação — os números só são diretamente comparáveis entre modelos avaliados nos mesmos dados.
            </p>
            <table className="mt-4 w-full text-sm">
              <thead>
                <tr className="text-left text-zinc-500">
                  <th className="py-1 font-normal" />
                  <th className="py-1 font-medium" style={{ color: COLORS.a }}>{modelLabel(a)}</th>
                  <th className="py-1 font-medium" style={{ color: COLORS.b }}>{modelLabel(b)}</th>
                </tr>
              </thead>
              <tbody>
                {metricRows(a, b).map((row) => {
                  const win =
                    row.higherIsBetter === null || row.a === null || row.b === null || row.a === row.b
                      ? null
                      : (row.a > row.b) === row.higherIsBetter
                        ? "a"
                        : "b";
                  const cell = (v: number | null, side: "a" | "b") => (
                    <td className={`py-1 ${win === side ? "font-medium text-emerald-300" : "text-zinc-300"}`}>
                      {v === null ? "—" : row.format(v)}
                    </td>
                  );
                  return (
                    <tr key={row.label} className="border-t border-zinc-800">
                      <td className="py-1 text-zinc-400">{row.label}</td>
                      {cell(row.a, "a")}
                      {cell(row.b, "b")}
                    </tr>
                  );
                })}
              </tbody>
            </table>

            <h3 className="mt-6 mb-2 text-sm text-zinc-400">Curva de perda</h3>
            <MultiLossChart a={a.metrics.loss_curve} b={b.metrics.loss_curve} />
          </section>

          <section className="mt-8 rounded-lg border border-zinc-800 p-5">
            <h2 className="text-lg font-medium text-zinc-100">Segmentação lado a lado</h2>
            <p className="mt-1 text-xs text-zinc-500">
              Escolha uma segmentação de cada modelo; as imagens em comum são comparadas.{" "}
              <Link to={`/projects/${projectId}/segmentation`} className="text-blue-400 hover:underline">
                Rodar novas segmentações
              </Link>
            </p>
            <div className="mt-4 flex flex-wrap gap-4">
              {runSelect(runAId, setRunAId, runsA)}
              {runSelect(runBId, setRunBId, runsB)}
            </div>

            {paired.length === 0 ? (
              <p className="mt-4 text-sm text-zinc-500">
                Nenhuma imagem em comum — rode os dois modelos sobre as mesmas imagens.
              </p>
            ) : (
              <>
                <table className="mt-4 w-full text-sm">
                  <thead>
                    <tr className="text-left text-zinc-500">
                      <th className="py-1 font-normal">Imagem</th>
                      <th className="py-1 font-medium" style={{ color: COLORS.a }}>Porosidade A</th>
                      <th className="py-1 font-medium" style={{ color: COLORS.b }}>Porosidade B</th>
                      <th className="py-1 font-normal">Δ (B − A)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {paired.map((p) => (
                      <tr
                        key={p.a.image_id}
                        onClick={() => setFocusImageId(p.a.image_id)}
                        className={`cursor-pointer border-t border-zinc-800 hover:bg-zinc-900 ${
                          focused?.a.image_id === p.a.image_id ? "bg-zinc-900" : ""
                        }`}
                      >
                        <td className="py-1 text-zinc-300">{p.a.filename}</td>
                        <td className="py-1 text-zinc-300">{pct(p.a.porosity)}</td>
                        <td className="py-1 text-zinc-300">{pct(p.b.porosity)}</td>
                        <td className="py-1 text-zinc-400">
                          {(p.b.porosity - p.a.porosity >= 0 ? "+" : "") + ((p.b.porosity - p.a.porosity) * 100).toFixed(2)} p.p.
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                {focused && (
                  <div className="mt-6 grid grid-cols-2 gap-4">
                    {([["A", a, focused.a, COLORS.a], ["B", b, focused.b, COLORS.b]] as const).map(([tag, m, r, color]) => (
                      <figure key={tag}>
                        <img
                          src={resultBinUrl(r.id)}
                          alt={`${focused.a.filename} segmentada por ${modelLabel(m)}`}
                          className="aspect-[4/3] w-full rounded-lg border border-zinc-800 bg-black object-contain"
                        />
                        <figcaption className="mt-1 text-xs" style={{ color }}>
                          {tag} — {modelLabel(m)} · {pct(r.porosity)}
                        </figcaption>
                      </figure>
                    ))}
                  </div>
                )}
              </>
            )}
          </section>
        </>
      )}
    </main>
  );
}
