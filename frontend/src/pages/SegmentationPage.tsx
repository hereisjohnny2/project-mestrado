import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  ImageSummary,
  ModelSummary,
  SegmentationResult,
  SegmentationRun,
  getProject,
  getRunResults,
  imageFileUrl,
  listModels,
  listRuns,
  resultBinUrl,
  resultOverlayUrl,
  runExportUrl,
  runStreamUrl,
  startSegmentationRun,
  uploadImages,
} from "../api/client";
import { useToast } from "../components/ToastProvider";

type SortKey = "filename" | "porosity" | "time_ms";
type ViewMode = "compare" | "overlay";

function CompareSlider({ original, other }: { original: string; other: string }) {
  const [pos, setPos] = useState(50);
  return (
    <div className="space-y-2">
      <div className="relative aspect-[4/3] w-full overflow-hidden rounded-lg border border-zinc-800 bg-black">
        <img src={other} alt="segmentada" className="absolute inset-0 h-full w-full object-contain" />
        <div className="absolute inset-0 overflow-hidden" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
          <img src={original} alt="original" className="h-full w-full object-contain" />
        </div>
        <div className="absolute inset-y-0 w-px bg-blue-400" style={{ left: `${pos}%` }} />
      </div>
      <input
        type="range"
        min={0}
        max={100}
        value={pos}
        onChange={(e) => setPos(Number(e.target.value))}
        className="w-full"
        aria-label="Comparar original e segmentada"
      />
    </div>
  );
}

export default function SegmentationPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const { showError, showSuccess } = useToast();

  const [images, setImages] = useState<ImageSummary[]>([]);
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [runs, setRuns] = useState<SegmentationRun[]>([]);
  const [modelId, setModelId] = useState<string>("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [uploading, setUploading] = useState(false);
  const [starting, setStarting] = useState(false);

  const [activeRun, setActiveRun] = useState<SegmentationRun | null>(null);
  const [results, setResults] = useState<SegmentationResult[]>([]);
  const [sortKey, setSortKey] = useState<SortKey>("filename");
  const [sortDesc, setSortDesc] = useState(false);
  const [focusId, setFocusId] = useState<string | null>(null);
  const [view, setView] = useState<ViewMode>("compare");
  const eventSourceRef = useRef<EventSource | null>(null);

  const refreshImages = () => {
    if (!projectId) return;
    getProject(projectId)
      .then((p) => setImages(p.images))
      .catch((e) => showError(e, "Não foi possível carregar as imagens."));
  };

  useEffect(() => {
    if (!projectId) return;
    refreshImages();
    listModels(projectId)
      .then((list) => {
        setModels(list);
        setModelId((cur) => cur || list[0]?.id || "");
      })
      .catch((e) => showError(e, "Não foi possível carregar os modelos."));
    listRuns(projectId)
      .then(setRuns)
      .catch((e) => showError(e, "Não foi possível carregar o histórico."));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  useEffect(() => () => eventSourceRef.current?.close(), []);

  const model = useMemo(() => models.find((m) => m.id === modelId) ?? null, [models, modelId]);
  const imageById = useMemo(() => new Map(images.map((i) => [i.id, i])), [images]);

  const sorted = useMemo(() => {
    const list = [...results];
    list.sort((a, b) => {
      const cmp = sortKey === "filename" ? a.filename.localeCompare(b.filename) : a[sortKey] - b[sortKey];
      return sortDesc ? -cmp : cmp;
    });
    return list;
  }, [results, sortKey, sortDesc]);

  const focused = results.find((r) => r.id === focusId) ?? sorted[0] ?? null;

  const openRun = (run: SegmentationRun) => {
    eventSourceRef.current?.close();
    setActiveRun(run);
    setResults([]);
    setFocusId(null);
    const load = () =>
      getRunResults(run.id)
        .then(setResults)
        .catch((e) => showError(e, "Não foi possível carregar os resultados."));

    if (run.status === "done" || run.status === "failed") {
      load();
      return;
    }
    const es = new EventSource(runStreamUrl(run.id));
    eventSourceRef.current = es;
    es.onmessage = (ev) => {
      const next: SegmentationRun = JSON.parse(ev.data);
      setActiveRun(next);
      setRuns((prev) => prev.map((r) => (r.id === next.id ? next : r)));
      if (next.done > 0) load();
      if (next.status === "done" || next.status === "failed") {
        es.close();
        if (next.status === "failed") showError(new Error(next.error ?? ""), "A segmentação falhou.");
        else showSuccess("Segmentação concluída.");
      }
    };
    es.onerror = () => es.close();
  };

  const onUpload = async (files: FileList | null) => {
    if (!projectId || !files || files.length === 0) return;
    setUploading(true);
    try {
      const created = await uploadImages(projectId, files);
      setImages((prev) => [...prev, ...created]);
      setSelected((prev) => new Set([...prev, ...created.map((i) => i.id)]));
      showSuccess(`${created.length} imagem(ns) enviada(s).`);
    } catch (e) {
      showError(e, "Não foi possível enviar as imagens.");
    } finally {
      setUploading(false);
    }
  };

  const onRun = async () => {
    if (!projectId || !modelId || selected.size === 0) return;
    setStarting(true);
    try {
      const run = await startSegmentationRun(modelId, [...selected]);
      setRuns((prev) => [run, ...prev]);
      openRun(run);
    } catch (e) {
      showError(e, "Não foi possível iniciar a segmentação.");
    } finally {
      setStarting(false);
    }
  };

  const toggle = (id: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const sortHeader = (key: SortKey, label: string) => (
    <button
      className="hover:text-zinc-100"
      onClick={() => {
        if (sortKey === key) setSortDesc(!sortDesc);
        else {
          setSortKey(key);
          setSortDesc(false);
        }
      }}
    >
      {label} {sortKey === key ? (sortDesc ? "↓" : "↑") : ""}
    </button>
  );

  const pct = activeRun && activeRun.total > 0 ? (activeRun.done / activeRun.total) * 100 : 0;

  return (
    <div className="mx-auto max-w-6xl p-8">
      <Link to={`/projects/${projectId}`} className="text-sm text-zinc-400 hover:text-zinc-200">
        ← Projeto
      </Link>
      <h1 className="mt-2 text-2xl font-semibold text-zinc-100">Segmentação em lote</h1>

      <section className="mt-6 grid gap-6 md:grid-cols-2">
        <div className="rounded-xl border border-zinc-800 p-4">
          <h2 className="text-sm font-medium text-zinc-200">1. Modelo</h2>
          {models.length === 0 ? (
            <p className="mt-2 text-sm text-zinc-400">
              Nenhum modelo publicado.{" "}
              <Link to={`/projects/${projectId}/training`} className="text-blue-400 hover:underline">
                Treine um modelo
              </Link>
              .
            </p>
          ) : (
            <>
              <select
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                className="mt-2 w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100"
              >
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name} v{m.version}
                  </option>
                ))}
              </select>
              {model && (
                <p className="mt-2 text-xs text-zinc-400">
                  Acurácia {(model.metrics.accuracy * 100).toFixed(2)}% · {model.metrics.loss_curve.length} épocas
                </p>
              )}
            </>
          )}
        </div>

        <div className="rounded-xl border border-zinc-800 p-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium text-zinc-200">2. Imagens ({selected.size} selecionadas)</h2>
            <div className="flex gap-2 text-xs">
              <button className="text-zinc-400 hover:text-zinc-200" onClick={() => setSelected(new Set(images.map((i) => i.id)))}>
                Todas
              </button>
              <button className="text-zinc-400 hover:text-zinc-200" onClick={() => setSelected(new Set())}>
                Nenhuma
              </button>
            </div>
          </div>
          <ul className="mt-2 max-h-40 space-y-1 overflow-y-auto text-sm">
            {images.map((img) => (
              <li key={img.id}>
                <label className="flex cursor-pointer items-center gap-2 text-zinc-300">
                  <input type="checkbox" checked={selected.has(img.id)} onChange={() => toggle(img.id)} />
                  {img.filename}
                </label>
              </li>
            ))}
            {images.length === 0 && <li className="text-zinc-500">Nenhuma imagem no projeto.</li>}
          </ul>
          <label className="mt-3 inline-block cursor-pointer rounded-lg border border-zinc-700 px-3 py-1.5 text-sm text-zinc-200 hover:bg-zinc-800">
            {uploading ? "Enviando..." : "Enviar lote"}
            <input type="file" accept="image/*" multiple hidden disabled={uploading} onChange={(e) => onUpload(e.target.files)} />
          </label>
        </div>
      </section>

      <button
        onClick={onRun}
        disabled={starting || !modelId || selected.size === 0}
        className="mt-4 rounded-lg bg-blue-600 px-5 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-40"
      >
        {starting ? "Iniciando..." : "Segmentar"}
      </button>

      {runs.length > 0 && (
        <section className="mt-8">
          <h2 className="text-sm font-medium text-zinc-200">Execuções anteriores</h2>
          <div className="mt-2 flex flex-wrap gap-2">
            {runs.map((r) => (
              <button
                key={r.id}
                onClick={() => openRun(r)}
                className={`rounded-lg border px-3 py-1.5 text-xs ${
                  activeRun?.id === r.id ? "border-blue-500 text-zinc-100" : "border-zinc-700 text-zinc-400 hover:bg-zinc-800"
                }`}
              >
                {new Date(r.created_at + "Z").toLocaleString()} · {r.done}/{r.total} · {r.status}
              </button>
            ))}
          </div>
        </section>
      )}

      {activeRun && (
        <section className="mt-8">
          {(activeRun.status === "pending" || activeRun.status === "running") && (
            <div className="mb-4">
              <div className="h-2 overflow-hidden rounded bg-zinc-800">
                <div className="h-full bg-blue-500 transition-all" style={{ width: `${pct}%` }} />
              </div>
              <p className="mt-1 text-xs text-zinc-400">
                {activeRun.done} de {activeRun.total} imagens
              </p>
            </div>
          )}
          {activeRun.status === "failed" && <p className="mb-4 text-sm text-red-400">Falhou: {activeRun.error}</p>}

          {results.length > 0 && (
            <div className="grid gap-6 lg:grid-cols-2">
              <div>
                <div className="mb-2 flex items-center justify-between">
                  <h2 className="text-sm font-medium text-zinc-200">Porosidade</h2>
                  <div className="flex gap-2 text-xs">
                    <a className="text-blue-400 hover:underline" href={runExportUrl(activeRun.id, "csv")}>
                      CSV
                    </a>
                    <a className="text-blue-400 hover:underline" href={runExportUrl(activeRun.id, "zip")}>
                      ZIP
                    </a>
                  </div>
                </div>
                <table className="w-full text-left text-sm">
                  <thead className="text-xs uppercase tracking-wide text-zinc-500">
                    <tr>
                      <th className="py-1">{sortHeader("filename", "Imagem")}</th>
                      <th>{sortHeader("porosity", "Porosidade")}</th>
                      <th>{sortHeader("time_ms", "Tempo (ms)")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sorted.map((r) => (
                      <tr
                        key={r.id}
                        onClick={() => setFocusId(r.id)}
                        className={`cursor-pointer border-t border-zinc-800 ${
                          focused?.id === r.id ? "bg-zinc-800/60" : "hover:bg-zinc-900"
                        }`}
                      >
                        <td className="py-1.5 text-zinc-200">{r.filename}</td>
                        <td className="text-zinc-300">{(r.porosity * 100).toFixed(2)}%</td>
                        <td className="text-zinc-400">{r.time_ms.toFixed(0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {focused && (
                <div>
                  <div className="mb-2 flex gap-2 text-xs">
                    {(["compare", "overlay"] as ViewMode[]).map((v) => (
                      <button
                        key={v}
                        onClick={() => setView(v)}
                        className={`rounded-lg border px-3 py-1 ${
                          view === v ? "border-blue-500 text-zinc-100" : "border-zinc-700 text-zinc-400"
                        }`}
                      >
                        {v === "compare" ? "Original × binarizada" : "Overlay"}
                      </button>
                    ))}
                  </div>
                  {view === "compare" ? (
                    <CompareSlider original={imageFileUrl(focused.image_id)} other={resultBinUrl(focused.id)} />
                  ) : (
                    <img
                      src={resultOverlayUrl(focused.id)}
                      alt={`overlay de ${imageById.get(focused.image_id)?.filename ?? focused.filename}`}
                      className="w-full rounded-lg border border-zinc-800"
                    />
                  )}
                </div>
              )}
            </div>
          )}
        </section>
      )}
    </div>
  );
}
