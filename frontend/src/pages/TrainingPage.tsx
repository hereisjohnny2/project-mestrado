import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  DatasetHistogram,
  DatasetSummary,
  ModelSummary,
  TrainingJob,
  datasetDownloadUrl,
  generateDataset,
  getDatasetHistogram,
  getTrainingJob,
  listDatasets,
  listModels,
  modelDownloadUrl,
  publishModel,
  startTrainingJob,
  trainingJobStreamUrl,
} from "../api/client";
import { useToast } from "../components/ToastProvider";

const DEFAULT_CONFIG = { epochs: 5, learning_rate: 0.0025, batch_size: 16, split_ratio: 0.8, seed: "" };

function Bars({ values, color }: { values: number[]; color: string }) {
  const max = Math.max(1, ...values);
  return (
    <div className="flex h-16 items-end gap-px">
      {values.map((v, i) => (
        <div
          key={i}
          className="flex-1 rounded-t-sm"
          style={{ height: `${(v / max) * 100}%`, backgroundColor: color, minHeight: v > 0 ? 1 : 0 }}
        />
      ))}
    </div>
  );
}

function LossChart({ values, totalEpochs }: { values: number[]; totalEpochs?: number }) {
  const W = 480;
  const H = 160;
  const pad = { l: 44, r: 12, t: 10, b: 24 };
  const n = Math.max(totalEpochs ?? 0, values.length, 2);
  const max = Math.max(...values, 1e-9);
  const min = Math.min(...values, 0);
  const x = (i: number) => pad.l + (i / (n - 1)) * (W - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - (v - min) / (max - min || 1)) * (H - pad.t - pad.b);
  const points = values.map((v, i) => `${x(i)},${y(v)}`).join(" ");
  const ticks = [max, (max + min) / 2, min];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Curva de perda por época">
      {ticks.map((t, i) => (
        <g key={i}>
          <line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} stroke="#3f3f46" strokeDasharray="3 3" />
          <text x={pad.l - 6} y={y(t) + 3} textAnchor="end" fontSize="10" fill="#a1a1aa">
            {t.toFixed(3)}
          </text>
        </g>
      ))}
      {values.length > 1 && <polyline points={points} fill="none" stroke="#3b82f6" strokeWidth="2" strokeLinejoin="round" />}
      {values.map((v, i) => (
        <circle key={i} cx={x(i)} cy={y(v)} r="3" fill="#3b82f6">
          <title>{`Época ${i + 1}: ${v.toFixed(4)}`}</title>
        </circle>
      ))}
      <text x={pad.l} y={H - 6} fontSize="10" fill="#a1a1aa">1</text>
      <text x={W - pad.r} y={H - 6} textAnchor="end" fontSize="10" fill="#a1a1aa">
        época {n}
      </text>
    </svg>
  );
}

function Histogram({ histogram }: { histogram: DatasetHistogram }) {
  return (
    <div className="grid grid-cols-2 gap-6">
      <div>
        <p className="mb-2 text-xs font-medium uppercase tracking-wide text-poro">Poro</p>
        <div className="space-y-2">
          <Bars values={histogram.pore.r} color="#ef4444" />
          <Bars values={histogram.pore.g} color="#22c55e" />
          <Bars values={histogram.pore.b} color="#3b82f6" />
        </div>
      </div>
      <div>
        <p className="mb-2 text-xs font-medium uppercase tracking-wide text-solido">Sólido</p>
        <div className="space-y-2">
          <Bars values={histogram.solid.r} color="#ef4444" />
          <Bars values={histogram.solid.g} color="#22c55e" />
          <Bars values={histogram.solid.b} color="#3b82f6" />
        </div>
      </div>
    </div>
  );
}

export default function TrainingPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const { showError, showSuccess } = useToast();

  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
  const [histogram, setHistogram] = useState<DatasetHistogram | null>(null);
  const [generating, setGenerating] = useState(false);

  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [job, setJob] = useState<TrainingJob | null>(null);
  const [starting, setStarting] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);

  const [models, setModels] = useState<ModelSummary[]>([]);
  const [modelName, setModelName] = useState("modelo");
  const [publishing, setPublishing] = useState(false);

  const selectedDataset = useMemo(
    () => datasets.find((d) => d.id === selectedDatasetId) ?? null,
    [datasets, selectedDatasetId],
  );

  const refreshDatasets = () => {
    if (!projectId) return;
    listDatasets(projectId)
      .then((list) => {
        setDatasets(list);
        setSelectedDatasetId((current) => current ?? list[0]?.id ?? null);
      })
      .catch((e) => showError(e, "Não foi possível carregar os datasets."));
  };

  const refreshModels = () => {
    if (!projectId) return;
    listModels(projectId)
      .then(setModels)
      .catch((e) => showError(e, "Não foi possível carregar os modelos."));
  };

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(refreshDatasets, [projectId]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(refreshModels, [projectId]);

  useEffect(() => {
    if (!selectedDatasetId) {
      setHistogram(null);
      return;
    }
    getDatasetHistogram(selectedDatasetId)
      .then(setHistogram)
      .catch((e) => showError(e, "Não foi possível carregar o histograma."));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDatasetId]);

  useEffect(() => {
    return () => eventSourceRef.current?.close();
  }, []);

  const onGenerateDataset = async () => {
    if (!projectId) return;
    setGenerating(true);
    try {
      const dataset = await generateDataset(projectId);
      showSuccess("Dataset gerado a partir das anotações.");
      setDatasets((prev) => [dataset, ...prev]);
      setSelectedDatasetId(dataset.id);
    } catch (e) {
      showError(e, "Não foi possível gerar o dataset. Anote ao menos uma imagem primeiro.");
    } finally {
      setGenerating(false);
    }
  };

  const jobStorageKey = `training-job:${projectId}`;

  const rememberJob = (id: string | null) => {
    try {
      if (id) localStorage.setItem(jobStorageKey, id);
      else localStorage.removeItem(jobStorageKey);
    } catch {
      // storage unavailable — the job just won't survive a reload.
    }
  };

  // Streams progress for a job that lives on the backend, so it can be
  // (re)attached after a page reload.
  const followJob = (jobId: string) => {
    eventSourceRef.current?.close();
    const es = new EventSource(trainingJobStreamUrl(jobId));
    eventSourceRef.current = es;
    es.onmessage = (evt) => {
      const snapshot: TrainingJob = JSON.parse(evt.data);
      setJob(snapshot);
      if (snapshot.status === "done" || snapshot.status === "failed") {
        es.close();
      }
    };
    es.onerror = () => {
      es.close();
      getTrainingJob(jobId).then(setJob).catch(() => undefined);
    };
  };

  useEffect(() => {
    if (!projectId) return;
    let id: string | null = null;
    try {
      id = localStorage.getItem(jobStorageKey);
    } catch {
      return;
    }
    if (!id) return;
    getTrainingJob(id)
      .then((restored) => {
        setJob(restored);
        if (restored.status === "pending" || restored.status === "running") followJob(restored.id);
      })
      .catch(() => rememberJob(null)); // job gone (e.g. backend restarted)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  const onStartTraining = async () => {
    if (!selectedDatasetId) return;
    setStarting(true);
    setJob(null);
    eventSourceRef.current?.close();
    try {
      const created = await startTrainingJob({
        dataset_id: selectedDatasetId,
        epochs: config.epochs,
        learning_rate: config.learning_rate,
        batch_size: config.batch_size,
        split_ratio: config.split_ratio,
        seed: config.seed === "" ? null : Number(config.seed),
      });
      setJob(created);
      rememberJob(created.id);
      followJob(created.id);
    } catch (e) {
      showError(e, "Não foi possível iniciar o treino.");
    } finally {
      setStarting(false);
    }
  };

  const onPublish = async () => {
    if (!job || job.status !== "done" || !modelName.trim()) return;
    setPublishing(true);
    try {
      await publishModel(job.id, modelName.trim());
      showSuccess("Modelo publicado.");
      refreshModels();
    } catch (e) {
      showError(e, "Não foi possível publicar o modelo.");
    } finally {
      setPublishing(false);
    }
  };

  const applyDissertationDefaults = () => setConfig(DEFAULT_CONFIG);

  if (!projectId) return null;

  return (
    <main className="mx-auto max-w-5xl px-6 py-12">
      <Link to={`/projects/${projectId}`} className="text-sm text-zinc-400 hover:text-zinc-200">
        ← Projeto
      </Link>
      <h1 className="mt-2 text-2xl font-semibold text-zinc-100">Dataset e treino</h1>

      {/* Dataset */}
      <section className="mt-8 rounded-lg border border-zinc-800 p-5">
        <div className="flex items-center justify-between gap-4">
          <h2 className="text-lg font-medium text-zinc-100">Dataset</h2>
          <button
            onClick={onGenerateDataset}
            disabled={generating}
            className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50"
          >
            {generating ? "Gerando..." : "Gerar dataset das anotações"}
          </button>
        </div>

        {datasets.length === 0 && (
          <p className="mt-4 text-sm text-zinc-400">Nenhum dataset ainda. Anote imagens e gere o primeiro.</p>
        )}

        {datasets.length > 0 && (
          <div className="mt-4">
            <select
              value={selectedDatasetId ?? ""}
              onChange={(e) => setSelectedDatasetId(e.target.value)}
              className="rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-sm text-zinc-200"
            >
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {new Date(d.created_at).toLocaleString()} — {d.n_pixels.toLocaleString()} px
                </option>
              ))}
            </select>

            {selectedDataset && (
              <div className="mt-4 flex flex-wrap items-center gap-3 text-sm text-zinc-300">
                <span className="flex items-center gap-1.5">
                  <span className="h-3 w-3 rounded-sm bg-poro" /> Poro: {selectedDataset.n_pore.toLocaleString()}
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-3 w-3 rounded-sm bg-solido" /> Sólido: {selectedDataset.n_solid.toLocaleString()}
                </span>
                <a
                  href={datasetDownloadUrl(selectedDataset.id)}
                  className="ml-auto rounded-lg border border-zinc-700 px-3 py-1.5 text-xs text-zinc-300 hover:bg-zinc-800"
                >
                  Baixar .dat
                </a>
              </div>
            )}

            {histogram && (
              <div className="mt-6">
                <p className="mb-2 text-sm text-zinc-400">
                  Histograma RGB por classe — mostra se as classes são separáveis no espaço de cor.
                </p>
                <Histogram histogram={histogram} />
              </div>
            )}
          </div>
        )}
      </section>

      {/* Training */}
      <section className="mt-8 rounded-lg border border-zinc-800 p-5">
        <h2 className="text-lg font-medium text-zinc-100">Treino</h2>

        <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-5">
          <label className="flex flex-col gap-1 text-xs text-zinc-400">
            Épocas
            <input
              type="number"
              min={1}
              value={config.epochs}
              onChange={(e) => setConfig((c) => ({ ...c, epochs: Number(e.target.value) }))}
              className="rounded-lg border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-200"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-zinc-400">
            Learning rate
            <input
              type="number"
              step={0.0001}
              value={config.learning_rate}
              onChange={(e) => setConfig((c) => ({ ...c, learning_rate: Number(e.target.value) }))}
              className="rounded-lg border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-200"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-zinc-400">
            Batch size
            <input
              type="number"
              min={1}
              value={config.batch_size}
              onChange={(e) => setConfig((c) => ({ ...c, batch_size: Number(e.target.value) }))}
              className="rounded-lg border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-200"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-zinc-400">
            Split treino
            <input
              type="number"
              step={0.05}
              min={0.1}
              max={0.95}
              value={config.split_ratio}
              onChange={(e) => setConfig((c) => ({ ...c, split_ratio: Number(e.target.value) }))}
              className="rounded-lg border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-200"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs text-zinc-400">
            Seed
            <input
              type="number"
              placeholder="aleatória"
              value={config.seed}
              onChange={(e) => setConfig((c) => ({ ...c, seed: e.target.value }))}
              className="rounded-lg border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-200"
            />
          </label>
        </div>

        <div className="mt-4 flex gap-2">
          <button
            onClick={applyDissertationDefaults}
            className="rounded-lg border border-zinc-700 px-3 py-1.5 text-sm text-zinc-300 hover:bg-zinc-800"
          >
            Padrão da dissertação
          </button>
          <button
            onClick={onStartTraining}
            disabled={!selectedDatasetId || starting || job?.status === "running"}
            className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50"
          >
            {starting || job?.status === "running" ? "Treinando..." : "Treinar"}
          </button>
        </div>

        {job && (
          <div className="mt-6">
            {(job.status === "running" || job.status === "pending") && (
              <div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-800">
                  <div
                    className="h-full bg-blue-600 transition-all"
                    style={{ width: `${job.epochs ? (job.epoch / job.epochs) * 100 : 0}%` }}
                  />
                </div>
                <p className="mt-1 text-xs text-zinc-400">
                  Época {job.epoch} de {job.epochs}
                  {job.loss_curve.length > 0 && ` — perda: ${job.loss_curve[job.loss_curve.length - 1].toFixed(4)}`}
                </p>
                <div className="mt-3">
                  <LossChart values={job.loss_curve} totalEpochs={job.epochs} />
                </div>
              </div>
            )}

            {job.status === "failed" && (
              <p className="text-sm text-red-400">Falha no treino: {job.error}</p>
            )}

            {job.status === "done" && job.metrics && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                  <Stat label="Acurácia" value={`${(job.metrics.accuracy * 100).toFixed(1)}%`} />
                  <Stat label="Duração" value={`${Math.round(job.metrics.duration_ms)} ms`} />
                  <Stat label="Precisão poro" value={(job.metrics.per_class.pore.precision * 100).toFixed(1) + "%"} />
                  <Stat label="Recall poro" value={(job.metrics.per_class.pore.recall * 100).toFixed(1) + "%"} />
                </div>

                <div>
                  <p className="mb-1 text-xs text-zinc-400">Curva de perda</p>
                  <LossChart values={job.metrics.loss_curve} />
                </div>

                <table className="w-full text-sm text-zinc-300">
                  <thead>
                    <tr className="text-left text-xs uppercase text-zinc-500">
                      <th className="py-1">Classe</th>
                      <th>Precisão</th>
                      <th>Recall</th>
                      <th>F1</th>
                      <th>IoU</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(job.metrics.per_class).map(([cls, m]) => (
                      <tr key={cls} className="border-t border-zinc-800">
                        <td className="py-1.5 capitalize">{cls === "pore" ? "Poro" : "Sólido"}</td>
                        <td>{(m.precision * 100).toFixed(1)}%</td>
                        <td>{(m.recall * 100).toFixed(1)}%</td>
                        <td>{(m.f1 * 100).toFixed(1)}%</td>
                        <td>{(m.iou * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                <div className="flex items-center gap-2">
                  <input
                    value={modelName}
                    onChange={(e) => setModelName(e.target.value)}
                    placeholder="nome do modelo"
                    className="rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-sm text-zinc-200"
                  />
                  <button
                    onClick={onPublish}
                    disabled={publishing || !modelName.trim()}
                    className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-50"
                  >
                    {publishing ? "Publicando..." : "Publicar modelo"}
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </section>

      {/* Published models */}
      <section className="mt-8 rounded-lg border border-zinc-800 p-5">
        <h2 className="text-lg font-medium text-zinc-100">Modelos publicados</h2>
        {models.length === 0 && <p className="mt-4 text-sm text-zinc-400">Nenhum modelo publicado ainda.</p>}
        <div className="mt-4 space-y-2">
          {models.map((m) => (
            <div key={m.id} className="flex items-center justify-between gap-4 rounded-lg border border-zinc-800 px-4 py-2.5">
              <div>
                <p className="text-sm font-medium text-zinc-100">
                  {m.name} <span className="text-zinc-500">v{m.version}</span>
                </p>
                <p className="text-xs text-zinc-400">
                  acurácia {(m.metrics.accuracy * 100).toFixed(1)}% — {new Date(m.created_at).toLocaleString()}
                </p>
              </div>
              <div className="flex gap-2 text-xs">
                <a href={modelDownloadUrl(m.id, "pt")} className="rounded border border-zinc-700 px-2 py-1 text-zinc-300 hover:bg-zinc-800">
                  .pt
                </a>
                <a href={modelDownloadUrl(m.id, "json")} className="rounded border border-zinc-700 px-2 py-1 text-zinc-300 hover:bg-zinc-800">
                  .json
                </a>
                <a href={modelDownloadUrl(m.id, "scripted")} className="rounded border border-zinc-700 px-2 py-1 text-zinc-300 hover:bg-zinc-800">
                  TorchScript
                </a>
              </div>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-zinc-800 px-3 py-2">
      <p className="text-xs text-zinc-500">{label}</p>
      <p className="text-lg font-medium text-zinc-100">{value}</p>
    </div>
  );
}
