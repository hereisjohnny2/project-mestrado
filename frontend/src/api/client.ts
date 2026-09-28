// Thin fetch wrappers over the backend API (plan §4.4). Every request goes
// through /api, which vite (dev) / nginx (prod) rewrite to the backend root.

export interface ProjectSummary {
  id: string;
  name: string;
  created_at: string;
}

export interface ImageSummary {
  id: string;
  project_id: string;
  filename: string;
  width: number;
  height: number;
  sha256: string;
  created_at: string;
  has_annotation: boolean;
}

export interface ProjectDetail extends ProjectSummary {
  images: ImageSummary[];
}

// Wraps a failed request with a message safe to show directly to the user
// (FastAPI's `detail` field when present, a friendly fallback otherwise),
// while keeping the technical detail around for the console/devtools.
export class ApiError extends Error {
  status: number;
  userMessage: string;

  constructor(status: number, userMessage: string, technicalDetail?: string) {
    super(technicalDetail ?? userMessage);
    this.status = status;
    this.userMessage = userMessage;
  }
}

const FALLBACK_MESSAGE = "Não foi possível completar a ação. Tente novamente.";
const NETWORK_MESSAGE = "Não foi possível conectar ao servidor. Verifique sua conexão.";

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail: string | undefined;
    try {
      const body = await res.json();
      detail = typeof body?.detail === "string" ? body.detail : undefined;
    } catch {
      // response wasn't JSON — keep detail undefined, fall back below.
    }
    throw new ApiError(res.status, detail ?? FALLBACK_MESSAGE, detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

async function request<T>(input: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(input, init);
  } catch (e) {
    throw new ApiError(0, NETWORK_MESSAGE, String(e));
  }
  return asJson<T>(res);
}

export function listProjects(): Promise<ProjectSummary[]> {
  return request("/api/projects");
}

export function createProject(name: string): Promise<ProjectSummary> {
  return request("/api/projects", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export function renameProject(projectId: string, name: string): Promise<ProjectSummary> {
  return request(`/api/projects/${projectId}`, {
    method: "PATCH",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export function deleteProject(projectId: string): Promise<void> {
  return request(`/api/projects/${projectId}`, { method: "DELETE" });
}

export function getProject(projectId: string): Promise<ProjectDetail> {
  return request(`/api/projects/${projectId}`);
}

export function uploadImages(projectId: string, files: FileList | File[]): Promise<ImageSummary[]> {
  const form = new FormData();
  for (const file of Array.from(files)) form.append("files", file);
  return request(`/api/projects/${projectId}/images`, { method: "POST", body: form });
}

export function getImage(imageId: string): Promise<ImageSummary> {
  return request(`/api/images/${imageId}`);
}

export function deleteImage(imageId: string): Promise<void> {
  return request(`/api/images/${imageId}`, { method: "DELETE" });
}

export function imageFileUrl(imageId: string): string {
  return `/api/images/${imageId}/file`;
}

export function imageMaskUrl(imageId: string): string {
  return `/api/images/${imageId}/mask`;
}

export async function fetchMaskBlob(imageId: string): Promise<Blob | null> {
  const res = await fetch(`${imageMaskUrl(imageId)}?t=${Date.now()}`);
  if (res.status === 404) return null;
  if (!res.ok) throw new ApiError(res.status, FALLBACK_MESSAGE);
  return res.blob();
}

export function saveMask(imageId: string, blob: Blob): Promise<ImageSummary> {
  return request(imageMaskUrl(imageId), {
    method: "PUT",
    headers: { "content-type": "image/png" },
    body: blob,
  });
}

export function clearMask(imageId: string): Promise<ImageSummary> {
  return request(imageMaskUrl(imageId), { method: "DELETE" });
}

// --- Fase 2: dataset + training ---------------------------------------

export interface DatasetSummary {
  id: string;
  project_id: string;
  n_pixels: number;
  n_pore: number;
  n_solid: number;
  sha256: string;
  created_at: string;
}

export interface DatasetHistogram {
  bin_edges: number[];
  pore: { r: number[]; g: number[]; b: number[] };
  solid: { r: number[]; g: number[]; b: number[] };
}

export interface TrainingMetrics {
  accuracy: number;
  loss_curve: number[];
  confusion_matrix: number[][];
  per_class: Record<string, { precision: number; recall: number; f1: number; iou: number }>;
  duration_ms: number;
}

export interface TrainingJob {
  id: string;
  dataset_id: string;
  status: "pending" | "running" | "done" | "failed";
  epoch: number;
  epochs: number;
  loss_curve: number[];
  error: string | null;
  model_id: string | null;
  metrics: TrainingMetrics | null;
}

export interface TrainingJobParams {
  dataset_id: string;
  epochs?: number;
  learning_rate?: number;
  batch_size?: number;
  split_ratio?: number;
  seed?: number | null;
}

export interface ModelSummary {
  id: string;
  dataset_id: string;
  name: string;
  version: number;
  metrics: TrainingMetrics;
  config: Record<string, unknown>;
  created_at: string;
}

export function listDatasets(projectId: string): Promise<DatasetSummary[]> {
  return request(`/api/projects/${projectId}/datasets`);
}

export function generateDataset(projectId: string): Promise<DatasetSummary> {
  return request(`/api/projects/${projectId}/datasets`, { method: "POST" });
}

export function importDataset(projectId: string, file: File): Promise<DatasetSummary> {
  const form = new FormData();
  form.append("file", file);
  return request(`/api/projects/${projectId}/datasets/import`, { method: "POST", body: form });
}

export function importModel(projectId: string, file: File, name: string, datasetId: string): Promise<ModelSummary> {
  const form = new FormData();
  form.append("file", file);
  form.append("name", name);
  form.append("dataset_id", datasetId);
  return request(`/api/projects/${projectId}/models/import`, { method: "POST", body: form });
}

export function getDatasetHistogram(datasetId: string): Promise<DatasetHistogram> {
  return request(`/api/datasets/${datasetId}/histogram`);
}

export function datasetDownloadUrl(datasetId: string): string {
  return `/api/datasets/${datasetId}/download`;
}

export function startTrainingJob(params: TrainingJobParams): Promise<TrainingJob> {
  return request("/api/training/jobs", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(params),
  });
}

export function getTrainingJob(jobId: string): Promise<TrainingJob> {
  return request(`/api/training/jobs/${jobId}`);
}

export function trainingJobStreamUrl(jobId: string): string {
  return `/api/training/jobs/${jobId}/stream`;
}

export function publishModel(jobId: string, name: string): Promise<ModelSummary> {
  return request(`/api/training/jobs/${jobId}/publish`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export function listModels(projectId: string): Promise<ModelSummary[]> {
  return request(`/api/projects/${projectId}/models`);
}

export function modelDownloadUrl(modelId: string, fmt: "pt" | "json" | "scripted" = "pt"): string {
  return `/api/models/${modelId}/download?fmt=${fmt}`;
}

// --- Fase 3: batch segmentation ---------------------------------------

export interface SegmentationRun {
  id: string;
  model_id: string;
  status: "pending" | "running" | "done" | "failed";
  total: number;
  done: number;
  error: string | null;
  created_at: string;
  finished_at: string | null;
}

export interface SegmentationResult {
  id: string;
  run_id: string;
  image_id: string;
  filename: string;
  porosity: number;
  time_ms: number;
}

export function startSegmentationRun(modelId: string, imageIds: string[]): Promise<SegmentationRun> {
  return request("/api/segmentation/runs", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model_id: modelId, image_ids: imageIds }),
  });
}

export function listRuns(projectId: string): Promise<SegmentationRun[]> {
  return request(`/api/projects/${projectId}/runs`);
}

export function getRunResults(runId: string): Promise<SegmentationResult[]> {
  return request(`/api/segmentation/runs/${runId}/results`);
}

export function deleteRun(runId: string): Promise<void> {
  return request(`/api/segmentation/runs/${runId}`, { method: "DELETE" });
}

export function runStreamUrl(runId: string): string {
  return `/api/segmentation/runs/${runId}/stream`;
}

export function runExportUrl(runId: string, fmt: "csv" | "zip"): string {
  return `/api/segmentation/runs/${runId}/export?fmt=${fmt}`;
}

export function resultBinUrl(resultId: string): string {
  return `/api/segmentation/results/${resultId}/bin`;
}

export function resultOverlayUrl(resultId: string): string {
  return `/api/segmentation/results/${resultId}/overlay`;
}
