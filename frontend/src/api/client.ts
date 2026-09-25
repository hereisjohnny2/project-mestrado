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
