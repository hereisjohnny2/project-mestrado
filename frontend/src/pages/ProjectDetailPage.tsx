import { ChangeEvent, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  ApiError,
  clearMask,
  deleteImage,
  deleteProject,
  getProject,
  imageFileUrl,
  ProjectDetail,
  renameProject,
  uploadImages,
} from "../api/client";
import { useToast } from "../components/ToastProvider";
import { useConfirm } from "../components/ConfirmProvider";
import Kbd from "../components/Kbd";

function isTypingTarget(el: EventTarget | null): boolean {
  const tag = (el as HTMLElement | null)?.tagName;
  return tag === "INPUT" || tag === "TEXTAREA";
}

type LoadState = "loading" | "ready" | "not-found" | "error";

export default function ProjectDetailPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [uploading, setUploading] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [nameDraft, setNameDraft] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const { showError, showSuccess } = useToast();
  const confirm = useConfirm();

  const refresh = () => {
    if (!projectId) return;
    setLoadState((s) => (s === "ready" ? s : "loading"));
    getProject(projectId)
      .then((p) => {
        setProject(p);
        setLoadState("ready");
      })
      .catch((e) => {
        if (e instanceof ApiError && e.status === 404) {
          setLoadState("not-found");
        } else {
          setLoadState("error");
          showError(e, "Não foi possível carregar o projeto.");
        }
      });
  };

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(refresh, [projectId]);

  useEffect(() => {
    if (renaming) inputRef.current?.select();
  }, [renaming]);

  const onFilesSelected = async (e: ChangeEvent<HTMLInputElement>) => {
    if (!projectId || !e.target.files || e.target.files.length === 0) return;
    setUploading(true);
    try {
      await uploadImages(projectId, e.target.files);
      refresh();
    } catch (e) {
      showError(e, "Não foi possível enviar as imagens.");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const startRename = () => {
    if (!project) return;
    setNameDraft(project.name);
    setRenaming(true);
  };

  const submitRename = async () => {
    if (!project || !nameDraft.trim() || nameDraft.trim() === project.name) {
      setRenaming(false);
      return;
    }
    try {
      const updated = await renameProject(project.id, nameDraft.trim());
      setProject((p) => (p ? { ...p, name: updated.name } : p));
      showSuccess("Projeto renomeado.");
    } catch (e) {
      showError(e, "Não foi possível renomear o projeto.");
    } finally {
      setRenaming(false);
    }
  };

  const onDeleteProject = async () => {
    if (!project) return;
    const ok = await confirm({
      title: `Excluir "${project.name}"?`,
      description: "Todas as imagens e anotações deste projeto serão apagadas permanentemente.",
      confirmLabel: "Excluir",
      danger: true,
    });
    if (!ok) return;
    try {
      await deleteProject(project.id);
      showSuccess("Projeto excluído.");
      navigate("/");
    } catch (e) {
      showError(e, "Não foi possível excluir o projeto.");
    }
  };

  const onRemoveImage = async (imageId: string, filename: string) => {
    const ok = await confirm({
      title: `Remover "${filename}"?`,
      description: "A imagem e sua anotação serão apagadas permanentemente.",
      confirmLabel: "Remover",
      danger: true,
    });
    if (!ok) return;
    try {
      await deleteImage(imageId);
      refresh();
    } catch (e) {
      showError(e, "Não foi possível remover a imagem.");
    }
  };

  const onClearAnnotation = async (imageId: string, filename: string) => {
    const ok = await confirm({
      title: `Limpar anotação de "${filename}"?`,
      description: "A máscara pintada nesta imagem será apagada.",
      confirmLabel: "Limpar",
      danger: true,
    });
    if (!ok) return;
    try {
      await clearMask(imageId);
      refresh();
    } catch (e) {
      showError(e, "Não foi possível limpar a anotação.");
    }
  };

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (isTypingTarget(e.target)) {
        if (e.key === "Enter" && renaming) submitRename();
        if (e.key === "Escape" && renaming) setRenaming(false);
        return;
      }
      if (e.key.toLowerCase() === "u") {
        e.preventDefault();
        fileInputRef.current?.click();
      } else if (e.key.toLowerCase() === "r") {
        e.preventDefault();
        startRename();
      } else if (e.key === "Delete" || e.key === "Backspace") {
        e.preventDefault();
        onDeleteProject();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project, renaming, nameDraft]);

  if (loadState === "not-found") {
    return (
      <main className="mx-auto flex max-w-5xl flex-col items-center px-6 py-24 text-center">
        <h1 className="text-xl font-semibold text-zinc-100">Projeto não encontrado</h1>
        <p className="mt-2 text-zinc-400">
          Esse projeto não existe ou foi excluído.
        </p>
        <Link
          to="/"
          className="mt-6 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500"
        >
          Voltar para o início
        </Link>
      </main>
    );
  }

  if (loadState === "error") {
    return (
      <main className="mx-auto flex max-w-5xl flex-col items-center px-6 py-24 text-center">
        <h1 className="text-xl font-semibold text-zinc-100">Não foi possível carregar o projeto</h1>
        <p className="mt-2 text-zinc-400">Verifique sua conexão e tente novamente.</p>
        <div className="mt-6 flex gap-2">
          <button
            onClick={refresh}
            className="rounded-lg border border-zinc-700 px-4 py-2 text-sm text-zinc-200 hover:bg-zinc-800"
          >
            Tentar novamente
          </button>
          <Link
            to="/"
            className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500"
          >
            Voltar para o início
          </Link>
        </div>
      </main>
    );
  }

  if (!project) {
    return (
      <main className="mx-auto max-w-5xl px-6 py-12">
        <p className="text-zinc-400">Carregando...</p>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-5xl px-6 py-12">
      <Link to="/" className="text-sm text-zinc-400 hover:text-zinc-200">
        ← Projetos
      </Link>

      <div className="mt-2 flex items-center justify-between gap-4">
        {renaming ? (
          <input
            ref={inputRef}
            value={nameDraft}
            onChange={(e) => setNameDraft(e.target.value)}
            onBlur={submitRename}
            className="rounded-lg border border-blue-500 bg-zinc-900 px-3 py-1.5 text-2xl font-semibold text-zinc-100 focus:outline-none"
          />
        ) : (
          <h1 className="text-2xl font-semibold text-zinc-100">{project.name}</h1>
        )}
        <div className="flex gap-2">
          <button
            onClick={startRename}
            className="rounded-lg border border-zinc-700 px-3 py-1.5 text-sm text-zinc-300 hover:bg-zinc-800"
          >
            Renomear <Kbd>R</Kbd>
          </button>
          <button
            onClick={onDeleteProject}
            className="rounded-lg border border-red-900 px-3 py-1.5 text-sm text-red-400 hover:bg-red-950"
          >
            Excluir projeto <Kbd>Del</Kbd>
          </button>
        </div>
      </div>

      <label className="mt-6 inline-block cursor-pointer rounded-lg border border-zinc-700 px-4 py-2 text-sm text-zinc-200 hover:bg-zinc-800">
        {uploading ? "Enviando..." : "Enviar imagens"} <Kbd>U</Kbd>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          onChange={onFilesSelected}
          disabled={uploading}
          hidden
        />
      </label>

      {project.images.length === 0 && <p className="mt-6 text-zinc-400">Nenhuma imagem ainda.</p>}
      <div className="mt-6 grid grid-cols-[repeat(auto-fill,minmax(200px,1fr))] gap-4">
        {project.images.map((img) => (
          <div key={img.id} className="group overflow-hidden rounded-lg border border-zinc-800">
            <Link to={`/images/${img.id}`} className="block">
              <img
                src={imageFileUrl(img.id)}
                alt={img.filename}
                loading="lazy"
                className="h-36 w-full bg-zinc-900 object-cover"
              />
            </Link>
            <div className="flex items-center justify-between gap-2 px-3 py-2">
              <span className="truncate text-sm text-zinc-300" title={img.filename}>
                {img.filename}
              </span>
              <span
                className={`shrink-0 rounded-full px-2 py-0.5 text-xs ${
                  img.has_annotation ? "bg-emerald-900/50 text-emerald-300" : "bg-zinc-800 text-zinc-400"
                }`}
              >
                {img.has_annotation ? "anotada" : "sem anotação"}
              </span>
            </div>
            <div className="flex gap-1 border-t border-zinc-800 px-2 py-1.5 opacity-0 transition-opacity group-hover:opacity-100">
              <button
                onClick={() => onClearAnnotation(img.id, img.filename)}
                disabled={!img.has_annotation}
                className="flex-1 rounded px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 disabled:opacity-30"
              >
                Limpar anotação
              </button>
              <button
                onClick={() => onRemoveImage(img.id, img.filename)}
                className="flex-1 rounded px-2 py-1 text-xs text-red-400 hover:bg-red-950"
              >
                Remover
              </button>
            </div>
          </div>
        ))}
      </div>
    </main>
  );
}
