import { FormEvent, useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { createProject, listProjects, ProjectSummary } from "../api/client";
import { useToast } from "../components/ToastProvider";
import Kbd from "../components/Kbd";

function isTypingTarget(el: EventTarget | null): boolean {
  const tag = (el as HTMLElement | null)?.tagName;
  return tag === "INPUT" || tag === "TEXTAREA";
}

export default function ProjectListPage() {
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [loadError, setLoadError] = useState(false);
  const [name, setName] = useState("");
  const [creating, setCreating] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const { showError } = useToast();

  const refresh = () => {
    setLoadError(false);
    listProjects()
      .then(setProjects)
      .catch((e) => {
        setLoadError(true);
        showError(e, "Não foi possível carregar os projetos.");
      });
  };

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === "n" && !isTypingTarget(e.target)) {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    setCreating(true);
    try {
      const project = await createProject(name.trim());
      navigate(`/projects/${project.id}`);
    } catch (e) {
      showError(e, "Não foi possível criar o projeto.");
    } finally {
      setCreating(false);
    }
  };

  return (
    <main className="mx-auto max-w-2xl px-6 py-12">
      <h1 className="text-2xl font-semibold text-zinc-100">Rock Segmentation</h1>
      <p className="mt-1 text-zinc-400">Projetos de anotação, treino e segmentação poro × sólido.</p>

      <form onSubmit={onSubmit} className="mt-6 flex gap-2">
        <input
          ref={inputRef}
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Nome do projeto"
          disabled={creating}
          className="flex-1 rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-zinc-100 placeholder:text-zinc-500 focus:border-blue-500 focus:outline-none"
        />
        <button
          type="submit"
          disabled={creating || !name.trim()}
          className="rounded-lg bg-blue-600 px-4 py-2 font-medium text-white hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Criar projeto
        </button>
      </form>
      <p className="mt-1.5 text-xs text-zinc-500">
        Atalho: <Kbd>N</Kbd> para focar o campo
      </p>

      <div className="mt-8">
        {projects === null && !loadError && <p className="text-zinc-400">Carregando...</p>}
        {loadError && (
          <div className="flex flex-col items-start gap-3 rounded-lg border border-zinc-800 px-4 py-4 text-zinc-400">
            <p>Não foi possível carregar os projetos.</p>
            <button
              onClick={refresh}
              className="rounded-lg border border-zinc-700 px-3 py-1.5 text-sm text-zinc-200 hover:bg-zinc-800"
            >
              Tentar novamente
            </button>
          </div>
        )}
        {projects?.length === 0 && <p className="text-zinc-400">Nenhum projeto ainda.</p>}
        <ul className="flex flex-col gap-2">
          {projects?.map((p) => (
            <li key={p.id}>
              <Link
                to={`/projects/${p.id}`}
                className="flex items-center justify-between rounded-lg border border-zinc-800 px-4 py-3 hover:border-zinc-600 hover:bg-zinc-900"
              >
                <span className="font-medium text-zinc-100">{p.name}</span>
                <span className="text-xs text-zinc-500">criado em {new Date(p.created_at).toLocaleString()}</span>
              </Link>
            </li>
          ))}
        </ul>
      </div>
    </main>
  );
}
