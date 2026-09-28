import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { listMyModels, listMyProjects, modelDownloadUrl, MyModel, MyProject } from "../api/client";
import { useAuth } from "../components/AuthProvider";
import { useToast } from "../components/ToastProvider";

type Tab = "projects" | "models";

function pct(v: unknown): string {
  return typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "—";
}

export default function AccountPage() {
  const { user } = useAuth();
  const { showError } = useToast();
  const [tab, setTab] = useState<Tab>("projects");
  const [projects, setProjects] = useState<MyProject[] | null>(null);
  const [models, setModels] = useState<MyModel[] | null>(null);

  useEffect(() => {
    listMyProjects()
      .then(setProjects)
      .catch((e) => showError(e, "Não foi possível carregar seus projetos."));
    listMyModels()
      .then(setModels)
      .catch((e) => showError(e, "Não foi possível carregar seus modelos."));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const tabClass = (t: Tab) =>
    `border-b-2 px-3 py-2 text-sm font-medium ${
      tab === t ? "border-blue-500 text-zinc-100" : "border-transparent text-zinc-400 hover:text-zinc-200"
    }`;

  return (
    <main className="mx-auto max-w-5xl px-6 py-12">
      <h1 className="text-2xl font-semibold text-zinc-100">{user?.name}</h1>
      <p className="mt-1 text-zinc-400">
        {user?.email}
        {user && <> · membro desde {new Date(user.created_at).toLocaleDateString()}</>}
      </p>

      <div role="tablist" className="mt-8 flex gap-1 border-b border-zinc-800">
        <button role="tab" aria-selected={tab === "projects"} onClick={() => setTab("projects")} className={tabClass("projects")}>
          Projetos{projects && ` (${projects.length})`}
        </button>
        <button role="tab" aria-selected={tab === "models"} onClick={() => setTab("models")} className={tabClass("models")}>
          Modelos{models && ` (${models.length})`}
        </button>
      </div>

      <div className="mt-6">
        {tab === "projects" && (
          <>
            {projects === null && <p className="text-zinc-400">Carregando...</p>}
            {projects?.length === 0 && (
              <p className="text-zinc-400">
                Nenhum projeto ainda.{" "}
                <Link to="/" className="text-blue-400 hover:text-blue-300">
                  Criar o primeiro
                </Link>
              </p>
            )}
            <ul className="flex flex-col gap-2">
              {projects?.map((p) => (
                <li key={p.id}>
                  <Link
                    to={`/projects/${p.id}`}
                    className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-zinc-800 px-4 py-3 hover:border-zinc-600 hover:bg-zinc-900"
                  >
                    <span className="font-medium text-zinc-100">{p.name}</span>
                    <span className="text-xs text-zinc-500">
                      {p.n_images} imagens ({p.n_annotated} anotadas) · {p.n_datasets} datasets · {p.n_models} modelos
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          </>
        )}

        {tab === "models" && (
          <>
            {models === null && <p className="text-zinc-400">Carregando...</p>}
            {models?.length === 0 && (
              <p className="text-zinc-400">Nenhum modelo publicado. Treine e publique um a partir de um projeto.</p>
            )}
            <ul className="flex flex-col gap-2">
              {models?.map((m) => (
                <li key={m.id} className="rounded-lg border border-zinc-800 px-4 py-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-medium text-zinc-100">
                      {m.name} <span className="text-zinc-500">v{m.version}</span>
                    </span>
                    <span className="text-xs text-zinc-500">{new Date(m.created_at).toLocaleString()}</span>
                  </div>
                  <div className="mt-1 flex flex-wrap items-center justify-between gap-2 text-sm text-zinc-400">
                    <span>
                      Projeto{" "}
                      <Link to={`/projects/${m.project_id}`} className="text-blue-400 hover:text-blue-300">
                        {m.project_name}
                      </Link>{" "}
                      · acurácia {pct(m.metrics?.accuracy)}
                    </span>
                    <span className="flex gap-3">
                      {(["pt", "json", "scripted"] as const).map((fmt) => (
                        <a
                          key={fmt}
                          href={modelDownloadUrl(m.id, fmt)}
                          className="text-zinc-300 underline-offset-2 hover:text-zinc-100 hover:underline"
                        >
                          .{fmt}
                        </a>
                      ))}
                    </span>
                  </div>
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </main>
  );
}
