import { useEffect, useState } from "react";

type ApiStatus = "loading" | "ok" | "error";

/**
 * Fase 0 placeholder: proves the frontend/backend wiring works end to end
 * (build, docker compose, CORS, proxy) before any real screen exists.
 * Fase 1 replaces this with the annotation canvas (plan §5.1).
 */
export default function App() {
  const [status, setStatus] = useState<ApiStatus>("loading");

  useEffect(() => {
    fetch("/api/health")
      .then((res) => (res.ok ? res.json() : Promise.reject(res)))
      .then(() => setStatus("ok"))
      .catch(() => setStatus("error"));
  }, []);

  return (
    <main className="app">
      <h1>Rock Segmentation</h1>
      <p>Versão web — anotação, treino e segmentação poro × sólido.</p>
      <p>
        Status da API:{" "}
        <span className={`status ${status}`}>
          {status === "loading" && "consultando..."}
          {status === "ok" && "conectado"}
          {status === "error" && "indisponível"}
        </span>
      </p>
    </main>
  );
}
