import { FormEvent, useState } from "react";
import { Link, Navigate, useLocation, useNavigate } from "react-router-dom";
import { ApiError } from "../api/client";
import { useAuth } from "../components/AuthProvider";

const MIN_PASSWORD = 8;

function messageFor(e: unknown, mode: "login" | "register"): string {
  if (e instanceof ApiError) {
    if (e.status === 401) return "E-mail ou senha incorretos.";
    if (e.status === 409) return "Este e-mail já está cadastrado.";
    if (e.status === 422) return "Verifique os dados informados.";
    return e.userMessage;
  }
  return mode === "login" ? "Não foi possível entrar." : "Não foi possível criar a conta.";
}

export default function AuthPage({ mode }: { mode: "login" | "register" }) {
  const { user, login, register } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const from = (location.state as { from?: string } | null)?.from ?? "/";

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (user) return <Navigate to={from} replace />;

  const isLogin = mode === "login";

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    if (!isLogin && password.length < MIN_PASSWORD) {
      setError(`A senha deve ter pelo menos ${MIN_PASSWORD} caracteres.`);
      return;
    }
    setBusy(true);
    try {
      if (isLogin) await login(email.trim(), password);
      else await register(name.trim(), email.trim(), password);
      navigate(from, { replace: true });
    } catch (err) {
      setError(messageFor(err, mode));
    } finally {
      setBusy(false);
    }
  };

  const field =
    "w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-zinc-100 placeholder:text-zinc-500 focus:border-blue-500 focus:outline-none";

  return (
    <main className="mx-auto max-w-sm px-6 py-16">
      <h1 className="text-2xl font-semibold text-zinc-100">Rock Segmentation</h1>
      <p className="mt-1 text-zinc-400">{isLogin ? "Entre na sua conta." : "Crie sua conta."}</p>

      <form onSubmit={onSubmit} className="mt-6 flex flex-col gap-3">
        {!isLogin && (
          <label className="flex flex-col gap-1 text-sm text-zinc-300">
            Nome
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              autoComplete="name"
              required
              maxLength={100}
              className={field}
            />
          </label>
        )}
        <label className="flex flex-col gap-1 text-sm text-zinc-300">
          E-mail
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="email"
            required
            className={field}
          />
        </label>
        <label className="flex flex-col gap-1 text-sm text-zinc-300">
          Senha
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete={isLogin ? "current-password" : "new-password"}
            required
            className={field}
          />
          {!isLogin && <span className="text-xs text-zinc-500">Mínimo de {MIN_PASSWORD} caracteres.</span>}
        </label>

        {error && (
          <p role="alert" className="rounded-lg border border-red-500/40 bg-red-950/60 px-3 py-2 text-sm text-red-100">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={busy}
          className="rounded-lg bg-blue-600 px-4 py-2 font-medium text-white hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {busy ? "Aguarde..." : isLogin ? "Entrar" : "Criar conta"}
        </button>
      </form>

      <p className="mt-4 text-sm text-zinc-400">
        {isLogin ? "Ainda não tem conta? " : "Já tem conta? "}
        <Link
          to={isLogin ? "/register" : "/login"}
          state={location.state}
          className="text-blue-400 hover:text-blue-300"
        >
          {isLogin ? "Criar conta" : "Entrar"}
        </Link>
      </p>
    </main>
  );
}
