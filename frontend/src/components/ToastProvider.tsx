import { createContext, ReactNode, useCallback, useContext, useRef, useState } from "react";
import { ApiError } from "../api/client";

type ToastKind = "error" | "success" | "info";

interface Toast {
  id: number;
  kind: ToastKind;
  message: string;
}

interface ToastContextValue {
  showError: (error: unknown, fallback?: string) => void;
  showSuccess: (message: string) => void;
  showInfo: (message: string) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

const AUTO_DISMISS_MS = 5000;

const KIND_STYLES: Record<ToastKind, string> = {
  error: "border-red-500/40 bg-red-950/90 text-red-100",
  success: "border-emerald-500/40 bg-emerald-950/90 text-emerald-100",
  info: "border-zinc-500/40 bg-zinc-900/90 text-zinc-100",
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const nextId = useRef(1);

  const dismiss = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const push = useCallback(
    (kind: ToastKind, message: string) => {
      const id = nextId.current++;
      setToasts((prev) => [...prev, { id, kind, message }]);
      setTimeout(() => dismiss(id), AUTO_DISMISS_MS);
    },
    [dismiss],
  );

  const showError = useCallback(
    (error: unknown, fallback?: string) => {
      // The backend's `detail` field is written for developers (English,
      // terse — "project not found"), not end users, so a caller-supplied
      // Portuguese fallback always wins when given. Without one we still
      // never show a raw error/stack trace — just a generic sentence.
      console.error(error);
      if (fallback) {
        push("error", fallback);
      } else if (error instanceof ApiError) {
        push("error", error.userMessage);
      } else {
        push("error", "Algo deu errado. Tente novamente.");
      }
    },
    [push],
  );

  const showSuccess = useCallback((message: string) => push("success", message), [push]);
  const showInfo = useCallback((message: string) => push("info", message), [push]);

  return (
    <ToastContext.Provider value={{ showError, showSuccess, showInfo }}>
      {children}
      <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-full max-w-sm flex-col gap-2">
        {toasts.map((t) => (
          <div
            key={t.id}
            role="alert"
            className={`pointer-events-auto flex items-start justify-between gap-3 rounded-lg border px-4 py-3 text-sm shadow-lg backdrop-blur ${KIND_STYLES[t.kind]}`}
          >
            <span>{t.message}</span>
            <button
              onClick={() => dismiss(t.id)}
              className="shrink-0 text-xs text-current/70 hover:text-current"
              aria-label="Fechar"
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used within a ToastProvider");
  return ctx;
}
