import { useEffect, useState } from "react";
import { ApiError, ClassDef, updateClasses } from "../../api/client";
import { useToast } from "../../components/ToastProvider";

const PALETTE = ["#2563eb", "#ea580c", "#16a34a", "#9333ea", "#dc2626", "#0891b2", "#ca8a04", "#db2777"];

interface Props {
  projectId: string;
  classes: ClassDef[];
  onChange: (classes: ClassDef[]) => void;
}

// Edits the project's class list. Indices are fixed by position (they are
// what the masks store), so classes can be renamed or recolored freely but
// only the last one can be removed — and only if no mask uses it.
export default function ClassEditor({ projectId, classes, onChange }: Props) {
  const [draft, setDraft] = useState<ClassDef[]>(classes);
  const [saving, setSaving] = useState(false);
  const { showError, showSuccess } = useToast();

  useEffect(() => setDraft(classes), [classes]);

  const dirty = JSON.stringify(draft) !== JSON.stringify(classes);

  const update = (i: number, patch: Partial<ClassDef>) =>
    setDraft((d) => d.map((c, j) => (j === i ? { ...c, ...patch } : c)));

  const add = () =>
    setDraft((d) => [...d, { index: d.length + 1, name: `Classe ${d.length + 1}`, color: PALETTE[d.length % PALETTE.length] }]);

  const removeLast = () => setDraft((d) => (d.length > 1 ? d.slice(0, -1) : d));

  const save = async () => {
    setSaving(true);
    try {
      const saved = await updateClasses(projectId, draft);
      onChange(saved);
      showSuccess("Classes salvas.");
    } catch (e) {
      // 409 / 422 carry a Portuguese reason (which class is still in use,
      // what is invalid) worth showing as-is.
      const explained = e instanceof ApiError && (e.status === 409 || e.status === 422);
      showError(e, explained ? undefined : "Não foi possível salvar as classes.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-2 rounded-lg border border-zinc-800 bg-zinc-950 p-3">
      {draft.map((c, i) => (
        <div key={c.index} className="flex items-center gap-2">
          <span className="w-5 text-right text-xs tabular-nums text-zinc-500">{c.index}</span>
          <input
            type="color"
            value={c.color}
            onChange={(e) => update(i, { color: e.target.value })}
            className="h-7 w-9 cursor-pointer rounded border border-zinc-700 bg-zinc-900"
            aria-label={`Cor da classe ${c.name}`}
          />
          <input
            value={c.name}
            maxLength={40}
            onChange={(e) => update(i, { name: e.target.value })}
            className="w-40 rounded-lg border border-zinc-700 bg-zinc-900 px-2 py-1 text-sm text-zinc-200"
            aria-label={`Nome da classe ${c.index}`}
          />
          {c.index === 1 && <span className="text-xs text-zinc-500">porosidade = fração desta classe</span>}
        </div>
      ))}
      <div className="flex items-center gap-2 pt-1">
        <button onClick={add} className="rounded-lg border border-zinc-700 px-3 py-1 text-xs text-zinc-300 hover:bg-zinc-800">
          + Classe
        </button>
        <button
          onClick={removeLast}
          disabled={draft.length <= 1}
          title="Só a última classe pode ser removida, e apenas se nenhuma anotação a usar"
          className="rounded-lg border border-zinc-700 px-3 py-1 text-xs text-zinc-300 hover:bg-zinc-800 disabled:opacity-40"
        >
          − Última
        </button>
        <button
          onClick={save}
          disabled={!dirty || saving}
          className="ml-auto rounded-lg bg-blue-600 px-3 py-1 text-xs font-medium text-white hover:bg-blue-500 disabled:opacity-40"
        >
          {saving ? "Salvando..." : "Salvar"}
        </button>
      </div>
    </div>
  );
}
