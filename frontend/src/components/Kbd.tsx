export default function Kbd({ children }: { children: string }) {
  return (
    <kbd className="ml-1.5 rounded border border-zinc-600 bg-zinc-800 px-1.5 py-0.5 text-[0.7rem] font-mono text-zinc-400">
      {children}
    </kbd>
  );
}
