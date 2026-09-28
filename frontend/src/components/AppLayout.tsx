import { Link, NavLink, Outlet } from "react-router-dom";
import { useAuth } from "./AuthProvider";

const navClass = ({ isActive }: { isActive: boolean }) =>
  `rounded-md px-2.5 py-1 text-sm ${isActive ? "bg-zinc-800 text-zinc-100" : "text-zinc-400 hover:text-zinc-200"}`;

export default function AppLayout() {
  const { user, logout } = useAuth();
  return (
    <>
      <header className="border-b border-zinc-800">
        <div className="mx-auto flex max-w-5xl items-center gap-4 px-6 py-3">
          <Link to="/" className="font-semibold text-zinc-100">
            Rock Segmentation
          </Link>
          <nav className="flex items-center gap-1">
            <NavLink to="/" end className={navClass}>
              Projetos
            </NavLink>
            <NavLink to="/account" className={navClass}>
              Minha conta
            </NavLink>
          </nav>
          <div className="ml-auto flex items-center gap-3 text-sm">
            <span className="text-zinc-400">{user?.name}</span>
            <button
              onClick={() => void logout()}
              className="rounded-lg border border-zinc-700 px-3 py-1 text-zinc-200 hover:bg-zinc-800"
            >
              Sair
            </button>
          </div>
        </div>
      </header>
      <Outlet />
    </>
  );
}
