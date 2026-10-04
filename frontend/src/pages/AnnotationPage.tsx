import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ClassDef, getImage, getProject, ImageSummary } from "../api/client";
import { useToast } from "../components/ToastProvider";
import AnnotationCanvas from "../features/annotation/AnnotationCanvas";
import ClassEditor from "../features/annotation/ClassEditor";

export default function AnnotationPage() {
  const { imageId } = useParams<{ imageId: string }>();
  const [image, setImage] = useState<ImageSummary | null>(null);
  const [classes, setClasses] = useState<ClassDef[] | null>(null);
  const [editing, setEditing] = useState(false);
  const { showError } = useToast();

  useEffect(() => {
    if (!imageId) return;
    getImage(imageId)
      .then((img) => {
        setImage(img);
        return getProject(img.project_id).then((p) => setClasses(p.classes));
      })
      .catch((e) => showError(e, "Não foi possível carregar a imagem."));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageId]);

  if (!imageId) return null;

  return (
    <div className="flex h-screen flex-col bg-zinc-950 text-zinc-100">
      <header className="flex items-center gap-4 border-b border-zinc-800 px-5 py-3">
        <Link to={image ? `/projects/${image.project_id}` : "/"} className="text-sm text-zinc-400 hover:text-zinc-200">
          ← Voltar
        </Link>
        <strong className="text-sm">{image?.filename}</strong>
        {image && (
          <span className="text-xs text-zinc-500">
            {image.width}×{image.height}px
          </span>
        )}
        <button
          onClick={() => setEditing((v) => !v)}
          className={`ml-auto rounded-lg border px-3 py-1.5 text-sm ${
            editing ? "border-blue-500 bg-blue-950/50 text-blue-200" : "border-zinc-700 text-zinc-300 hover:bg-zinc-800"
          }`}
        >
          Classes {classes ? `(${classes.length})` : ""}
        </button>
      </header>
      {editing && image && classes && (
        <div className="border-b border-zinc-800 px-5 py-3">
          <ClassEditor projectId={image.project_id} classes={classes} onChange={setClasses} />
        </div>
      )}
      {classes && <AnnotationCanvas imageId={imageId} classes={classes} />}
    </div>
  );
}
