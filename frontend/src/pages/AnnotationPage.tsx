import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getImage, ImageSummary } from "../api/client";
import { useToast } from "../components/ToastProvider";
import AnnotationCanvas from "../features/annotation/AnnotationCanvas";

export default function AnnotationPage() {
  const { imageId } = useParams<{ imageId: string }>();
  const [image, setImage] = useState<ImageSummary | null>(null);
  const { showError } = useToast();

  useEffect(() => {
    if (imageId) getImage(imageId).then(setImage).catch((e) => showError(e, "Não foi possível carregar a imagem."));
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
      </header>
      <AnnotationCanvas imageId={imageId} />
    </div>
  );
}
