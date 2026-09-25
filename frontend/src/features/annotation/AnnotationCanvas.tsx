import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { clearMask, fetchMaskBlob, imageFileUrl, saveMask } from "../../api/client";
import { useToast } from "../../components/ToastProvider";
import { useConfirm } from "../../components/ConfirmProvider";
import Kbd from "../../components/Kbd";

// Mask encoding on disk (plan §4.3): a PNG where each pixel's value is the
// class index — 0 = unannotated, 1 = poro, 2 = solido. We keep that same
// index in memory as a flat Uint8Array (`maskData`, one byte per pixel) and
// only turn it into translucent display colors when compositing a frame.

type Tool = "brush" | "eraser";
type AnnotationClass = 1 | 2; // 1 = poro, 2 = solido

const CLASS_COLOR: Record<AnnotationClass, [number, number, number]> = {
  1: [37, 99, 235], // poro — blue
  2: [234, 88, 12], // solido — orange
};
const OVERLAY_ALPHA = 150;
const AUTOSAVE_DEBOUNCE_MS = 2000;
const UNDO_LIMIT = 40;

interface Props {
  imageId: string;
}

type SaveState = "idle" | "pending" | "saving" | "saved" | "error";

export default function AnnotationCanvas({ imageId }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const baseCanvasRef = useRef<HTMLCanvasElement | null>(null); // offscreen: original image
  const scratchCanvasRef = useRef<HTMLCanvasElement | null>(null); // offscreen: mask overlay compositing scratch
  const maskDataRef = useRef<Uint8Array | null>(null); // one class index per pixel
  const dimsRef = useRef({ width: 0, height: 0 });

  const undoStack = useRef<Uint8Array[]>([]);
  const redoStack = useRef<Uint8Array[]>([]);
  const strokeStartSnapshot = useRef<Uint8Array | null>(null);

  const [loaded, setLoaded] = useState(false);
  const [tool, setTool] = useState<Tool>("brush");
  const [activeClass, setActiveClass] = useState<AnnotationClass>(1);
  const [brushSize, setBrushSize] = useState(16);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [counts, setCounts] = useState({ poro: 0, solido: 0 });
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [, forceRender] = useState(0);

  const { showError } = useToast();
  const confirm = useConfirm();

  const spaceHeld = useRef(false);
  const isPanning = useRef(false);
  const isPainting = useRef(false);
  const lastPoint = useRef<{ x: number; y: number } | null>(null);
  const lastScreen = useRef<{ x: number; y: number } | null>(null);
  const autosaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const recountAll = useCallback(() => {
    const data = maskDataRef.current;
    if (!data) return;
    let poro = 0;
    let solido = 0;
    for (let i = 0; i < data.length; i++) {
      if (data[i] === 1) poro++;
      else if (data[i] === 2) solido++;
    }
    setCounts({ poro, solido });
  }, []);

  const composite = useCallback(() => {
    const canvas = canvasRef.current;
    const base = baseCanvasRef.current;
    const data = maskDataRef.current;
    if (!canvas || !base || !data) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const { width, height } = dimsRef.current;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = false;
    ctx.setTransform(zoom, 0, 0, zoom, pan.x, pan.y);
    ctx.drawImage(base, 0, 0);

    // Overlay the mask, translated from class indices to translucent color.
    const overlay = ctx.createImageData(width, height);
    const out = overlay.data;
    for (let i = 0, p = 0; i < data.length; i++, p += 4) {
      const cls = data[i];
      if (cls === 0) continue;
      const [r, g, b] = CLASS_COLOR[cls as AnnotationClass];
      out[p] = r;
      out[p + 1] = g;
      out[p + 2] = b;
      out[p + 3] = OVERLAY_ALPHA;
    }
    // createImageData + drawImage needs an intermediate canvas since
    // putImageData ignores the current transform.
    let scratch = scratchCanvasRef.current;
    if (!scratch || scratch.width !== width || scratch.height !== height) {
      scratch = document.createElement("canvas");
      scratch.width = width;
      scratch.height = height;
      scratchCanvasRef.current = scratch;
    }
    scratch.getContext("2d")!.putImageData(overlay, 0, 0);
    ctx.drawImage(scratch, 0, 0);
  }, [zoom, pan]);

  // Load image + existing mask once.
  useEffect(() => {
    let cancelled = false;
    setLoaded(false);

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = imageFileUrl(imageId);
    img.onload = async () => {
      if (cancelled) return;
      const width = img.naturalWidth;
      const height = img.naturalHeight;
      dimsRef.current = { width, height };

      const base = document.createElement("canvas");
      base.width = width;
      base.height = height;
      base.getContext("2d")!.drawImage(img, 0, 0);
      baseCanvasRef.current = base;

      let data = new Uint8Array(width * height);
      const maskBlob = await fetchMaskBlob(imageId);
      if (maskBlob && !cancelled) {
        const maskImg = await createImageBitmap(maskBlob);
        const tmp = document.createElement("canvas");
        tmp.width = width;
        tmp.height = height;
        const tctx = tmp.getContext("2d")!;
        tctx.drawImage(maskImg, 0, 0);
        const px = tctx.getImageData(0, 0, width, height).data;
        for (let i = 0, p = 0; i < data.length; i++, p += 4) {
          data[i] = px[p] as 0 | 1 | 2;
        }
      }
      if (cancelled) return;
      maskDataRef.current = data;
      undoStack.current = [];
      redoStack.current = [];

      const canvas = canvasRef.current;
      const container = containerRef.current;
      if (canvas && container) {
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
        const fitZoom = Math.min(canvas.width / width, canvas.height / height, 1);
        setZoom(fitZoom);
        setPan({
          x: (canvas.width - width * fitZoom) / 2,
          y: (canvas.height - height * fitZoom) / 2,
        });
      }

      setLoaded(true);
      recountAll();
    };

    return () => {
      cancelled = true;
    };
  }, [imageId, recountAll]);

  useEffect(() => {
    if (loaded) composite();
  }, [loaded, composite]);

  const scheduleAutosave = useCallback(() => {
    setSaveState("pending");
    if (autosaveTimer.current) clearTimeout(autosaveTimer.current);
    autosaveTimer.current = setTimeout(async () => {
      const data = maskDataRef.current;
      const { width, height } = dimsRef.current;
      if (!data || !width || !height) return;
      setSaveState("saving");
      try {
        const exportCanvas = document.createElement("canvas");
        exportCanvas.width = width;
        exportCanvas.height = height;
        const ectx = exportCanvas.getContext("2d")!;
        const imgData = ectx.createImageData(width, height);
        const out = imgData.data;
        for (let i = 0, p = 0; i < data.length; i++, p += 4) {
          const v = data[i];
          out[p] = v;
          out[p + 1] = v;
          out[p + 2] = v;
          out[p + 3] = 255;
        }
        ectx.putImageData(imgData, 0, 0);
        const blob: Blob = await new Promise((resolve, reject) =>
          exportCanvas.toBlob((b) => (b ? resolve(b) : reject(new Error("toBlob failed"))), "image/png"),
        );
        await saveMask(imageId, blob);
        setSaveState("saved");
      } catch (e) {
        setSaveState("error");
        showError(e, "Não foi possível salvar a anotação.");
      }
    }, AUTOSAVE_DEBOUNCE_MS);
  }, [imageId, showError]);

  const imageCoordsFromEvent = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const screenX = e.clientX - rect.left;
      const screenY = e.clientY - rect.top;
      return { x: (screenX - pan.x) / zoom, y: (screenY - pan.y) / zoom, screenX, screenY };
    },
    [pan, zoom],
  );

  const stampAt = useCallback((x: number, y: number, radius: number, value: 0 | AnnotationClass) => {
    const data = maskDataRef.current;
    const { width, height } = dimsRef.current;
    if (!data) return;
    const x0 = Math.max(0, Math.floor(x - radius));
    const x1 = Math.min(width - 1, Math.ceil(x + radius));
    const y0 = Math.max(0, Math.floor(y - radius));
    const y1 = Math.min(height - 1, Math.ceil(y + radius));
    const r2 = radius * radius;
    for (let py = y0; py <= y1; py++) {
      for (let px = x0; px <= x1; px++) {
        const dx = px + 0.5 - x;
        const dy = py + 0.5 - y;
        if (dx * dx + dy * dy <= r2) data[py * width + px] = value;
      }
    }
  }, []);

  const stampSegment = useCallback(
    (from: { x: number; y: number }, to: { x: number; y: number }, radius: number, value: 0 | AnnotationClass) => {
      const dist = Math.hypot(to.x - from.x, to.y - from.y);
      const steps = Math.max(1, Math.ceil(dist / Math.max(1, radius / 2)));
      for (let s = 0; s <= steps; s++) {
        const t = s / steps;
        stampAt(from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t, radius, value);
      }
    },
    [stampAt],
  );

  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (!loaded) return;
      const pt = imageCoordsFromEvent(e);
      if (!pt) return;
      (e.target as HTMLCanvasElement).setPointerCapture(e.pointerId);

      if (spaceHeld.current || e.button === 1) {
        isPanning.current = true;
        lastScreen.current = { x: pt.screenX, y: pt.screenY };
        return;
      }

      isPainting.current = true;
      strokeStartSnapshot.current = maskDataRef.current ? maskDataRef.current.slice() : null;
      lastPoint.current = { x: pt.x, y: pt.y };
      const value: 0 | AnnotationClass = tool === "eraser" ? 0 : activeClass;
      stampAt(pt.x, pt.y, brushSize / 2, value);
      composite();
    },
    [loaded, imageCoordsFromEvent, tool, activeClass, brushSize, stampAt, composite],
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      const pt = imageCoordsFromEvent(e);
      if (!pt) return;

      if (isPanning.current && lastScreen.current) {
        const dx = pt.screenX - lastScreen.current.x;
        const dy = pt.screenY - lastScreen.current.y;
        lastScreen.current = { x: pt.screenX, y: pt.screenY };
        setPan((p) => ({ x: p.x + dx, y: p.y + dy }));
        return;
      }

      if (isPainting.current && lastPoint.current) {
        const value: 0 | AnnotationClass = tool === "eraser" ? 0 : activeClass;
        stampSegment(lastPoint.current, pt, brushSize / 2, value);
        lastPoint.current = { x: pt.x, y: pt.y };
        composite();
      }
    },
    [imageCoordsFromEvent, tool, activeClass, brushSize, stampSegment, composite],
  );

  const endStroke = useCallback(() => {
    if (isPainting.current && strokeStartSnapshot.current) {
      undoStack.current.push(strokeStartSnapshot.current);
      if (undoStack.current.length > UNDO_LIMIT) undoStack.current.shift();
      redoStack.current = [];
      strokeStartSnapshot.current = null;
      recountAll();
      scheduleAutosave();
    }
    isPainting.current = false;
    isPanning.current = false;
    lastPoint.current = null;
    lastScreen.current = null;
  }, [recountAll, scheduleAutosave]);

  const undo = useCallback(() => {
    const data = maskDataRef.current;
    const prev = undoStack.current.pop();
    if (!prev || !data) return;
    redoStack.current.push(data.slice());
    maskDataRef.current = prev;
    composite();
    recountAll();
    scheduleAutosave();
  }, [composite, recountAll, scheduleAutosave]);

  const redo = useCallback(() => {
    const data = maskDataRef.current;
    const next = redoStack.current.pop();
    if (!next || !data) return;
    undoStack.current.push(data.slice());
    maskDataRef.current = next;
    composite();
    recountAll();
    scheduleAutosave();
  }, [composite, recountAll, scheduleAutosave]);

  const clearAnnotation = useCallback(async () => {
    const { width, height } = dimsRef.current;
    if (!width || !height) return;
    const ok = await confirm({
      title: "Limpar anotação?",
      description: "Toda a máscara pintada nesta imagem será apagada.",
      confirmLabel: "Limpar",
      danger: true,
    });
    if (!ok) return;
    if (autosaveTimer.current) clearTimeout(autosaveTimer.current);
    maskDataRef.current = new Uint8Array(width * height);
    undoStack.current = [];
    redoStack.current = [];
    composite();
    recountAll();
    setSaveState("saving");
    try {
      await clearMask(imageId);
      setSaveState("idle");
    } catch (e) {
      setSaveState("error");
      showError(e, "Não foi possível limpar a anotação.");
    }
  }, [imageId, confirm, composite, recountAll, showError]);

  const fitToScreen = useCallback(() => {
    const canvas = canvasRef.current;
    const { width, height } = dimsRef.current;
    if (!canvas || !width || !height) return;
    const fitZoom = Math.min(canvas.width / width, canvas.height / height, 1);
    setZoom(fitZoom);
    setPan({ x: (canvas.width - width * fitZoom) / 2, y: (canvas.height - height * fitZoom) / 2 });
  }, []);

  // Keyboard shortcuts: 1/2 select class, space+drag pans, ctrl+z / ctrl+shift+z undo/redo, 0 fits.
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        spaceHeld.current = true;
        e.preventDefault();
      } else if (e.key === "1") {
        setActiveClass(1);
        setTool("brush");
      } else if (e.key === "2") {
        setActiveClass(2);
        setTool("brush");
      } else if (e.key.toLowerCase() === "b") {
        setTool("brush");
      } else if (e.key.toLowerCase() === "e") {
        setTool("eraser");
      } else if (e.key === "0") {
        fitToScreen();
      } else if (e.key.toLowerCase() === "c") {
        clearAnnotation();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (e.shiftKey) redo();
        else undo();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") {
        e.preventDefault();
        redo();
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === "Space") spaceHeld.current = false;
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [fitToScreen, undo, redo, clearAnnotation]);

  const onWheel = useCallback(
    (e: React.WheelEvent<HTMLCanvasElement>) => {
      e.preventDefault();
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const screenX = e.clientX - rect.left;
      const screenY = e.clientY - rect.top;
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      setZoom((z) => {
        const newZoom = Math.min(20, Math.max(0.05, z * factor));
        setPan((p) => ({
          x: screenX - ((screenX - p.x) / z) * newZoom,
          y: screenY - ((screenY - p.y) / z) * newZoom,
        }));
        return newZoom;
      });
    },
    [],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const observer = new ResizeObserver(() => {
      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
      if (loaded) composite();
      else forceRender((n) => n + 1);
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, [loaded, composite]);

  const cursorStyle = useMemo(() => {
    if (spaceHeld.current) return "grab";
    return tool === "eraser" ? "cell" : "crosshair";
  }, [tool]);

  const saveIndicatorStyle: Record<SaveState, string> = {
    idle: "bg-zinc-800 text-zinc-400",
    pending: "bg-amber-900/50 text-amber-300",
    saving: "bg-amber-900/50 text-amber-300",
    saved: "bg-emerald-900/50 text-emerald-300",
    error: "bg-red-900/50 text-red-300",
  };

  const toolButton = (active: boolean) =>
    `rounded-lg border px-3 py-1.5 text-sm ${
      active ? "border-blue-500 bg-blue-950/50 text-blue-200" : "border-zinc-700 text-zinc-300 hover:bg-zinc-800"
    }`;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex flex-wrap items-center gap-4 border-b border-zinc-800 px-5 py-2.5">
        <div className="flex items-center gap-2">
          <button className={toolButton(tool === "brush")} onClick={() => setTool("brush")}>
            Pincel <Kbd>B</Kbd>
          </button>
          <button className={toolButton(tool === "eraser")} onClick={() => setTool("eraser")}>
            Borracha <Kbd>E</Kbd>
          </button>
        </div>
        <div className="flex items-center gap-2">
          <button
            className={`rounded-lg border px-3 py-1.5 text-sm ${
              activeClass === 1
                ? "border-poro bg-blue-950/50 text-blue-200"
                : "border-zinc-700 text-zinc-300 hover:bg-zinc-800"
            }`}
            onClick={() => setActiveClass(1)}
          >
            Poro <Kbd>1</Kbd>
          </button>
          <button
            className={`rounded-lg border px-3 py-1.5 text-sm ${
              activeClass === 2
                ? "border-solido bg-orange-950/50 text-orange-200"
                : "border-zinc-700 text-zinc-300 hover:bg-zinc-800"
            }`}
            onClick={() => setActiveClass(2)}
          >
            Sólido <Kbd>2</Kbd>
          </button>
        </div>
        <label className="flex items-center gap-2 text-sm text-zinc-300">
          Tamanho
          <input
            type="range"
            min={1}
            max={100}
            value={brushSize}
            onChange={(e) => setBrushSize(Number(e.target.value))}
            className="accent-blue-500"
          />
          <span className="w-10 tabular-nums text-zinc-400">{brushSize}px</span>
        </label>
        <div className="flex items-center gap-2">
          <button
            onClick={undo}
            className="rounded-lg border border-zinc-700 px-3 py-1.5 text-sm text-zinc-300 hover:bg-zinc-800"
          >
            Desfazer <Kbd>Ctrl+Z</Kbd>
          </button>
          <button
            onClick={redo}
            className="rounded-lg border border-zinc-700 px-3 py-1.5 text-sm text-zinc-300 hover:bg-zinc-800"
          >
            Refazer <Kbd>Ctrl+⇧+Z</Kbd>
          </button>
          <button
            onClick={fitToScreen}
            className="rounded-lg border border-zinc-700 px-3 py-1.5 text-sm text-zinc-300 hover:bg-zinc-800"
          >
            Ajustar <Kbd>0</Kbd>
          </button>
          <button
            onClick={clearAnnotation}
            className="rounded-lg border border-red-900 px-3 py-1.5 text-sm text-red-400 hover:bg-red-950"
          >
            Limpar <Kbd>C</Kbd>
          </button>
        </div>
        <div className="flex items-center gap-3 text-sm text-zinc-300">
          <span className="flex items-center gap-1.5">
            <span className="h-3 w-3 rounded-sm bg-poro" /> Poro: {counts.poro.toLocaleString()}
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-3 w-3 rounded-sm bg-solido" /> Sólido: {counts.solido.toLocaleString()}
          </span>
        </div>
        <div className={`ml-auto rounded-full px-3 py-1 text-xs ${saveIndicatorStyle[saveState]}`}>
          {saveState === "idle" && "sem alterações"}
          {saveState === "pending" && "alterações pendentes..."}
          {saveState === "saving" && "salvando..."}
          {saveState === "saved" && "salvo"}
          {saveState === "error" && "erro ao salvar"}
        </div>
      </div>

      <div className="relative min-h-0 flex-1 overflow-hidden bg-zinc-900" ref={containerRef}>
        {!loaded && (
          <div className="absolute inset-0 flex items-center justify-center text-zinc-400">Carregando imagem...</div>
        )}
        <canvas
          ref={canvasRef}
          style={{ cursor: cursorStyle, touchAction: "none" }}
          className="block"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={endStroke}
          onPointerLeave={endStroke}
          onWheel={onWheel}
          onContextMenu={(e) => e.preventDefault()}
        />
      </div>
    </div>
  );
}
