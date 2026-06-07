import { useEffect, useRef } from "react";
import { fabric } from "fabric";

export default function Canvas() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = new fabric.Canvas(canvasRef.current, {
      isDrawingMode: true,
      backgroundColor: "#ffffff",
    });

    canvas.freeDrawingBrush.width = 4;
    canvas.freeDrawingBrush.color = "#111827";

    return () => canvas.dispose();
  }, []);

  return (
    <div className="canvas-shell">
      <canvas ref={canvasRef} width="512" height="512" />
    </div>
  );
}
