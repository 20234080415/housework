import { useEffect, useRef, useState } from "react";
import axios from "axios";
import { fabric } from "fabric";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function Canvas() {
  const canvasElementRef = useRef(null);
  const fabricCanvasRef = useRef(null);
  const pollTimerRef = useRef(null);
  const [brushWidth, setBrushWidth] = useState(6);
  const [tool, setTool] = useState("brush");
  const [prompt, setPrompt] = useState("");
  const [steps, setSteps] = useState(20);
  const [cfgScale, setCfgScale] = useState(7.5);
  const [cnScale] = useState(1.0);
  const [statusText, setStatusText] = useState("等待输入");
  const [isLoading, setIsLoading] = useState(false);
  const [resultImage, setResultImage] = useState("");
  const [errorText, setErrorText] = useState("");

  useEffect(() => {
    const canvas = new fabric.Canvas(canvasElementRef.current, {
      isDrawingMode: true,
      backgroundColor: "#000000",
      width: 512,
      height: 512,
    });

    canvas.freeDrawingBrush.width = brushWidth;
    canvas.freeDrawingBrush.color = "#ffffff";
    fabricCanvasRef.current = canvas;

    return () => {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
      }
      canvas.dispose();
    };
  }, []);

  useEffect(() => {
    const canvas = fabricCanvasRef.current;
    if (!canvas) {
      return;
    }

    canvas.freeDrawingBrush.width = brushWidth;
    canvas.freeDrawingBrush.color = tool === "eraser" ? "#000000" : "#ffffff";
  }, [brushWidth, tool]);

  const handleClear = () => {
    const canvas = fabricCanvasRef.current;
    if (!canvas) {
      return;
    }

    // 清空画布后恢复黑色背景。
    canvas.clear();
    canvas.setBackgroundColor("#000000", canvas.renderAll.bind(canvas));
    canvas.isDrawingMode = true;
  };

  const stopPolling = () => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  };

  const buildImageSrc = (base64) => {
    if (!base64) {
      return "";
    }
    return base64.startsWith("data:image") ? base64 : `data:image/png;base64,${base64}`;
  };

  const pollTaskStatus = (taskId) => {
    stopPolling();
    pollTimerRef.current = setInterval(async () => {
      try {
        const response = await axios.get(`${API_URL}/api/status/${taskId}`);
        const payload = response.data;

        if (payload.code === 404) {
          stopPolling();
          setIsLoading(false);
          setStatusText("等待输入");
          setErrorText(payload.msg || "任务不存在");
          return;
        }

        const taskStatus = payload.data?.status;
        if (taskStatus === "done") {
          stopPolling();
          setResultImage(buildImageSrc(payload.data?.result_base64));
          setIsLoading(false);
          setStatusText("生成完成");
          return;
        }

        if (taskStatus === "failed") {
          stopPolling();
          setIsLoading(false);
          setStatusText("等待输入");
          setErrorText("生成失败，请稍后重试");
        }
      } catch (error) {
        stopPolling();
        setIsLoading(false);
        setStatusText("等待输入");
        setErrorText(error.message || "状态查询失败");
      }
    }, 2000);
  };

  const handleGenerate = async () => {
    const canvas = fabricCanvasRef.current;
    if (!canvas || !prompt.trim()) {
      setErrorText("请输入文本提示词");
      return;
    }

    try {
      setErrorText("");
      setResultImage("");
      setIsLoading(true);
      setStatusText("生成中...");

      const sketchBase64 = canvas.toDataURL({
        format: "png",
      });

      const response = await axios.post(`${API_URL}/api/generate`, {
        sketch_base64: sketchBase64,
        prompt,
        steps,
        cfg_scale: cfgScale,
        cn_scale: cnScale,
      });
      const payload = response.data;
      const taskId = payload.data?.task_id;

      if (!taskId) {
        throw new Error(payload.msg || "任务创建失败");
      }

      pollTaskStatus(taskId);
    } catch (error) {
      setIsLoading(false);
      setStatusText("等待输入");
      setErrorText(error.message || "生成请求失败");
    }
  };

  return (
    <section style={styles.workspace}>
      <style>{`
        @keyframes canvas-spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>

      <div style={styles.panel}>
        <div style={styles.panelHeader}>
          <h2 style={styles.title}>画板</h2>
          <span style={styles.badge}>{tool === "eraser" ? "橡皮擦" : "画笔"}</span>
        </div>
        <div style={styles.canvasFrame}>
          <canvas ref={canvasElementRef} width="512" height="512" />
        </div>
        <div style={styles.toolRow}>
          <button
            type="button"
            style={tool === "brush" ? styles.activeButton : styles.button}
            onClick={() => setTool("brush")}
          >
            画笔
          </button>
          <button
            type="button"
            style={tool === "eraser" ? styles.activeButton : styles.button}
            onClick={() => setTool("eraser")}
          >
            橡皮擦
          </button>
          <button type="button" style={styles.secondaryButton} onClick={handleClear}>
            清空
          </button>
        </div>
        <label style={styles.label}>
          粗细 {brushWidth}px
          <input
            style={styles.range}
            type="range"
            min="2"
            max="20"
            value={brushWidth}
            onChange={(event) => setBrushWidth(Number(event.target.value))}
          />
        </label>
      </div>

      <div style={styles.panel}>
        <h2 style={styles.title}>控制</h2>
        <label style={styles.label}>
          提示词
          <textarea
            style={styles.textarea}
            value={prompt}
            placeholder="描述你想生成的图像..."
            onChange={(event) => setPrompt(event.target.value)}
          />
        </label>
        <label style={styles.label}>
          steps {steps}
          <input
            style={styles.range}
            type="range"
            min="10"
            max="50"
            value={steps}
            onChange={(event) => setSteps(Number(event.target.value))}
          />
        </label>
        <label style={styles.label}>
          cfg_scale {cfgScale.toFixed(1)}
          <input
            style={styles.range}
            type="range"
            min="1"
            max="15"
            step="0.1"
            value={cfgScale}
            onChange={(event) => setCfgScale(Number(event.target.value))}
          />
        </label>
        <button
          type="button"
          style={styles.generateButton}
          disabled={isLoading}
          onClick={handleGenerate}
        >
          生成图像
        </button>
        {errorText ? <p style={styles.error}>{errorText}</p> : null}
      </div>

      <div style={styles.panel}>
        <div style={styles.panelHeader}>
          <h2 style={styles.title}>结果</h2>
          <span style={styles.status}>{statusText}</span>
        </div>
        <div style={styles.resultFrame}>
          {isLoading ? <div style={styles.spinner} /> : null}
          {!isLoading && resultImage ? (
            <img style={styles.resultImage} src={resultImage} alt="生成结果" />
          ) : null}
          {!isLoading && !resultImage ? <span style={styles.emptyText}>等待输入</span> : null}
        </div>
      </div>
    </section>
  );
}

const styles = {
  workspace: {
    display: "grid",
    gridTemplateColumns: "minmax(280px, 540px) minmax(240px, 320px) minmax(280px, 540px)",
    gap: "20px",
    alignItems: "start",
    maxWidth: "1440px",
    margin: "0 auto",
  },
  panel: {
    minWidth: 0,
  },
  panelHeader: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "12px",
    marginBottom: "12px",
  },
  title: {
    margin: "0 0 12px",
    fontSize: "18px",
    fontWeight: 700,
  },
  badge: {
    padding: "4px 8px",
    borderRadius: "6px",
    color: "#ffffff",
    background: "#111827",
    fontSize: "13px",
  },
  canvasFrame: {
    width: "512px",
    maxWidth: "100%",
    aspectRatio: "1 / 1",
    border: "1px solid #111827",
    background: "#000000",
    overflow: "hidden",
  },
  toolRow: {
    display: "flex",
    gap: "8px",
    flexWrap: "wrap",
    marginTop: "14px",
  },
  button: {
    border: 0,
    borderRadius: "6px",
    padding: "10px 14px",
    color: "#111827",
    background: "#e5e7eb",
    cursor: "pointer",
  },
  activeButton: {
    border: 0,
    borderRadius: "6px",
    padding: "10px 14px",
    color: "#ffffff",
    background: "#111827",
    cursor: "pointer",
  },
  secondaryButton: {
    border: "1px solid #d1d5db",
    borderRadius: "6px",
    padding: "10px 14px",
    color: "#111827",
    background: "#ffffff",
    cursor: "pointer",
  },
  label: {
    display: "grid",
    gap: "8px",
    marginTop: "16px",
    fontSize: "14px",
    fontWeight: 600,
  },
  range: {
    width: "100%",
  },
  textarea: {
    width: "100%",
    minHeight: "160px",
    resize: "vertical",
    border: "1px solid #d1d5db",
    borderRadius: "6px",
    padding: "12px",
    font: "inherit",
    lineHeight: 1.5,
  },
  generateButton: {
    width: "100%",
    marginTop: "20px",
    border: 0,
    borderRadius: "6px",
    padding: "12px 16px",
    color: "#ffffff",
    background: "#2563eb",
    cursor: "pointer",
  },
  error: {
    margin: "12px 0 0",
    color: "#b91c1c",
    fontSize: "14px",
  },
  status: {
    color: "#4b5563",
    fontSize: "14px",
  },
  resultFrame: {
    display: "grid",
    placeItems: "center",
    width: "512px",
    maxWidth: "100%",
    aspectRatio: "1 / 1",
    border: "1px solid #d1d5db",
    background: "#ffffff",
    overflow: "hidden",
  },
  spinner: {
    width: "48px",
    height: "48px",
    border: "5px solid #dbeafe",
    borderTopColor: "#2563eb",
    borderRadius: "50%",
    animation: "canvas-spin 0.9s linear infinite",
  },
  resultImage: {
    width: "100%",
    height: "100%",
    objectFit: "contain",
  },
  emptyText: {
    color: "#6b7280",
    fontSize: "14px",
  },
};
