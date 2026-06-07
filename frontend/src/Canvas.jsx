import { useEffect, useRef, useState } from "react";
import axios from "axios";
import { fabric } from "fabric";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const CANVAS_SIZE = 512;

export default function Canvas() {
  const canvasElementRef = useRef(null);
  const fabricCanvasRef = useRef(null);
  const pollTimerRef = useRef(null);
  const [brushWidth, setBrushWidth] = useState(6);
  const [tool, setTool] = useState("brush");
  const [prompt, setPrompt] = useState("");
  const [steps, setSteps] = useState(20);
  const [cfgScale, setCfgScale] = useState(7.5);
  const [cnScale, setCnScale] = useState(1.0);
  const [statusText, setStatusText] = useState("等待输入");
  const [isLoading, setIsLoading] = useState(false);
  const [resultImage, setResultImage] = useState("");
  const [errorText, setErrorText] = useState("");
  const [taskId, setTaskId] = useState("");

  useEffect(() => {
    const canvas = new fabric.Canvas(canvasElementRef.current, {
      isDrawingMode: true,
      backgroundColor: "#050505",
      width: CANVAS_SIZE,
      height: CANVAS_SIZE,
      selection: false,
    });

    canvas.freeDrawingBrush.width = brushWidth;
    canvas.freeDrawingBrush.color = "#ffffff";
    fabricCanvasRef.current = canvas;

    return () => {
      stopPolling();
      canvas.dispose();
    };
  }, []);

  useEffect(() => {
    const canvas = fabricCanvasRef.current;
    if (!canvas) {
      return;
    }

    canvas.freeDrawingBrush.width = brushWidth;
    canvas.freeDrawingBrush.color = tool === "eraser" ? "#050505" : "#ffffff";
  }, [brushWidth, tool]);

  const stopPolling = () => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  };

  const resetCanvas = () => {
    const canvas = fabricCanvasRef.current;
    if (!canvas) {
      return;
    }

    // 清空画布后恢复黑色背景。
    canvas.clear();
    canvas.setBackgroundColor("#050505", canvas.renderAll.bind(canvas));
    canvas.isDrawingMode = true;
  };

  const buildImageSrc = (base64) => {
    if (!base64) {
      return "";
    }
    return base64.startsWith("data:image") ? base64 : `data:image/png;base64,${base64}`;
  };

  const updateFailedState = (message) => {
    stopPolling();
    setIsLoading(false);
    setStatusText("等待输入");
    setErrorText(message);
  };

  const pollTaskStatus = (nextTaskId) => {
    stopPolling();
    pollTimerRef.current = setInterval(async () => {
      try {
        const response = await axios.get(`${API_URL}/api/status/${nextTaskId}`);
        const payload = response.data;

        if (payload.code === 404) {
          updateFailedState(payload.msg || "任务不存在");
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
          updateFailedState("生成失败，请稍后重试");
        }
      } catch (error) {
        updateFailedState(error.message || "状态查询失败");
      }
    }, 2000);
  };

  const handleGenerate = async () => {
    const canvas = fabricCanvasRef.current;
    if (!canvas) {
      setErrorText("画板未初始化");
      return;
    }

    if (!prompt.trim()) {
      setErrorText("请输入文本提示词");
      return;
    }

    try {
      setErrorText("");
      setResultImage("");
      setTaskId("");
      setIsLoading(true);
      setStatusText("生成中...");

      const sketchBase64 = canvas.toDataURL({
        format: "png",
      });

      const response = await axios.post(`${API_URL}/api/generate`, {
        sketch_base64: sketchBase64,
        prompt: prompt.trim(),
        steps,
        cfg_scale: cfgScale,
        cn_scale: cnScale,
      });
      const payload = response.data;
      const nextTaskId = payload.data?.task_id;

      if (!nextTaskId) {
        throw new Error(payload.msg || "任务创建失败");
      }

      setTaskId(nextTaskId);
      pollTaskStatus(nextTaskId);
    } catch (error) {
      updateFailedState(error.message || "生成请求失败");
    }
  };

  return (
    <section className="studio">
      <article className="panel drawing-panel">
        <div className="panel-heading">
          <div>
            <p className="section-kicker">Sketch</p>
            <h2>画板</h2>
          </div>
          <span className="status-pill">{tool === "eraser" ? "橡皮擦" : "画笔"}</span>
        </div>

        <div className="canvas-frame">
          <canvas ref={canvasElementRef} width={CANVAS_SIZE} height={CANVAS_SIZE} />
        </div>

        <div className="tool-grid">
          <button
            type="button"
            className={tool === "brush" ? "tool-button active" : "tool-button"}
            onClick={() => setTool("brush")}
          >
            画笔
          </button>
          <button
            type="button"
            className={tool === "eraser" ? "tool-button active" : "tool-button"}
            onClick={() => setTool("eraser")}
          >
            橡皮擦
          </button>
          <button type="button" className="tool-button ghost" onClick={resetCanvas}>
            清空
          </button>
        </div>

        <label className="control-label">
          <span>
            画笔粗细 <strong>{brushWidth}px</strong>
          </span>
          <input
            type="range"
            min="2"
            max="20"
            value={brushWidth}
            onChange={(event) => setBrushWidth(Number(event.target.value))}
          />
        </label>
      </article>

      <article className="panel control-panel">
        <div className="panel-heading">
          <div>
            <p className="section-kicker">Prompt</p>
            <h2>生成控制</h2>
          </div>
        </div>

        <label className="control-label">
          <span>文本提示词</span>
          <textarea
            value={prompt}
            placeholder="描述你想生成的图像..."
            onChange={(event) => setPrompt(event.target.value)}
          />
        </label>

        <div className="slider-stack">
          <label className="control-label">
            <span>
              采样步数 <strong>{steps}</strong>
            </span>
            <input
              type="range"
              min="10"
              max="50"
              value={steps}
              onChange={(event) => setSteps(Number(event.target.value))}
            />
          </label>

          <label className="control-label">
            <span>
              CFG 强度 <strong>{cfgScale.toFixed(1)}</strong>
            </span>
            <input
              type="range"
              min="1"
              max="15"
              step="0.1"
              value={cfgScale}
              onChange={(event) => setCfgScale(Number(event.target.value))}
            />
          </label>

          <label className="control-label">
            <span>
              ControlNet 强度 <strong>{cnScale.toFixed(1)}</strong>
            </span>
            <input
              type="range"
              min="0.1"
              max="2"
              step="0.1"
              value={cnScale}
              onChange={(event) => setCnScale(Number(event.target.value))}
            />
          </label>
        </div>

        <button
          type="button"
          className="generate-button"
          disabled={isLoading}
          onClick={handleGenerate}
        >
          {isLoading ? "生成中..." : "生成图像"}
        </button>

        {taskId ? <p className="task-id">任务 ID：{taskId}</p> : null}
        {errorText ? <p className="error-text">{errorText}</p> : null}
      </article>

      <article className="panel result-panel">
        <div className="panel-heading">
          <div>
            <p className="section-kicker">Result</p>
            <h2>生成结果</h2>
          </div>
          <span className="status-pill">{statusText}</span>
        </div>

        <div className="result-frame">
          {isLoading ? (
            <div className="loading-state">
              <div className="spinner" />
              <p>模型正在生成，请稍候</p>
            </div>
          ) : null}

          {!isLoading && resultImage ? (
            <img className="result-image" src={resultImage} alt="生成结果" />
          ) : null}

          {!isLoading && !resultImage ? (
            <div className="empty-result">
              <span>等待输入</span>
              <p>绘制草图并填写提示词后，结果会显示在这里。</p>
            </div>
          ) : null}
        </div>
      </article>
    </section>
  );
}
