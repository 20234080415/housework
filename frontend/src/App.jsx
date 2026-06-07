import Canvas from "./Canvas.jsx";

export default function App() {
  return (
    <main className="app">
      <header className="app-header">
        <div>
          <p className="eyebrow">ControlNet Sketch Studio</p>
          <h1>草图引导图像生成系统</h1>
          <p className="subtitle">基于 ControlNet + Stable Diffusion v1.5</p>
        </div>
        <div className="header-meta">
          <span>512 x 512</span>
          <span>异步生成</span>
        </div>
      </header>
      <Canvas />
    </main>
  );
}
