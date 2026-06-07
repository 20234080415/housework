import Canvas from "./Canvas.jsx";

export default function App() {
  return (
    <main className="app">
      <header className="app-header">
        <h1>草图引导图像生成系统</h1>
        <p>基于 ControlNet + Stable Diffusion v1.5</p>
      </header>
      <Canvas />
    </main>
  );
}
