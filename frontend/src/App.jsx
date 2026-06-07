import Canvas from "./Canvas.jsx";

export default function App() {
  return (
    <main className="app">
      <section className="toolbar">
        <h1>ControlNet Sketch</h1>
        <button type="button">生成图像</button>
      </section>
      <Canvas />
    </main>
  );
}
