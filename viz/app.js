// 3D-визуализация внимания: каждый слой модели — этаж, токены стоят в ряд, нити — веса внимания.
// Данные — JSON из scripts/dump_attention.py: {tokens: [...], layers: [слой][голова][i][j]}.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const GAP = 1.25;        // расстояние между токенами по x
const FLOOR = 3.4;       // высота этажа
const MIN_W = 0.03;      // слабее этого веса нить не рисуем
const COOL = new THREE.Color("#1b3a99"), MID = new THREE.Color("#6aa9ff"), WARM = new THREE.Color("#ffb457");

const view = document.getElementById("view");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
view.append(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color("#05070f");
scene.fog = new THREE.FogExp2("#05070f", 0.018);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 200);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.6;
controls.addEventListener("start", () => (controls.autoRotate = false));

const world = new THREE.Group();
scene.add(world);

const state = { data: null, head: "avg", layer: "all", sel: -1 };
let tokenMeshes = [], floorLines = [];

// ── тексты на canvas ───────────────────────────────────────────────────────
function label(text, color = "#cfe0ff") {
  const c = document.createElement("canvas");
  c.width = 256; c.height = 96;
  const g = c.getContext("2d");
  g.font = "500 44px system-ui, sans-serif";
  g.textAlign = "center"; g.textBaseline = "middle";
  g.fillStyle = color;
  g.fillText(text.replace(/ /g, "·"), 128, 50, 240);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, transparent: true, depthWrite: false }));
  s.scale.set(1.6, 0.6, 1);
  return s;
}

function ramp(w) {                       // 0..1 → холодный → синий → тёплый
  const t = Math.min(1, w);
  return t < 0.5 ? COOL.clone().lerp(MID, t * 2) : MID.clone().lerp(WARM, (t - 0.5) * 2);
}

// ── сборка сцены под выбранный пример ──────────────────────────────────────
function clearWorld() {
  while (world.children.length) {
    const o = world.children.pop();
    o.traverse?.((x) => { x.geometry?.dispose?.(); x.material?.map?.dispose?.(); x.material?.dispose?.(); });
  }
  tokenMeshes = []; floorLines = [];
}

function build(data) {
  clearWorld();
  state.data = data;
  state.sel = -1;
  const T = data.tokens.length, L = data.layers.length, H = data.layers[0].length;
  const width = (T - 1) * GAP;
  const boxGeo = new THREE.BoxGeometry(0.9, 0.5, 0.9);
  for (let l = 0; l < L; l++) {
    const y = l * FLOOR;
    const plane = new THREE.Mesh(
      new THREE.BoxGeometry(width + 2.4, 0.06, 2.2),
      new THREE.MeshBasicMaterial({ color: "#0e1740", transparent: true, opacity: 0.85 })
    );
    plane.position.set(0, y - 0.3, 0);
    world.add(plane);
    const edge = new THREE.LineSegments(
      new THREE.EdgesGeometry(plane.geometry),
      new THREE.LineBasicMaterial({ color: "#2b3a80" })
    );
    edge.position.copy(plane.position);
    world.add(edge);
    const tag = label(`слой ${l + 1}`, "#7f93c9");
    tag.position.set(-width / 2 - 2.4, y, 0);
    world.add(tag);
    for (let i = 0; i < T; i++) {
      const m = new THREE.Mesh(boxGeo, new THREE.MeshBasicMaterial({ color: "#3b5bdb" }));
      m.position.set(i * GAP - width / 2, y, 0);
      m.userData = { i, l };
      world.add(m);
      tokenMeshes.push(m);
      const s = label(data.tokens[i], l === 0 ? "#ffffff" : "#9fb2e6");
      s.position.set(m.position.x, y - 0.95, 0);
      world.add(s);
    }
    const lines = new THREE.LineSegments(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })
    );
    world.add(lines);
    floorLines.push(lines);
  }
  const mid = ((L - 1) * FLOOR) / 2;
  controls.target.set(0, mid, 0);
  // камера отходит ровно настолько, чтобы башня влезла и по ширине, и по высоте (важно для узких экранов)
  const tan = Math.tan((camera.fov * Math.PI) / 360);
  const dist = Math.max((width / 2 + 3) / (tan * camera.aspect), (mid + 4) / tan) * 1.05;
  camera.position.set(dist * 0.3, mid + 3, dist);
  controls.update();
  buildUI(H, L);
  refresh();
}

// вес внимания токена i к токену j в слое l: усреднение голов или одна голова
function weight(l, i, j) {
  const heads = state.data.layers[l];
  if (state.head === "avg") return heads.reduce((s, h) => s + h[i][j], 0) / heads.length;
  return heads[state.head][i][j];
}

// нити: дуга над рядом токенов, высота дуги растёт с расстоянием между токенами
function refresh() {
  const { tokens } = state.data;
  const T = tokens.length, width = (T - 1) * GAP;
  state.data.layers.forEach((_, l) => {
    const lines = floorLines[l];
    lines.visible = state.layer === "all" || state.layer === l;
    if (!lines.visible) return;
    const pos = [], col = [];
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < i; j++) {                       // j == i (себя) не рисуем: нить была бы нулевой длины
        const w = weight(l, i, j);
        if (w < MIN_W) continue;
        const k = state.sel < 0 ? 1 : i === state.sel ? 1.6 : 0.12;   // выбранный токен — ярче, остальные — тише
        const c = ramp(Math.pow(w, 0.6)).multiplyScalar(Math.min(1, (0.25 + w * 1.4) * k));
        const x0 = i * GAP - width / 2, x1 = j * GAP - width / 2;
        const h = 0.4 + (i - j) * 0.32;
        let px = x0, py = l * FLOOR + 0.3;
        for (let s = 1; s <= 14; s++) {
          const t = s / 14, u = 1 - t;
          const x = u * u * x0 + 2 * u * t * ((x0 + x1) / 2) + t * t * x1;
          const y = l * FLOOR + 0.3 + 2 * u * t * h;
          pos.push(px, py, 0, x, y, 0);
          col.push(c.r, c.g, c.b, c.r, c.g, c.b);
          px = x; py = y;
        }
      }
    }
    lines.geometry.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    lines.geometry.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    lines.geometry.computeBoundingSphere();
  });
  tokenMeshes.forEach((m) => {
    const on = m.userData.i === state.sel;
    m.material.color.set(on ? "#ffb457" : m.userData.l === 0 ? "#6aa9ff" : "#3b5bdb");
    m.scale.setScalar(on ? 1.35 : 1);
  });
  document.getElementById("hint").textContent = state.sel < 0
    ? "Нажмите на токен: подсветятся нити, идущие от него на всех этажах."
    : `Токен «${tokens[state.sel]}» (№ ${state.sel + 1}) смотрит только назад: на себя и на предыдущие токены. Нажмите на пустое место, чтобы сбросить.`;
}

// ── интерфейс ──────────────────────────────────────────────────────────────
function pills(host, items, current, onPick) {
  host.replaceChildren();
  items.forEach(([value, text]) => {
    const b = document.createElement("button");
    b.textContent = text;
    b.className = value === current ? "on" : "";
    b.onclick = () => { onPick(value); pills(host, items, value, onPick); };
    host.append(b);
  });
}
function buildUI(H, L) {
  state.head = "avg"; state.layer = "all";
  pills(document.getElementById("heads"), [["avg", "среднее"], ...Array.from({ length: H }, (_, h) => [h, String(h + 1)])], "avg", (v) => { state.head = v; refresh(); });
  pills(document.getElementById("layers"), [["all", "все"], ...Array.from({ length: L }, (_, l) => [l, String(l + 1)])], "all", (v) => { state.layer = v; refresh(); });
}

// ── выбор токена и подсказка ───────────────────────────────────────────────
const ray = new THREE.Raycaster(), ptr = new THREE.Vector2(), tip = document.getElementById("tip");
function pick(e) {
  const r = renderer.domElement.getBoundingClientRect();
  ptr.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ptr, camera);
  return ray.intersectObjects(tokenMeshes)[0]?.object ?? null;
}
let down = null;
renderer.domElement.addEventListener("pointerdown", (e) => (down = { x: e.clientX, y: e.clientY }));
renderer.domElement.addEventListener("pointerup", (e) => {
  if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) return;   // это было вращение, а не клик
  const hit = pick(e);
  state.sel = hit ? (hit.userData.i === state.sel ? -1 : hit.userData.i) : -1;
  refresh();
});
renderer.domElement.addEventListener("pointermove", (e) => {
  const hit = pick(e);
  if (!hit || !state.data) { tip.style.display = "none"; return; }
  const { i, l } = hit.userData;
  const top = [...Array(i).keys()].map((j) => [j, weight(l, i, j)]).sort((a, b) => b[1] - a[1]).slice(0, 3);
  tip.innerHTML = `<b>«${state.data.tokens[i]}»</b> · слой ${l + 1}` + (top.length ? "<br>смотрит на: " + top.map(([j, w]) => `«${state.data.tokens[j]}» ${(w * 100).toFixed(0)}%`).join(", ") : "<br>первый токен: смотрит только на себя");
  tip.style.display = "block";
  tip.style.left = Math.min(e.clientX + 14, innerWidth - 270) + "px";
  tip.style.top = e.clientY + 14 + "px";
});

// ── запуск ──────────────────────────────────────────────────────────────────
function resize() {
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);
resize();
(function loop() { requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();

const sel = document.getElementById("sel");
const index = await (await fetch("data/index.json")).json();
index.forEach((e, k) => sel.append(new Option(e.title, k)));
const load = async (k) => build(await (await fetch("data/" + index[k].file)).json());
sel.onchange = () => load(+sel.value);
const q = new URLSearchParams(location.search).get("example");
if (q && index[+q]) sel.value = q;
await load(+sel.value);
window.viz = { state, refresh }; // для отладки из консоли
