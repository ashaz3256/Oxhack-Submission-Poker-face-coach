// Serve MediaPipe's WebAssembly from our own origin so the face scanner works offline.
import { cpSync, existsSync, mkdirSync } from 'node:fs'

const src = 'node_modules/@mediapipe/tasks-vision/wasm'
if (existsSync(src)) {
  mkdirSync('public/mediapipe', { recursive: true })
  cpSync(src, 'public/mediapipe', { recursive: true })
}
