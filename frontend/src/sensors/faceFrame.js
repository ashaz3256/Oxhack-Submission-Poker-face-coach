// Mirror of backend/tilt/face_features.py `frame_primitives`. Keep the two in step:
// backend/tests/test_face_parity.py runs this file under node and compares the numbers.
// Input: MediaPipe FaceMesh landmarks (478 points, normalised {x, y}).
// Output: [yaw, pitch, nose_x, nose_y, gaze_x, gaze_y, eye_open, brow, mouth] (PRIMITIVES order).

export const PRIMITIVES = ['yaw', 'pitch', 'nose_x', 'nose_y', 'gaze_x', 'gaze_y', 'eye_open', 'brow', 'mouth']

const NOSE_TIP = 1
const FOREHEAD = 10
const CHIN = 152
const R_OUTER = 33, R_INNER = 133, R_UPPER = 159, R_LOWER = 145
const L_INNER = 362, L_OUTER = 263, L_UPPER = 386, L_LOWER = 374
const R_BROW = 105, L_BROW = 334
const LIP_UPPER = 13, LIP_LOWER = 14
const IRIS_A = 468, IRIS_B = 473

const div = (a, b) => (Math.abs(b) > 1e-9 ? a / b : NaN)

export function framePrimitives(lm) {
  const x = (i) => lm[i].x
  const y = (i) => lm[i].y
  const faceH = y(CHIN) - y(FOREHEAD)
  const eyeSpan = x(L_OUTER) - x(R_OUTER)
  const midX = (x(R_OUTER) + x(L_OUTER)) / 2
  const midY = (y(R_OUTER) + y(L_OUTER)) / 2

  const rCentre = (x(R_OUTER) + x(R_INNER)) / 2
  const aIsRight = Math.abs(x(IRIS_A) - rCentre) <= Math.abs(x(IRIS_B) - rCentre)
  const rIris = aIsRight ? IRIS_A : IRIS_B
  const lIris = aIsRight ? IRIS_B : IRIS_A
  const gxR = div(x(rIris) - x(R_INNER), x(R_OUTER) - x(R_INNER))
  const gxL = div(x(lIris) - x(L_INNER), x(L_OUTER) - x(L_INNER))
  const gyR = div(y(rIris) - y(R_UPPER), y(R_LOWER) - y(R_UPPER))
  const gyL = div(y(lIris) - y(L_UPPER), y(L_LOWER) - y(L_UPPER))

  return [
    div(x(NOSE_TIP) - midX, eyeSpan),
    div(y(NOSE_TIP) - midY, faceH),
    div(x(NOSE_TIP), eyeSpan),
    div(y(NOSE_TIP), faceH),
    (gxR + (1 - gxL)) / 2,
    (gyR + gyL) / 2,
    div(((y(R_LOWER) - y(R_UPPER)) + (y(L_LOWER) - y(L_UPPER))) / 2, faceH),
    div(((y(R_UPPER) - y(R_BROW)) + (y(L_UPPER) - y(L_BROW))) / 2, faceH),
    div(y(LIP_LOWER) - y(LIP_UPPER), faceH),
  ]
}
