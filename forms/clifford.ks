# clifford: a strange attractor. px / py are the previous point, so every
# gpu thread walks its own chaotic orbit; the params drift slowly with time
@points 40m
@iters 200
@palette ember
@zoom 0.4
@gamma 2.6
let a = -1.7 + 0.12 * sin(t * 0.21)
let e = 1.8 + 0.1 * cos(t * 0.17)
let c = -0.9
let d = -1.2 + 0.08 * sin(t * 0.13)
x = sin(a * py) + c * cos(a * px)
y = sin(e * px) + d * cos(e * py)
h = smoothstep(0, 1.6, mag(x - px, y - py))
