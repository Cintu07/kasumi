# ink: scattered points pushed along a noise field. diffuse, no hot spots,
# the best case for naive atomics
@points 40m
@palette ink
let x0 = r1 * 3.6 - 1.8
let y0 = r2 * 2 - 1
let f = noise(x0 * 1.3, y0 * 1.3 + t * 0.07) * tau
let g = noise(x0 * 3.1 + 7, y0 * 3.1 - t * 0.05)
x = x0 + cos(f) * 0.12 + g * 0.03
y = y0 + sin(f) * 0.12
b = smoothstep(-0.2, 0.9, g)
h = fract(f * 0.16)
