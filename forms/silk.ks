# silk: four hundred strands, each a slow wave, braided by time
@points 20m
@palette mono
@zoom 1.35
param strands = 400
param k = 3
let s = floor(u * strands) / strands
let v = fract(u * strands) * tau
let w = sin(v * k + s * 6 + t * 0.4)
let r = 0.5 + 0.22 * w * cos(s * tau * 2 - t * 0.3)
x = r * cos(v + 0.6 * s + 0.15 * sin(t * 0.2)) * 1.25
y = r * sin(v * 2 + s * 3) * 0.6 + 0.08 * sin(s * tau * 3 + t)
