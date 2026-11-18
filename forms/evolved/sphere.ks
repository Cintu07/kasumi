# knot: a torus knot made of rings, with depth of field from z
@points 30m
@palette mono
@dof 0.06
@focus 0
param p = 3.624
param q = 7
let s = floor(u * 1500) / 1500
let v = fract(u * 1500) * tau
let a = s * tau + t * 0.05
let rr = 0.55 + 0.22 * cos(q * a)
let cx = rr * cos(p * a)
let cy = rr * sin(p * a)
let cz = 0.22 * sin(q * a)
let ring = 0.05 + 0.02 * (t * 0.05)
let x3 = cx + ring * (cos(v) + 0.2665 * sin(3.55 * ring + -0.5359 * t)) * cos(p * a)
let y3 = cy + ring * cos(v) * sin(p * a)
let z3 = cz + ring * sin(v)
let ang = u * 1500
x = x3 * cos(ang) - z3 * sin(ang)
y = y3 * 0.9 + z3 * (0.25 + -0.1439 * cos(3.042 * ang + 0.3242 * t))
z = x3 * sin(ang) + z3 * cos(ang)
h = s
