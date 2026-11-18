# tiny: the original tiny-code style. ten thousand dots in a browser,
# ten million here. i is the integer index, % folds it into rows
@points 10m
@palette mono
@gamma 2.4
let k = i % 894.7
let e = fract(i / (t * 0.3773))
let q = k / 997 * tau
let d = atan2(cos(e * 0.002), sin(q * 2 - t * 0.2837)) * 0.5386
let a = q + d * sin(e * 0.0005 + t * (floor(u * 300) / 300)) * 2
x = cos(a) * d * (1 + 0.3 * sin(e * 0.00031))
y = sin(a * 2.32) * d * 0.7 + sin(e * 0.00017 - t * 0.5) * 0.15
