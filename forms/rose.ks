# rose: three hundred petal strands that all cross the centre, so millions
# of points pile onto the same few pixels. the worst case for naive atomics
@points 40m
@palette sakura
param k = 5
let s = floor(u * 300) / 300
let v = fract(u * 300) * tau
let r = sin(k * v + s * 0.9 + t * 0.2) * (0.8 + 0.2 * cos(s * tau * 3 + t * 0.3))
x = r * cos(v + s * 0.35)
y = r * sin(v + s * 0.35)
h = s
