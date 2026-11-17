# tiny: the tiny-code sketch style. in a browser this is a few thousand dots
# placed by the integer index; here every dot is a smear of a thousand.
# i % 240 walks along a row, floor(i / 240) picks the row
@points 12m
@palette mono
@gamma 2.4
@zoom 1.1
let k = i % 240
let row = floor(i / 240) % 64
let e = r1 * 0.02
let q = k / 240 * tau
let ring = row / 64
let d = 0.25 + 0.55 * ring + 0.06 * sin(q * 5 + ring * 9 - t * 0.6)
let a = q + ring * 1.7 + 0.25 * sin(ring * tau * 2 + t * 0.3) + e
x = cos(a) * d * (1 + 0.2 * cos(q * 3 + t * 0.4))
y = sin(a) * d * 0.85
b = 0.4 + ring
