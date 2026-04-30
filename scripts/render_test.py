import cairosvg
svg = '<svg width="100" height="100"><circle cx="50" cy="50" r="40" fill="red"/></svg>'
cairosvg.svg2png(bytestring=svg.encode(), write_to="data/stats/examples/test_render.png")
print("Cairo rendering works!")
