import os
import base64
from collections import deque
from PIL import Image

src_path = r"C:\Users\vinit\.gemini\antigravity-ide\brain\acf34c39-07ca-48af-9634-e8455012d628\.user_uploaded\media_1793034531720.jpg"
im = Image.open(src_path).convert("RGB")

cx, cy = 516, 496
crop_size = 780
half = crop_size // 2

sq = Image.new("RGB", (crop_size, crop_size), (255, 255, 255))
box = (max(0, cx - half), max(0, cy - half), min(im.width, cx + half), min(im.height, cy + half))
cropped = im.crop(box)
paste_x = (crop_size - cropped.width) // 2
paste_y = (crop_size - cropped.height) // 2
sq.paste(cropped, (paste_x, paste_y))

master = sq.resize((512, 512), Image.Resampling.LANCZOS).convert("RGBA")
w, h = master.size
pix = master.load()

visited = set()
q = deque([(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)])
for pt in q:
    visited.add(pt)

while q:
    x, y = q.popleft()
    r, g, b, a = pix[x, y]
    if r > 240 and g > 240 and b > 240:
        pix[x, y] = (255, 255, 255, 0)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                visited.add((nx, ny))
                nr, ng, nb, _ = pix[nx, ny]
                if nr > 235 and ng > 235 and nb > 235:
                    q.append((nx, ny))

out_dir = r"d:\adv_learner_project"

# 1. PWA & Apple icons
master.save(os.path.join(out_dir, "android-chrome-512x512.png"), "PNG")
master.resize((192, 192), Image.Resampling.LANCZOS).save(os.path.join(out_dir, "android-chrome-192x192.png"), "PNG")
master.resize((180, 180), Image.Resampling.LANCZOS).save(os.path.join(out_dir, "apple-touch-icon.png"), "PNG")

# 2. Google Search (48x48 is Google's official baseline, 96x96 is 2x)
master.resize((96, 96), Image.Resampling.LANCZOS).save(os.path.join(out_dir, "favicon-96x96.png"), "PNG")
master.resize((48, 48), Image.Resampling.LANCZOS).save(os.path.join(out_dir, "favicon-48x48.png"), "PNG")
master.resize((32, 32), Image.Resampling.LANCZOS).save(os.path.join(out_dir, "favicon-32x32.png"), "PNG")
master.resize((16, 16), Image.Resampling.LANCZOS).save(os.path.join(out_dir, "favicon-16x16.png"), "PNG")

# 3. Standard favicon.ico with embedded multi-sizes
master.save(
    os.path.join(out_dir, "favicon.ico"),
    format="ICO",
    sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
)

# 4. SVG Favicon
with open(os.path.join(out_dir, "android-chrome-512x512.png"), "rb") as f:
    b64_data = base64.b64encode(f.read()).decode("utf-8")

svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="100%" height="100%">
  <image width="512" height="512" href="data:image/png;base64,{b64_data}"/>
</svg>
"""

with open(os.path.join(out_dir, "favicon.svg"), "w", encoding="utf-8") as f:
    f.write(svg_content)

# 5. site.webmanifest
manifest_content = """{
  "name": "MLM Insights",
  "short_name": "MLM Insights",
  "icons": [
    {
      "src": "/favicon-48x48.png",
      "sizes": "48x48",
      "type": "image/png"
    },
    {
      "src": "/android-chrome-192x192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "/android-chrome-512x512.png",
      "sizes": "512x512",
      "type": "image/png"
    }
  ],
  "theme_color": "#0B1F3A",
  "background_color": "#FFFFFF",
  "display": "standalone"
}
"""
with open(os.path.join(out_dir, "site.webmanifest"), "w", encoding="utf-8") as f:
    f.write(manifest_content)

print("Generated all favicon files successfully!")
