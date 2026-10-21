from pdf2image import convert_from_path

images = convert_from_path(
    "sample_application/passport.pdf",
    dpi=300,
    poppler_path="/opt/homebrew/bin"
)

print("Number of pages:", len(images))

for i, img in enumerate(images):
    print(img.size)
    img.save(f"page_{i}.jpg")