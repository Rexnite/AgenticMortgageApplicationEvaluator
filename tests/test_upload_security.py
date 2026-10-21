import io
import json

import pytest
from PIL import Image

from models.document import Document
from ui_helpers import UploadValidationError, result_export, validate_uploads
from workflow.state import MortgageState


class Upload:
    def __init__(self, name, data):
        self.name, self._data = name, data
    def getvalue(self):
        return self._data


def png_bytes():
    output = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(output, format="PNG")
    return output.getvalue()


def test_upload_validation_accepts_image_and_ignores_path_components():
    validated = validate_uploads([Upload("../../passport.png", png_bytes())])
    assert validated[0][0] == ".png"


@pytest.mark.parametrize("upload,message", [
    (Upload("empty.pdf", b""), "empty"),
    (Upload("fake.pdf", b"not a pdf"), "corrupt"),
    (Upload("script.exe", b"x"), "unsupported"),
])
def test_invalid_uploads_are_rejected_without_exposing_paths(upload, message):
    with pytest.raises(UploadValidationError, match=message):
        validate_uploads([upload])


def test_export_excludes_document_paths_and_raw_text():
    state = MortgageState(documents=[Document(file_name="random.pdf",
        file_path="C:/private/upload.pdf", extension=".pdf", size=10,
        extracted_text="private applicant text")])
    exported = result_export(state)
    payload = json.loads(exported)
    assert "file_path" not in payload["documents"][0]
    assert "extracted_text" not in payload["documents"][0]
    assert "private applicant text" not in exported
