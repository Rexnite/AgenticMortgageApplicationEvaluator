from pathlib import Path


class FileReader:

    @staticmethod
    def get_documents(folder: Path):

        supported = [
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg"
        ]

        files = []

        for file in folder.iterdir():

            if file.suffix.lower() in supported:

                files.append(file)

        return files