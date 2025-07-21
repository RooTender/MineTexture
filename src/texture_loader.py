import requests
import io
import zipfile
import os

class TextureUtils:
    def extract(self, path_or_bytes, dest: str):
        with zipfile.ZipFile(path_or_bytes) as zip:
            for member in zip.namelist():
                if member.endswith(".png"):
                    dest_dir = os.path.join(dest, os.path.dirname(member))
                    os.makedirs(dest_dir, exist_ok=True)

                    with zip.open(member) as source, open(
                        os.path.join(dest_dir, os.path.basename(member)), "wb"
                    ) as target:
                        target.write(source.read())


class VanillaTexture(TextureUtils):

    def __init__(self) -> None:
        self.storage_dir = os.path.join('data', 'vanilla')
        os.makedirs(self.storage_dir, exist_ok=True)

        manifest_url = "https://launchermeta.mojang.com/mc/game/version_manifest.json"
        resp = requests.get(manifest_url)
        resp.raise_for_status()

        self.manifest = resp.json()

    def get_versions(self):
        return [item["id"] for item in self.manifest["versions"]]

    def download(self, version):
        url = next(v["url"] for v in self.manifest["versions"] if v["id"] == version)
        resp = requests.get(url)
        resp.raise_for_status()

        version_data = resp.json()
        client_url = version_data["downloads"]["client"]["url"]

        resp = requests.get(client_url)
        resp.raise_for_status()
        jar_bytes = io.BytesIO(resp.content)

        download_dir = os.path.join(self.storage_dir, version)
        try:
            os.makedirs(download_dir)
        except OSError:
            os.removedirs(download_dir)
            os.makedirs(download_dir)

        self.extract(jar_bytes, download_dir)

class StyledTexture:
    def __init__(self, style_name: str) -> None:
        self.storage_dir = os.path.join('data', style_name)
        os.makedirs(self.storage_dir, exist_ok=False)
