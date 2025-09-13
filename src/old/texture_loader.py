import requests
import io
import zipfile
import os

from pathlib import Path
import shutil
from PIL import Image

class TextureUtils:

    def extract(self, path_or_bytes, dest: str):
        with zipfile.ZipFile(path_or_bytes) as zip:
            for member in zip.namelist():

                if not os.path.join("minecraft", "textures") in member:
                    continue

                if member.endswith(".png") or member.endswith(".mcmeta"):
                    dest_dir = os.path.join(dest, os.path.dirname(member))
                    os.makedirs(dest_dir, exist_ok=True)

                    with zip.open(member) as source, open(
                        os.path.join(dest_dir, os.path.basename(member)), "wb"
                    ) as target:
                        target.write(source.read())

    def decompose_animations(self, root_dir: str):
        for dirpath, _, files in os.walk(root_dir):
            anim_definitions = [file for file in files if file.endswith(".mcmeta")]

            for definition in anim_definitions:
                img_name = Path(definition.removesuffix(".mcmeta")).stem

                img_path = os.path.join(dirpath, f"{img_name}.png")
                meta_path = os.path.join(dirpath, definition)

                os.remove(meta_path)

                try:
                    img = Image.open(img_path)
                except OSError:
                    continue

                width, height = img.size

                if width != height:
                    frame_size = min(width, height)
                    frames = max(width, height) // frame_size

                    for i in range(frames):
                        if width > height:
                            frame = img.crop((i * frame_size, 0, (i + 1) * frame_size, frame_size))
                        else:
                            frame = img.crop((0, i * frame_size, frame_size, (i + 1) * frame_size))
                        
                        frame.save(os.path.join(dirpath, f"{img_name}_{i}.png"))

                    img.close()

                    os.remove(img_path)


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

    def setup(self, version):
        url = next(v["url"] for v in self.manifest["versions"] if v["id"] == version)
        resp = requests.get(url)
        resp.raise_for_status()

        version_data = resp.json()
        client_url = version_data["downloads"]["client"]["url"]

        resp = requests.get(client_url)
        resp.raise_for_status()
        jar_bytes = io.BytesIO(resp.content)

        download_dir = os.path.join(self.storage_dir, version)

        if os.path.exists(download_dir):
            shutil.rmtree(download_dir)
       
        os.makedirs(download_dir)

        self.extract(jar_bytes, download_dir)
        self.decompose_animations(download_dir)

class StyledTexture(TextureUtils):

    def __init__(self) -> None:
        self.storage_dir = os.path.join('data', 'styled')
        os.makedirs(self.storage_dir, exist_ok=True)


    def setup(self, path_to_zip: str):
        style_name = Path(path_to_zip).stem
        dest_dir = os.path.join(self.storage_dir, style_name)

        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)

        self.extract(path_to_zip, dest_dir)
        self.decompose_animations(dest_dir)
