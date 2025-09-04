import shutil
import zipfile
import requests
from tqdm import tqdm
from pathlib import Path

def download_and_unzip(url, zip_path, extract_to):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total = int(response.headers.get('content-length', 0))
    with open(zip_path, 'wb') as f, tqdm(
        desc=f"Downloading: {zip_path.name}",
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)

    print(f"Infalting: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid zip file!")
        return

    print(f"Removing zip: {zip_path}")
    zip_path.unlink()

if __name__ == "__main__":
    # Set up directories
    scenario_name = 'DT1'
    scenario_dir = Path(f"scenarios/{scenario_name}")
    scenario_dir.mkdir(parents=True, exist_ok=True)

    # Download and extract wireless data
    print("Preparing wireless data...")
    download_and_unzip(
        "https://www.dropbox.com/scl/fi/esl0jd1idkwarmk9sr1xz/wireless.zip?rlkey=qhwdiu07oasfn1h9pa5xfjv0i&st=pjgypntm&dl=1",
        scenario_dir / "wireless.zip",
        scenario_dir
    )

    # Download and extract parameter files
    print("Preparing parameter files...")
    param_dir = scenario_dir / "param"
    param_dir.mkdir(parents=True, exist_ok=True)
    download_and_unzip(
        "https://www.dropbox.com/scl/fo/9sfd6u8912l7o407fqi30/AN2NIxPUrXvMEVjImsHmX2g?rlkey=qqxzkohhnmgjgz2abf6cvb32h&st=0kbpp7v4&dl=1",
        scenario_dir / "param.zip",
        param_dir
    )

    # Copy wireless params.mat file to wireless folder
    wireless_dir = scenario_dir / "wireless"
    shutil.copy(param_dir / "params.mat", wireless_dir / "params.mat")

    print(f"DeepVerse scenario {scenario_name} is ready!")