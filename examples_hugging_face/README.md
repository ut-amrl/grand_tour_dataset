# 🔧 Additional Installations

Some scripts require extra dependencies beyond the base installation.

---

## 🖼 Dynamic Points Filtering & Nerfstudio Conversion

For:

* `dynamic_points_filtering_using_images.py`
* `nerfstudio_convert.py`

These scripts require **PyTorch** and the **Transformers** library:

```bash
# Assumes you have the grand_tour venv sourced.
uv pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers
```

---

## 🗺 Elevation Map Generation

For:

* `generate_elevation_maps.py`

We provide a Python-only installable version of **Elevation Mapping** (medium-tested, but stable enough for use).

```bash
# Assumes you have the grand_tour venv sourced.
cd ~/git
git clone git@github.com:leggedrobotics/elevation_mapping_cupy.git -b dev/python_library_installation
cd elevation_mapping_cupy

# Install dependencies
uv pip install -r requirements.txt

# Install CuPy (adapt the CUDA version as needed)
uv pip install cupy-cuda12x

# Install PyTorch (adapt the CUDA/Python version if needed)
uv pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121
```