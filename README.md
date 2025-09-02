# Fermi surfaces plots and Hall conductivity

Scripts to generate figures and analysis artifacts for the manuscript:

**“Emergence of _p_-wave collinear magnetism in antiferromagnets with reflection-asymmetric magnetic motifs.”**

---
## How to cite

If these scripts support a publication, please cite the associated manuscript:

> _Emergence of p‑wave collinear magnetism in antiferromagnets with reflection‑asymmetric magnetic motifs_ (To be published, 2025).

Optionally include a software citation (commit hash + repository URL).

This repository contains two standalone Python scripts used to produce (i) high‑quality Fermi‑surface/contour visualizations and (ii) anomalous Hall conductivity (AHE) plots for antiferromagnets with magnetic motifs connected by mirror symmetry. Both scripts are self‑contained and rely only on common scientific Python packages.

---

## Repository contents

- `fermi_contours_fullHQ.py` — plots Fermi contours/constant‑energy cuts and related band-structure views at high resolution for publication figures.
- `ahe_flexible_plots_smoothed.py` — computes and plots anomalous Hall conductivity curves with optional smoothing and flexible styling.
- `Figures/` (optional) — recommended output folder for generated images (PDF/PNG/SVG).
- `data/` (optional) — place any inputs here if your workflow requires external numeric data.

---

## Requirements

- Python ≥ 3.9
- Recommended packages:
  - `numpy`
  - `matplotlib`
  - `scipy` (if smoothing/filters are used)
  - `pandas` (only if your local workflow loads tabular inputs)
  
Install basics with:
```bash
pip install numpy matplotlib scipy pandas
```

> If a `requirements.txt` is later added, prefer:
> ```bash
> pip install -r requirements.txt
> ```

---

## Quick start

1. **Clone** or download this repository.
2. (Optional) **Create outputs folder**:
   ```bash
   mkdir -p Figures
   ```
3. **Run** a script with default settings:
   ```bash
   python fermi_contours_fullHQ.py
   python ahe_flexible_plots_smoothed.py
   ```

Each script writes plots to the current directory unless an output path is provided (see `-h`).

---

## Usage

Both scripts provide command‑line help. Run:
```bash
python fermi_contours_fullHQ.py -h
python ahe_flexible_plots_smoothed.py -h
```

Typical usage patterns (your exact options may differ—use `-h` for the authoritative list):

### Fermi contours
```bash
python fermi_contours_fullHQ.py   --mu 0.0   --resolution 1200   --outfile Figures/fermi_contours.pdf
```
**What it does**
- Generates high‑quality constant‑energy (Fermi) contours.
- Offers figure‑quality controls (e.g., DPI/resolution, line weight, labels).
- Designed for reproducible, camera‑ready figures.

**Outputs**
- Vector or raster images (`.pdf`, `.png`, or `.svg`) saved to the specified path.

### Anomalous Hall conductivity (AHE)
```bash
python ahe_flexible_plots_smoothed.py   --smooth   --window 15   --outfile Figures/ahe_curves.pdf
```
**What it does**
- Computes/plots AHE curves and derived quantities.
- Optional smoothing/denoising and flexible plot styling.
- Suitable for side‑by‑side comparisons across parameter sets.

**Outputs**
- Publication‑ready plots saved in the chosen format.

> **Tip**: Combine with versioned inputs (e.g., CSV/JSON) to guarantee reproducibility. If your workflow reads external data, place it under `data/` and point the corresponding CLI option to the file.

---

## Reproducibility checklist

- Fixed random seeds if randomness is introduced (N/A by default).
- Specify the exact commit hash in your manuscript or lab notes.
- Preserve the command lines (including all flags and inputs) used to generate each figure.
- Record the Python and package versions:
  ```bash
  python -V
  python -c "import numpy, matplotlib, scipy, pandas; print(numpy.__version__, matplotlib.__version__, scipy.__version__, pandas.__version__)"
  ```

---

## Troubleshooting

- **Blank or empty plots**: Check that the Fermi level/energy window is within your data range and that the input path (if any) is correct.
- **Fonts/overlaps in final PDF**: Increase figure size or font size via the CLI; export as SVG if you need post‑processing in vector editors.
- **Performance**: Lower the resolution or sub‑sampling; close other matplotlib windows to free memory.
- **Backend issues**: If running on a headless server, set `MPLBACKEND=Agg`:
  ```bash
  MPLBACKEND=Agg python fermi_contours_fullHQ.py
  ```

---

## License

Specify your preferred license (e.g., MIT, BSD‑3‑Clause, or CC‑BY‑4.0 for figure scripts). If absent, the default is “all rights reserved.”

---

## Contributing

Pull requests for minor fixes (typos, docstrings, CLI help) are welcome. For larger changes, please open an issue first to discuss scope and compatibility with the figures in the manuscript.

---

## Acknowledgments

If you use these scripts in your work, consider acknowledging the repository and the paper.
