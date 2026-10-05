# Related work

Short list of papers and tools we actually care about. Add a row when we use or reject something. Do not crawl the literature on a timer.

| Work | Link | Why it matters here | What we will not copy |
|------|------|---------------------|------------------------|
| Kumar et al., 2020 — Automated Motion Tracking and Data Extraction for Red Blood Cell Biomechanics | https://doi.org/10.1002/cpcy.75 | High-speed RBC in constricting channels; 8-bit AVI; workflow: background → contrast → channel ROI → segment → track | Image-Pro thresholds; we borrow the **workflow**, not commercial software |
| Keyhole model, 2014 — tracking and deformation through a microstenosis | https://doi.org/10.1080/21681163.2014.957868 | Identity-preserving track + deformation in a contraction; motion-constrained search | MATLAB implementation (not verified as downloadable); not our first experiment |
| RBCZigZagAI (Link et al., 2023) | https://github.com/berndporr/RBCZigZagAI | OpenCV AVI read, background, localized detection | End task is **classification** (native vs modified), not trajectory/speed. Accuracy numbers are not tracking accuracy |
| Fiji TrackMate | https://imagej.net/plugins/trackmate/ | Detection, linking, overlays, manual correction if classical CV is enough | Not required until a simple in-house linker fails |
| RBCdataset | https://github.com/icimrak/RBCdataset | Example annotations (boxes + IDs) for learning the task | Different footage; success there is not success on our AVI |
