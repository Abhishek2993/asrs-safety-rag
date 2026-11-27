# data/sample

`asrs_sample.csv` holds 200 de-identified NASA ASRS reports (2012-2022) in the
ASRS Database Online two-row-header export format, so the project runs without
any download.

* **Source:** NASA Aviation Safety Reporting System (ASRS) Database Online,
  https://asrs.arc.nasa.gov/, retrieved through the public mirror
  `elihoole/asrs-aviation-reports` on Hugging Face.
* **Selection:** `python scripts/fetch_asrs_mirror.py --splits validation test --sample 200`
  (seed 13), stratified so every contributing-factor label is represented.
* **Use:** ASRS data are public and de-identified by NASA. Reports are
  voluntary, unverified first-person accounts and are **not** a statistically
  representative sample of aviation events. Do not attempt to re-identify
  reporters or use the data to assign blame. Please credit NASA ASRS when
  reusing it.
