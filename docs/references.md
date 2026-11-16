# Sources and provenance

Sources checked on 14 November 2026. No third-party measured data are bundled.

1. UK Power Networks / London Datastore. *SmartMeter Energy Consumption Data in London Households*. https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d . Official description: 5,567 households, November 2011–February 2014, half-hourly kWh, approximately 167 million rows / 10 GB expanded. Separate 2013 dynamic tariff information is available; the package's illustrative tariff is not that calendar. Provider lists Creative Commons Attribution; inspect the downloaded terms and preserve attribution.
2. Hu, Y., Koren, Y., and Volinsky, C. (2008). *Collaborative Filtering for Implicit Feedback Datasets*. IEEE ICDM. DOI: https://doi.org/10.1109/ICDM.2008.22 . Basis of confidence-weighted implicit matrix factorisation; implementation here is a transparent small-data NumPy ALS version.
3. Surprise project and documentation. https://github.com/NicolasHug/Surprise and https://surprise.readthedocs.io/en/stable/ . Rating prediction baselines; Surprise does not support genuine implicit feedback as its general input objective. SVD denotes its regularized latent-factor rating algorithm, not direct truncated decomposition of a zero-filled matrix.
4. REFIT provider record: https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/ . Related dataset paper DOI: https://doi.org/10.1038/sdata.2016.122 . Future extension only.
5. ACN-Data provider: https://ev.caltech.edu/dataset . Dataset paper: *ACN-Data: Analysis and Applications of an Open EV Charging Dataset*, DOI https://doi.org/10.1145/3307772.3328313 . Future extension only.
6. Ausgrid Solar Home Electricity Data provider path: https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data . Future extension only; URL availability may change, and no automatic retrieval is implemented.

## Citation discipline
Cite the original data creators in any paper. Cite actual software versions from the execution manifest. Cite this implementation as software if released, but do not invent a repository URL or DOI. No novelty, significance or real-world savings claim follows from the synthetic demonstration.
